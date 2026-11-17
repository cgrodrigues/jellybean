# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Runtime tests with FakeLLM: loader, registry, loop, delegation and context isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jellybean.runtime.agent_runner import AgentRunner, Runtime
from jellybean.runtime.context import RunContext
from jellybean.runtime.llm_client import FakeLLM, LLMResponse, ToolCall, call, last_tool_result
from jellybean.runtime.skill_loader import SkillFormatError, discover_skills, parse_skill_file
from jellybean.runtime.tool_registry import ToolRegistryError
from jellybean.skills._shared import data_tools

FINISH = call("finish", status="done", summary="all good", artifact_ids=[], issues=[])


def _runtime(ctx: RunContext, script: dict, **kwargs) -> Runtime:
    return Runtime(ctx=ctx, llm=FakeLLM(script), **kwargs)


def _trace(ctx: RunContext) -> list[dict]:
    return [json.loads(line) for line in ctx.trace_path.read_text().splitlines()]


# --------------------------------------------------------------------------- loader
def _write_skill(root: Path, name: str, text: str) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "SKILL.md").write_text(text)


def test_loader_reads_front_matter_and_ignores_shared(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "alpha",
        "---\nname: alpha\ndescription: d\nmax_iterations: 3\nshared_tools: [get_spec]\n---\nYou are alpha.",
    )
    _write_skill(tmp_path, "_shared", "---\nname: _shared\ndescription: x\n---\nignored")
    (tmp_path / "no_skill_md").mkdir()
    skills = discover_skills(tmp_path, package="pkg")
    assert list(skills) == ["alpha"]
    alpha = skills["alpha"]
    assert (alpha.max_iterations, alpha.shared_tools, alpha.system_prompt) == (3, ("get_spec",), "You are alpha.")
    assert alpha.module == "pkg.alpha.tools"


@pytest.mark.parametrize(
    "text",
    [
        "no front matter",
        "---\nname: x\n---\nbody",
        "---\nname: beta\ndescription: d\nextra: 1\n---\nbody",
        "---\nname: beta\ndescription: d\n---\n   ",
    ],
)
def test_loader_rejects_malformed_skill(tmp_path: Path, text: str) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text(text)
    with pytest.raises(SkillFormatError):
        parse_skill_file(path)


def test_loader_requires_matching_name(tmp_path: Path) -> None:
    _write_skill(tmp_path, "gamma", "---\nname: other\ndescription: d\n---\nbody")
    with pytest.raises(SkillFormatError, match="must match"):
        discover_skills(tmp_path)


def test_real_skills_load() -> None:
    skills = discover_skills()
    assert {"orquestrador", "profiler"} <= set(skills)


# ------------------------------------------------------------------------- registry
def test_registry_scopes_tools_per_agent(run_ctx: RunContext) -> None:
    rt = _runtime(run_ctx, {})
    profiler = rt.registry.tools_for("profiler")
    orchestrator = rt.registry.tools_for("orquestrador")
    assert "profile_table" in profiler and "profile_table" not in orchestrator
    assert "delegate" in orchestrator and "delegate" not in profiler
    assert "run_sql" in profiler and "run_sql" not in orchestrator  # not declared by the orchestrator
    with pytest.raises(ToolRegistryError):
        rt.registry.register_skill("bad", data_tools, ["does_not_exist"])


# ----------------------------------------------------------------------------- loop
def test_loop_runs_tools_and_finishes(run_ctx: RunContext) -> None:
    rt = _runtime(
        run_ctx,
        {
            "profiler": [
                call("list_datasets"),
                call("finish", status="done", summary="listed", artifact_ids=["x"], issues=["i1"]),
            ]
        },
    )
    result = rt.run_agent("profiler", "list them")
    assert (result.status, result.summary, result.artifact_ids, result.issues) == ("done", "listed", ["x"], ["i1"])
    assert result.iterations == 2 and result.input_tokens == 200
    kinds = [e["kind"] for e in _trace(run_ctx)]
    assert kinds.count("llm_call") == 2 and "tool_call" in kinds and kinds[-1] == "agent_end"
    tool_event = next(e for e in _trace(run_ctx) if e["kind"] == "tool_call")
    assert tool_event["agent"] == "profiler" and tool_event["tool"] == "list_datasets"


def test_tool_results_reach_the_model(run_ctx: RunContext) -> None:
    llm = FakeLLM({"profiler": [call("list_datasets"), FINISH]})
    Runtime(ctx=run_ctx, llm=llm).run_agent("profiler", "t")
    second = llm.calls[1].messages
    assert second[-1].role == "tool"
    assert {d["name"] for d in json.loads(second[-1].content)["summary"]["datasets"]} == {"events", "customers", "wide"}


def test_unknown_tool_and_bad_finish_are_recoverable(run_ctx: RunContext) -> None:
    llm = FakeLLM({"profiler": [call("hack_the_planet"), call("finish", status="bogus", summary="x"), FINISH]})
    result = Runtime(ctx=run_ctx, llm=llm).run_agent("profiler", "t")
    assert result.status == "done"
    assert "Unknown tool" in llm.calls[1].messages[-1].content
    assert "Invalid finish arguments" in llm.calls[2].messages[-1].content


def test_text_only_reply_is_nudged_then_accepted(run_ctx: RunContext) -> None:
    script = {"profiler": [LLMResponse(text="thinking"), LLMResponse(text="final words")]}
    llm = FakeLLM(script)
    result = Runtime(ctx=run_ctx, llm=llm).run_agent("profiler", "t")
    assert "finish" in llm.calls[1].messages[-1].content
    assert result.status == "done" and result.summary == "final words"
    assert "agent ended without calling finish" in result.issues


def test_max_iterations(run_ctx: RunContext) -> None:
    rt = _runtime(run_ctx, {"profiler": [call("list_datasets") for _ in range(80)]})
    result = rt.run_agent("profiler", "t")
    assert result.status == "failed" and "max_iterations" in result.issues
    assert result.iterations == rt.skills["profiler"].max_iterations


def test_token_budget_stops_cleanly(run_ctx: RunContext) -> None:
    rt = _runtime(run_ctx, {"profiler": [call("list_datasets") for _ in range(10)]})
    rt.budget.limit = 250  # FakeLLM reports 120 tokens per call
    result = rt.run_agent("profiler", "t")
    assert result.status == "failed" and result.issues == ["token_budget_exhausted"]
    assert result.iterations == 3


def test_llm_errors_fail_the_agent_not_the_process(run_ctx: RunContext) -> None:
    def explode(messages):
        raise ConnectionError("network down")

    result = _runtime(run_ctx, {"profiler": [explode]}).run_agent("profiler", "t")
    assert result.status == "failed" and "network down" in result.summary
    events = [json.loads(line) for line in run_ctx.trace_path.read_text(encoding="utf-8").splitlines()]
    assert any(e["kind"] == "llm_error" and "network down" in e["error"] for e in events)
    assert events[-1]["kind"] == "agent_end" and "network down" in events[-1]["reason"]
    assert "ConnectionError" in run_ctx.errors_path.read_text(encoding="utf-8")


def test_unknown_tool_and_delegate_failures_are_traced(run_ctx: RunContext) -> None:
    script = {
        "orquestrador": [
            call("no_such_tool"),
            call("delegate", agent="ghost", task="x"),
            call("finish", status="done", summary="ok"),
        ]
    }
    _runtime(run_ctx, script).run_agent("orquestrador", "t")
    events = [json.loads(line) for line in run_ctx.trace_path.read_text(encoding="utf-8").splitlines()]
    errors = {e["tool"]: e["error"] for e in events if e["kind"] == "tool_call" and e["status"] == "error"}
    assert "Unknown tool" in errors["no_such_tool"] and "Unknown agent" in errors["delegate"]


# ----------------------------------------------------------------- delegation/isolation
def test_delegation_isolates_context(run_ctx: RunContext) -> None:
    secret = "ORCHESTRATOR-PRIVATE-REASONING"
    script = {
        "orquestrador": [
            LLMResponse(text=secret, tool_calls=(ToolCall("c1", "delegate", {"agent": "profiler", "task": "do it"}),)),
            lambda messages: call("finish", status="done", summary=last_tool_result(messages)["summary"]["summary"]),
        ],
        "profiler": [call("finish", status="done", summary="profiled", artifact_ids=[])],
    }
    llm = FakeLLM(script)
    result = Runtime(ctx=run_ctx, llm=llm).run_agent("orquestrador", "analyse events")
    assert result.summary == "profiled"
    profiler_call = next(c for c in llm.calls if c.agent == "profiler")
    # Clean context: one user message with spec summary + task, nothing from the orchestrator.
    assert len(profiler_call.messages) == 1
    first = profiler_call.messages[0].content
    assert "do it" in first and secret not in first and "analyse events" not in first
    assert "You are the **profiler**" in profiler_call.system
    assert "delegate" not in profiler_call.tool_names
    orchestrator_call = next(c for c in llm.calls if c.agent == "orquestrador")
    assert "Agents you can delegate to" in orchestrator_call.messages[0].content


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"agent": "orquestrador", "task": "x"}, "itself"),
        ({"agent": "nobody", "task": "x"}, "Unknown agent"),
        ({"agent": "profiler", "task": "x", "artifact_ids": ["ghost_v1"]}, "Unknown artifact"),
        ({"agent": "profiler", "task": "x" * 2001}, "too long"),
    ],
)
def test_delegate_validation(run_ctx: RunContext, arguments: dict, message: str) -> None:
    llm = FakeLLM({"orquestrador": [call("delegate", **arguments), FINISH]})
    Runtime(ctx=run_ctx, llm=llm).run_agent("orquestrador", "t")
    assert message in llm.calls[1].messages[-1].content


def test_critic_receives_only_the_finding_id(run_ctx: RunContext) -> None:
    """The critic skill arrives in phase 5; a stand-in skill checks the delegate rule now."""
    from jellybean.runtime.skill_loader import SkillDefinition

    skills = discover_skills()
    skills["critico"] = SkillDefinition(
        name="critico",
        description="critic",
        system_prompt="You are the critic.",
        directory=Path("."),
        module="jellybean.skills._shared.data_tools",
    )
    llm = FakeLLM(
        {
            "orquestrador": [
                call("delegate", agent="critico", task="F001 looks like fraud because..."),
                call("delegate", agent="critico", task="F001"),
                FINISH,
            ],
            "critico": [FINISH],
        }
    )
    Runtime(ctx=run_ctx, llm=llm, skills=skills).run_agent("orquestrador", "t")
    assert "only the finding id" in llm.calls[1].messages[-1].content
    critic_call = next(c for c in llm.calls if c.agent == "critico")
    assert "Try to refute finding F001." in critic_call.messages[0].content
    assert "fraud" not in critic_call.messages[0].content


def test_each_agent_run_starts_fresh(run_ctx: RunContext) -> None:
    llm = FakeLLM({"profiler": [call("list_datasets"), FINISH, FINISH]})
    rt = Runtime(ctx=run_ctx, llm=llm)
    rt.run_agent("profiler", "first")
    rt.run_agent("profiler", "second")
    assert len(llm.calls[2].messages) == 1 and "second" in llm.calls[2].messages[0].content


def test_agent_runner_uses_skill_iteration_limit(run_ctx: RunContext) -> None:
    rt = _runtime(run_ctx, {})
    assert AgentRunner(rt, rt.skills["profiler"]).max_iterations == 60  # profiler/SKILL.md


def test_agent_result_lists_its_outputs(run_ctx: RunContext) -> None:
    llm = FakeLLM({"profiler": [call("run_sql", query="SELECT 1 AS x", save_as="tiny"), FINISH]})
    result = Runtime(ctx=run_ctx, llm=llm).run_agent("profiler", "t")
    assert result.outputs == [
        {"id": "tiny_v1", "type": "table", "rows": 1, "path": "artifacts/tiny_v1.parquet", "tool": "run_sql"}
    ]


def test_delegate_returns_output_ids_without_paths(run_ctx: RunContext) -> None:
    llm = FakeLLM(
        {
            "orquestrador": [call("delegate", agent="profiler", task="make a table"), FINISH],
            "profiler": [call("run_sql", query="SELECT 1 AS x", save_as="tiny"), FINISH],
        }
    )
    Runtime(ctx=run_ctx, llm=llm).run_agent("orquestrador", "t")
    second_turn = [c for c in llm.calls if c.agent == "orquestrador"][1]
    delegated = json.loads(second_turn.messages[-1].content)["summary"]
    assert delegated["outputs"] == [{"id": "tiny_v1", "type": "table", "rows": 1}]
    assert "artifacts/" not in second_turn.messages[-1].content  # no paths reach the LLM


def test_agents_receive_the_context_notes(run_ctx: RunContext) -> None:
    import yaml

    spec = {
        "run_id": run_ctx.run_id,
        "instruction": "x",
        "seed": 7,
        "context_notes": "amount is in EUR; phone is the subscriber number",
    }
    run_ctx.spec_path.write_text(yaml.safe_dump(spec))
    llm = FakeLLM({"critico": [FINISH]})
    from jellybean.runtime.skill_loader import discover_skills

    Runtime(ctx=run_ctx, llm=llm, skills=discover_skills()).run_agent("critico", "Try to refute finding F001.")
    first = llm.calls[0]
    assert "amount is in EUR" in first.messages[0].content  # spec summary at the start of the task
    assert "# Context notes" in first.system  # the skill says how to use them


# ------------------------------------------------------- repeated timed-out calls
def test_a_call_that_timed_out_is_not_executed_again(run_ctx: RunContext) -> None:
    run_ctx.settings.config.limits.tool_timeouts = {"list_datasets": 0.0001}
    script = {
        "profiler": [
            call("list_datasets"),
            call("list_datasets"),  # identical: refused without running
            call("get_schema", dataset_or_artifact="events"),  # another call still runs
            FINISH,
        ]
    }
    llm = FakeLLM(script)
    result = Runtime(ctx=run_ctx, llm=llm).run_agent("profiler", "t")
    assert result.status == "done"
    first, second = llm.calls[1].messages[-1].content, llm.calls[2].messages[-1].content
    assert "timed out" in first and '"timed_out":true' in first.replace(" ", "")
    assert "Not run" in second and "Change the arguments" in second
    events = [json.loads(line) for line in run_ctx.trace_path.read_text(encoding="utf-8").splitlines()]
    calls = [e for e in events if e["kind"] == "tool_call" and e["tool"] == "list_datasets"]
    assert len(calls) == 2 and calls[1]["duration_s"] == 0.0  # the second one never reached the tool
    assert '"status":"ok"' in llm.calls[3].messages[-1].content.replace(" ", "")
