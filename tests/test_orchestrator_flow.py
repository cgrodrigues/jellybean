"""Orchestrator tools, human control point 1 and a scripted end-to-end run (FakeLLM)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext, use_run
from jellybean.runtime.human import AutoApproveHuman, ConsoleHuman, ScriptedHuman
from jellybean.runtime.llm_client import FakeLLM, LLMResponse, Message, ToolCall, call, last_tool_result
from jellybean.spec import load_spec
from tests.conftest import dummy_settings

FINISH = call("finish", status="done", summary="ok")


def _artifact(messages: list[Message], tool: str) -> str:
    """Artifact id returned by the latest call of ``tool`` in the conversation."""
    names = {}
    for m in messages:
        for c in m.tool_calls:
            names[c.id] = c.name
    for m in reversed(messages):
        if m.role == "tool" and names.get(m.tool_call_id) == tool:
            return json.loads(m.content)["artifact_id"]
    raise AssertionError(f"no result for {tool}")


def _orchestrator_script(primary: str, enrichment: str) -> list:
    """Orchestrator steps of a normal run up to control 1."""
    return [
        call("init_run", instruction=f"analyse {primary} enriched with {enrichment}", report_language="en"),
        call("resolve_dataset", name=primary),
        call("resolve_dataset", name=enrichment),
        lambda m: call(
            "update_spec",
            patch={
                "datasets": [
                    {"name": primary, "path": f"./data/{primary}", "role": "primary"},
                    {"name": enrichment, "path": f"./data/{enrichment}", "role": "enrichment"},
                ]
            },
        ),
        call("delegate", agent="profiler", task="Profile the datasets in the spec and propose the spec."),
        call("prepare_spec_confirmation"),
        lambda m: (
            call("confirm_spec")
            if last_tool_result(m)["summary"]["auto_confirmable"]
            else call("ask_human", **last_tool_result(m)["summary"]["question"])
        ),
        lambda m: call("confirm_spec") if "choice" in last_tool_result(m)["summary"] else FINISH,
        FINISH,
    ]


def _profiler_script(primary: str, enrichment: str, time_column: str) -> list:
    return [
        call("get_spec"),
        call("profile_table", dataset=primary),
        call("profile_table", dataset=enrichment),
        call("infer_semantic_types", dataset=primary),
        call("infer_semantic_types", dataset=enrichment),
        call("infer_grain", dataset=primary),
        call("suggest_entities", dataset=primary),
        call("check_temporal_coverage", dataset=primary, time_column=time_column),
        call("discover_join_candidates", left=primary, right=enrichment),
        lambda m: call(
            "test_join",
            left=primary,
            right=enrichment,
            keys=[
                [
                    last_tool_result(m)["summary"]["best"]["left_column"],
                    last_tool_result(m)["summary"]["best"]["right_column"],
                ]
            ],
            normalization=[last_tool_result(m)["summary"]["best"]["normalization"]],
            candidates_artifact=last_tool_result(m)["artifact_id"],
        ),
        lambda m: call(
            "compile_quality_issues",
            artifact_ids=[
                _artifact(m, "profile_table"),
                _artifact(m, "infer_grain"),
                _artifact(m, "check_temporal_coverage"),
                _artifact(m, "test_join"),
            ],
        ),
        lambda m: call(
            "propose_spec",
            join_tests=[_artifact(m, "test_join")],
            grain=_artifact(m, "infer_grain"),
            entities=_artifact(m, "suggest_entities"),
            semantic_types=[a for a in _all_artifacts(m, "infer_semantic_types")],
            temporal=_artifact(m, "check_temporal_coverage"),
            quality_issues=_artifact(m, "compile_quality_issues"),
        ),
        lambda m: call(
            "finish", status="done", summary="proposed", artifact_ids=[_artifact(m, "compile_quality_issues")]
        ),
    ]


def _result_of(messages: list[Message], tool: str) -> dict:
    """Parsed result of the latest call of ``tool``."""
    names = {c.id: c.name for m in messages for c in m.tool_calls}
    for m in reversed(messages):
        if m.role == "tool" and names.get(m.tool_call_id) == tool:
            return json.loads(m.content)
    raise AssertionError(f"no result for {tool}")


def _full_orchestrator_script(primary: str, enrichment: str) -> list:
    """Orchestrator steps through control 1, control 2, preparation and features."""
    return [
        *_orchestrator_script(primary, enrichment)[:6],
        call("confirm_spec"),
        call("delegate", agent="preparer", task="propose: propose a cleaning plan from the quality issues."),
        call("prepare_cleaning_approval"),
        lambda m: call("ask_human", **last_tool_result(m)["summary"]["question"]),
        call("approve_cleaning_plan"),
        call("delegate", agent="preparer", task="apply: apply the approved plan and materialise the joins."),
        call("delegate", agent="feature_engineer", task="Build the per-entity features."),
        call("update_spec", patch={"phase": "detection"}),
        call("delegate", agent="detector", task="Run the detection methods and consolidate findings."),
        call("list_findings"),
        lambda m: LLMResponse(
            text="",
            tool_calls=tuple(
                ToolCall(f"crit{i}", "delegate", {"agent": "critico", "task": f["id"]})
                for i, f in enumerate(last_tool_result(m)["summary"]["findings"])
            ),
        ),
        call("list_findings"),
        lambda m: call(
            "delegate",
            agent="analista",
            task="Describe these confirmed findings: "
            + ", ".join(f["id"] for f in last_tool_result(m)["summary"]["findings"] if f["verdict"] == "confirmed"),
        ),
        call("review_findings"),
        call("update_spec", patch={"phase": "report"}),
        FINISH,
    ]


def _preparer_script() -> list:
    """Two preparer runs: propose, then apply."""
    return [
        call("get_spec"),
        lambda m: call(
            "propose_cleaning_plan",
            issues_artifact=_result_of(m, "get_spec")["summary"]["evidence_artifacts"]["quality_issues"],
        ),
        lambda m: call(
            "finish", status="done", summary="plan proposed", artifact_ids=[last_tool_result(m)["artifact_id"]]
        ),
        call("apply_cleaning_plan"),
        call("materialize_join", join_index=0),
        lambda m: call("finish", status="done", summary="joined", artifact_ids=[last_tool_result(m)["artifact_id"]]),
    ]


def _feature_script(entity: str, time_column: str, link: str) -> list:
    return [
        call("get_spec"),
        lambda m: call(
            "build_entity_features",
            input=_result_of(m, "get_spec")["summary"]["enriched_artifact"],
            entity=entity,
            time_column=time_column,
        ),
        lambda m: call(
            "build_graph",
            input=_result_of(m, "get_spec")["summary"]["enriched_artifact"],
            entity=entity,
            link_columns=[link],
            features=last_tool_result(m)["artifact_id"],
        ),
        lambda m: call("scale_encode", features=last_tool_result(m)["summary"]["features_table"]),
        lambda m: call("reduce_dimensions", features=last_tool_result(m)["artifact_id"], method="pca", n_components=5),
        lambda m: call("finish", status="done", summary="features built"),
    ]


def _detector_script() -> list:
    def runs(m: list[Message]) -> list[str]:
        names = {c.id: c.name for x in m for c in x.tool_calls}
        return [
            json.loads(x.content)["artifact_id"]
            for x in m
            if x.role == "tool"
            and names.get(x.tool_call_id, "").startswith("run_")
            and json.loads(x.content)["status"] == "ok"
        ]

    def spec(m: list[Message]) -> dict:
        return _result_of(m, "get_spec")["summary"]

    return [
        call("get_spec"),
        lambda m: LLMResponse(
            text="",
            tool_calls=tuple(
                ToolCall(f"o{i}", "run_outlier_detector", {"features": spec(m)["features"]["scaled"], "method": meth})
                for i, meth in enumerate(("isolation_forest", "ecod", "lof"))
            ),
        ),
        lambda m: call("run_clustering", features=spec(m)["features"]["embedding"], method="hdbscan"),
        lambda m: call("run_graph_analysis", graph=spec(m)["features"]["graph"], method="components"),
        lambda m: call("run_contextual_outliers", features=spec(m)["features"]["table"], context_column="attr_segment"),
        lambda m: call(
            "run_timeseries_anomaly",
            input=spec(m)["enriched_artifact"],
            time_column="start_time",
            measure="count",
            group_by="cell_id",
        ),
        lambda m: call(
            "run_timeseries_anomaly",
            input=spec(m)["enriched_artifact"],
            time_column="start_time",
            measure="duration_sec",
            granularity="day",
        ),
        lambda m: call("consolidate_findings", run_ids=runs(m)),
        lambda m: call("finish", status="done", summary="findings consolidated"),
    ]


def _critic_script(copies: int = 12) -> list:
    """One critic run per finding: checks, then the SKILL.md verdict policy."""

    def fid(m: list[Message]) -> str:
        return m[0].content.split("Try to refute finding ")[1].split(".")[0]

    def verdict(m: list[Message]) -> LLMResponse:
        artifact = _result_of(m, "artifact_check")["summary"]
        stability = _result_of(m, "stability_check")["summary"].get("stability", 0)
        choice = (
            "artifact"
            if artifact["suspected"]
            else "unstable"
            if stability < 0.6
            else "weak"
            if artifact.get("low_support_share", 0) >= 0.5
            else "confirmed"
        )
        return call("set_verdict", finding=fid(m), verdict=choice, evidence={"stability": stability})

    one = [
        lambda m: call("get_finding", finding=fid(m)),
        lambda m: call("stability_check", finding=fid(m), n_runs=2),
        lambda m: call("ensemble_agreement", finding=fid(m)),
        lambda m: call("artifact_check", finding=fid(m)),
        lambda m: call("match_pattern_registry", finding=fid(m)),
        verdict,
        lambda m: call("finish", status="done", summary=f"verdict for {fid(m)}"),
    ]
    return one * copies


def _analyst_script() -> list:
    def ids(m: list[Message]) -> list[str]:
        text = m[0].content.split("Describe these confirmed findings:")[1].split("##")[0]
        return [x.strip(" .\n") for x in text.split(",") if x.strip(" .\n")]

    def each(tool: str, **extra) -> object:
        return lambda m: LLMResponse(
            text="",
            tool_calls=tuple(ToolCall(f"{tool}{i}", tool, {"finding": f, **extra}) for i, f in enumerate(ids(m))),
        )

    return [
        each("describe_group"),
        each("fit_surrogate_rules"),
        each("get_exemplars", k=3),
        each("write_finding", description="Scripted description using only the numbers from the tools."),
        lambda m: call("finish", status="done", summary="described"),
    ]


def _all_artifacts(messages: list[Message], tool: str) -> list[str]:
    names = {c.id: c.name for m in messages for c in m.tool_calls}
    return [
        json.loads(m.content)["artifact_id"] for m in messages if m.role == "tool" and names.get(m.tool_call_id) == tool
    ]


# ------------------------------------------------------------------ end to end (dummy)
def test_end_to_end_full_flow(dummy_base: Path, tmp_path: Path) -> None:
    settings = dummy_settings(dummy_base, tmp_path / "ws")
    ctx = RunContext.create(settings, seed=42, run_id="e2e")
    llm = FakeLLM(
        {
            "orquestrador": _full_orchestrator_script("traffic", "cust_segmentation"),
            "profiler": _profiler_script("traffic", "cust_segmentation", "start_time"),
            "preparer": _preparer_script(),
            "feature_engineer": _feature_script("msisdn", "start_time", "imei"),
            "detector": _detector_script(),
            "critico": _critic_script(),
            "analista": _analyst_script(),
        }
    )
    runtime = Runtime(
        ctx=ctx,
        llm=llm,
        human=AutoApproveHuman(),
        options={"instruction": "analyse traffic enriched with cust_segmentation"},
    )
    result = runtime.run_agent("orquestrador", "analyse traffic enriched with cust_segmentation")
    assert result.status == "done", result

    spec = load_spec(ctx.spec_path)
    expected = yaml.safe_load((dummy_base / "truth" / "expected_spec.yaml").read_text())
    join = spec.joins[0]
    assert join.status == "confirmed"
    assert [list(k) for k in join.keys] == expected["joins"][0]["keys"]
    assert join.key_normalization == expected["joins"][0]["key_normalization"]
    assert spec.entity.column == expected["entity"]["column"]
    assert spec.time_column == expected["time_column"]
    assert spec.grain.columns == expected["grain"]["columns"]
    assert spec.phase == "report"
    # High confidence -> confirmed automatically; control 2 answered by auto-approve.
    assert {d.by for d in spec.decisions} == {"auto"}
    assert any(d.topic == "cleaning_approval" for d in spec.decisions)
    assert spec.cleaning_plan.status == "applied" and set(spec.prepared) == {"traffic", "cust_segmentation"}
    assert spec.enriched_artifact and spec.features.table and spec.features.scaled and spec.features.embedding
    assert spec.features.graph
    # Phase 5: every finding judged in an isolated critic run; trap -> artifact; control 3 recorded.
    from jellybean.skills._shared.findings import all_findings

    with use_run(ctx):
        findings = all_findings()
    assert findings and all(f.critic_verdict for f in findings)
    assert all(f.description for f in findings if f.critic_verdict == "confirmed")
    trap = [f for f in findings if f.type == "temporal" and f.context.get("measure") == "duration_sec"]
    assert trap and all(f.critic_verdict == "artifact" for f in trap)
    reviews = [d for d in spec.decisions if d.topic.startswith("finding_review:")]
    assert reviews and {d.choice for d in reviews} == {"No feedback"}  # auto-approve teaches nothing
    critic_calls = [c for c in llm.calls if c.agent == "critico"]
    starts = [c for c in critic_calls if len(c.messages) == 1]
    assert len(starts) == len(findings)  # one fresh, isolated context per finding
    assert all(
        "Try to refute finding" in c.messages[0].content and "DETECTOR" not in c.messages[0].content for c in starts
    )
    assert any(d.topic == "control_1:join[0]" for d in spec.decisions)
    assert "quality_issues" in spec.evidence_artifacts and spec.columns["duration_sec"].semantic_type == "measure"

    # Hygiene over the whole trace.
    events = [json.loads(line) for line in ctx.trace_path.read_text().splitlines()]
    tools = [e for e in events if e["kind"] == "tool_call"]
    assert max(e["output_chars"] for e in tools) <= 4000
    assert max(e["raw_rows"] for e in tools) <= 20
    assert {e["agent"] for e in tools} == {
        "orquestrador",
        "profiler",
        "preparer",
        "feature_engineer",
        "detector",
        "critico",
        "analista",
    }
    assert ctx.db_path.exists()


# ----------------------------------------------------------------- control 1 (human)
def _propose_on_fixture(ctx: RunContext) -> Runtime:
    llm = FakeLLM(
        {
            "orquestrador": [*_orchestrator_script("events", "customers")[:5], FINISH],
            "profiler": _profiler_script("events", "customers", "ts"),
        }
    )
    runtime = Runtime(ctx=ctx, llm=llm, options={"instruction": "analyse events enriched with customers"})
    assert runtime.run_agent("orquestrador", "go").status == "done"
    return runtime


def _tool(name: str):
    from jellybean.skills.orquestrador import tools as ot

    return getattr(ot, name)


def test_control_1_requires_human_when_ambiguous(run_ctx: RunContext) -> None:
    runtime = _propose_on_fixture(run_ctx)
    runtime.human = ScriptedHuman(["Confirm proposal"])
    with runtime.active():
        # Make the entity ambiguous: choose a column that is not the top candidate.
        _tool("update_spec")(patch={"entity": {"column": "event_id", "alternatives": ["phone"]}})
        prep = _tool("prepare_spec_confirmation")()
        assert prep.summary["auto_confirmable"] is False
        question = prep.summary["question"]
        assert question["topic"] == "spec_confirmation" and question["recommended"] == "Confirm proposal"
        refused = _tool("confirm_spec")()
        assert refused.status == "error" and "ask_human" in refused.summary["error"]
        answer = _tool("ask_human")(**question)
        assert answer.summary == {"choice": "Confirm proposal", "by": "human", "note": ""}
        confirmed = _tool("confirm_spec")()
        assert confirmed.status == "ok" and confirmed.summary["by"] == "human"
    spec = load_spec(run_ctx.spec_path)
    assert spec.joins[0].status == "confirmed" and spec.phase == "cleaning"
    assert spec.decisions[0].topic == "spec_confirmation" and spec.decisions[0].by == "human"


def test_control_1_auto_on_clear_fixture(run_ctx: RunContext) -> None:
    runtime = _propose_on_fixture(run_ctx)
    with runtime.active():
        prep = _tool("prepare_spec_confirmation")()
        assert prep.summary["auto_confirmable"] is True, prep.summary
        assert _tool("confirm_spec")().summary["by"] == "auto"


def test_init_run_keeps_original_instruction_and_labels(run_ctx: RunContext) -> None:
    runtime = Runtime(ctx=run_ctx, llm=FakeLLM({}), options={"instruction": "the real one", "labels": "known"})
    with runtime.active():
        result = _tool("init_run")(instruction="a paraphrase", report_language="pt")
        assert result.warnings and _tool("init_run")(instruction="x").warnings == ["spec.yaml already exists; reused."]
    spec = load_spec(run_ctx.spec_path)
    assert (spec.instruction, spec.report_language, spec.labels_dataset) == ("the real one", "pt", "known")


def test_ask_human_validation_and_auto(run_ctx: RunContext) -> None:
    runtime = Runtime(ctx=run_ctx, llm=FakeLLM({}))
    with runtime.active():
        assert _tool("ask_human")(question="q", options=["a", "a"]).status == "error"
        assert _tool("ask_human")(question="q", options=["a", "b"], recommended="c").status == "error"
        auto = _tool("ask_human")(question="q", options=["a", "b"], recommended="b")
        assert auto.summary["choice"] == "b" and auto.summary["by"] == "auto"


def test_update_spec_rejects_invalid_patch(run_ctx: RunContext) -> None:
    runtime = Runtime(ctx=run_ctx, llm=FakeLLM({}))
    with runtime.active():
        assert _tool("update_spec")(patch={"x": 1}).status == "error"  # no spec yet
        _tool("init_run")(instruction="i")
        assert _tool("update_spec")(patch={"phase": "nonsense"}).status == "error"
        assert _tool("update_spec")(patch={"seed": 1}).status == "error"
        assert _tool("update_spec")(patch={"time_column": "ts"}).status == "ok"


# ---------------------------------------------------------------------- console human
@pytest.mark.parametrize(
    ("answers", "expected"),
    [([""], ("b", "")), (["1"], ("a", "")), (["a | looks fine"], ("a", "looks fine")), (["9", "x", "2"], ("b", ""))],
)
def test_console_human(answers: list[str], expected: tuple[str, str]) -> None:
    replies = iter(answers)
    out = io.StringIO()
    human = ConsoleHuman(read=lambda prompt: next(replies), out=out)
    answer = human.ask("Pick one", ["a", "b"], {"coverage": 0.97}, recommended="b")
    assert (answer.choice, answer.note) == expected and answer.by == "human"
    assert "(recommended)" in out.getvalue() and "coverage" in out.getvalue()


def test_use_run_restores_context(run_ctx: RunContext, settings) -> None:
    other = RunContext.create(settings, seed=1, run_id="other")
    with use_run(other):
        pass
    from jellybean.runtime.context import current_run

    assert current_run().run_id == run_ctx.run_id


# ----------------------------------------------------------------- control 2
def _with_plan(ctx: RunContext, explicit: bool = True) -> Runtime:
    runtime = _propose_on_fixture(ctx)
    from jellybean.skills._shared.artifacts import ArtifactStore
    from jellybean.skills._shared.spec_tools import patch_spec

    ops = [
        {
            "id": "P01",
            "dataset": "events",
            "operation": {"op": "dedupe"},
            "issue_ids": ["Q1"],
            "justification": "dups",
            "requires_explicit_approval": False,
        }
    ]
    if explicit:
        ops.append(
            {
                "id": "P02",
                "dataset": "events",
                "operation": {
                    "op": "fix_units",
                    "column": "amount",
                    "factor": 10,
                    "file": "date=2026-10-02/part-0.parquet",
                },
                "issue_ids": ["Q2"],
                "justification": "units",
                "requires_explicit_approval": True,
            }
        )
    with runtime.active():
        plan = ArtifactStore(ctx).save_json("cleaning_plan", {"ops": ops, "not_addressed": []})
        _tool("confirm_spec")()
        patch_spec({"cleaning_plan": {"artifact_id": plan.id, "status": "proposed"}})
    return runtime


@pytest.mark.parametrize(
    ("answer", "status", "approved"),
    [
        ("Approve recommended plan", "approved", ["P01"]),
        ("Approve plan including inferred fixes", "approved", ["P01", "P02"]),
        ("Reject plan", "rejected", []),
    ],
)
def test_control_2_paths(run_ctx: RunContext, answer: str, status: str, approved: list[str]) -> None:
    runtime = _with_plan(run_ctx)
    runtime.human = ScriptedHuman([answer])
    with runtime.active():
        assert "Ask the human first" in _tool("approve_cleaning_plan")().summary["error"]
        question = _tool("prepare_cleaning_approval")().summary["question"]
        assert question["recommended"] == "Approve recommended plan" and len(question["options"]) == 3
        assert any("[explicit approval]" in line for line in question["evidence"]["operations"])
        _tool("ask_human")(**question)
        result = _tool("approve_cleaning_plan")()
        assert result.summary["status"] == status
    spec = load_spec(run_ctx.spec_path)
    assert spec.cleaning_plan.status == status and spec.cleaning_plan.approved_ops == approved


def test_control_2_without_explicit_ops_has_two_options(run_ctx: RunContext) -> None:
    runtime = _with_plan(run_ctx, explicit=False)
    with runtime.active():
        question = _tool("prepare_cleaning_approval")().summary["question"]
    assert question["options"] == ["Approve recommended plan", "Reject plan"]


def test_control_2_requires_a_proposed_plan(run_ctx: RunContext) -> None:
    runtime = _propose_on_fixture(run_ctx)
    with runtime.active():
        assert "No proposed cleaning plan" in _tool("prepare_cleaning_approval")().summary["error"]


def test_control_point_package_cannot_be_altered(run_ctx: RunContext) -> None:
    """Regression: the orchestrator must not change options/recommendation of a control point."""
    runtime = _with_plan(run_ctx)
    with runtime.active():
        question = _tool("prepare_cleaning_approval")().summary["question"]
        tampered = {**question, "recommended": "Reject plan"}
        refused = _tool("ask_human")(**tampered)
        assert refused.status == "error" and "exactly" in refused.summary["error"]
        fewer = {**question, "options": ["Reject plan", "Approve recommended plan"]}
        assert _tool("ask_human")(**fewer).status == "error"
        assert _tool("ask_human")(**question).summary["choice"] == "Approve recommended plan"
        # Once answered, the package is consumed; ordinary questions are unaffected.
        assert _tool("ask_human")(question="other?", options=["x", "y"], recommended="y").status == "ok"


# ----------------------------------------------------------------- control 3
def _save_findings(spec: list[tuple[str, str]]) -> None:
    from jellybean.skills._shared.findings import Finding, save_finding

    for fid, verdict in spec:
        save_finding(
            Finding(
                id=fid,
                type="outlier_group",
                entity_column="phone",
                entity_ids_artifact="x_v1",
                size=3,
                methods=["lof"],
                critic_verdict=verdict,
                description="d" * 600,
                top_features=[{"feature": f"feat_{i}", "direction": "higher", "effect": 9.9} for i in range(8)],
            )
        )


def test_control_3_review_and_feedback(run_ctx: RunContext) -> None:
    from jellybean.skills._shared.findings import load_finding

    runtime = _propose_on_fixture(run_ctx)
    runtime.human = ScriptedHuman(["Confirmed pattern", "False alarm"])
    with runtime.active():
        assert _tool("review_findings")().summary["reviewed"] == 0
        _save_findings([("F001", "confirmed"), ("F002", "confirmed"), ("F003", "weak")])
        result = _tool("review_findings")().summary
    assert result["feedback"] == {"F001": "confirmed", "F002": "false_alarm"}
    assert result["register_as_patterns"] == ["F001"]
    assert runtime.human.questions == [
        "Is finding F001 (outlier_group, 3 members, novelty None) a real pattern?",
        "Is finding F002 (outlier_group, 3 members, novelty None) a real pattern?",
    ]
    with use_run(run_ctx):
        assert load_finding("F002").human_feedback == "false_alarm"
    topics = [d.topic for d in load_spec(run_ctx.spec_path).decisions]
    assert "finding_review:F001" in topics and "finding_review:F002" in topics


def test_control_3_many_findings_are_all_reviewed(run_ctx: RunContext) -> None:
    """Regression: with many findings the review must not depend on a truncated tool output."""
    runtime = _propose_on_fixture(run_ctx)
    with runtime.active():
        _save_findings([(f"F{i:03d}", "confirmed") for i in range(1, 13)])
        result = _tool("review_findings")()
    assert result.summary["reviewed"] == 12 and not result.truncated
    assert set(result.summary["feedback"].values()) == {None}  # auto-approve: "No feedback"
    assert result.summary["register_as_patterns"] == []


def test_rework_limit_is_enforced(run_ctx: RunContext) -> None:
    llm = FakeLLM(
        {
            "orquestrador": [call("delegate", agent="detector", task="detect") for _ in range(4)] + [FINISH],
            "detector": [call("finish", status="done", summary="ok") for _ in range(4)],
        }
    )
    runtime = Runtime(ctx=run_ctx, llm=llm)
    runtime.run_agent("orquestrador", "go")
    detector_runs = [c for c in llm.calls if c.agent == "detector"]
    assert len(detector_runs) == 3  # 1 + max_rework_cycles (2)
    assert "Rework limit reached" in llm.calls[-1].messages[-1].content


def test_reproposal_with_same_operations_is_unchanged(run_ctx: RunContext) -> None:
    """Regression: an auto-approved re-proposal identical to the applied cleaning changes nothing."""
    from jellybean.skills._shared.spec_tools import patch_spec
    from jellybean.skills.preparer import tools as pr

    runtime = _with_plan(run_ctx)
    with runtime.active():
        question = _tool("prepare_cleaning_approval")().summary["question"]
        _tool("ask_human")(**question)
        assert _tool("approve_cleaning_plan")().summary["status"] == "approved"
        assert pr.apply_cleaning_plan().status == "ok"
        applied = load_spec(run_ctx.spec_path).cleaning_plan
        assert applied.status == "applied" and applied.applied_signature
        # Re-propose the very same plan (as a critic-driven rework would) and approve the recommended option.
        plan = load_spec(run_ctx.spec_path).cleaning_plan.artifact_id
        patch_spec({"cleaning_plan": {"artifact_id": plan, "status": "proposed", "approved_ops": []}})
        question = _tool("prepare_cleaning_approval")().summary["question"]
        _tool("ask_human")(**question)
        again = _tool("approve_cleaning_plan")().summary
        assert again["status"] == "unchanged"
        assert load_spec(run_ctx.spec_path).cleaning_plan.status == "applied"
        # A human approving the inferred fix as well is a real change.
        patch_spec({"cleaning_plan": {"status": "proposed", "approved_ops": []}})
        runtime.human = ScriptedHuman(["Approve plan including inferred fixes"])
        question = _tool("prepare_cleaning_approval")().summary["question"]
        _tool("ask_human")(**question)
        assert _tool("approve_cleaning_plan")().summary["status"] == "approved"
