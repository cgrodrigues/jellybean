"""Phase 6: report, evaluation, feedback, resume, partial reports and prompt caching.

Builds on the phase-5 run (deterministic tools over the dummy data). Runs that
must not alter that shared state work on a copy of its workspace folder.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from jellybean.cli import app
from jellybean.evaluation import evaluate, format_evaluation
from jellybean.reporting import build_report
from jellybean.runtime.agent_runner import budget_tokens
from jellybean.runtime.context import RunContext, use_run
from jellybean.runtime.llm_client import FakeLLM, LangChainLLMClient, LLMResponse, call
from jellybean.settings import AppConfig, EnvSettings, Settings, reset_settings
from jellybean.skills._shared.findings import all_findings, load_finding
from jellybean.spec import load_spec

runner = CliRunner()


@pytest.fixture
def cli_env(phase5: dict, monkeypatch: pytest.MonkeyPatch) -> RunContext:
    """Point the CLI's settings at the phase-5 workspace, data and ground truth."""
    ctx: RunContext = phase5["ctx"]
    monkeypatch.setenv("JELLYBEAN_WORKSPACE_ROOT", str(ctx.settings.workspace_root))
    monkeypatch.setenv("JELLYBEAN_DATA_ROOT", str(ctx.settings.data_root))
    monkeypatch.setenv("JELLYBEAN_GROUND_TRUTH_ROOT", str(ctx.settings.ground_truth_root))
    reset_settings()
    yield ctx
    reset_settings()


def _copy_run(ctx: RunContext, new_id: str) -> RunContext:
    target = ctx.settings.workspace_root / new_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ctx.run_dir, target)
    for path in target.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    spec = yaml.safe_load((target / "spec.yaml").read_text())
    spec["run_id"] = new_id
    (target / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return RunContext.open(ctx.settings, new_id, seed=ctx.seed)


# ---------------------------------------------------------------------- report
def test_report_contents(phase5: dict) -> None:
    ctx = phase5["ctx"]
    text = build_report(ctx)
    for section in (
        "## Analysis specification",
        "## Decisions",
        "## Data preparation",
        "## Findings by verdict",
        "## Human feedback and pattern registry",
        "## Limitations",
        "## Run hygiene",
    ):
        assert section in text, section
    assert "strip_prefix_plus351" in text
    assert "matched_cust_segmentation = false" in text  # unmatched rows appear in the report
    assert "fix_units" in text and "Operations not applied" in text  # the unit fix that was not approved
    with use_run(ctx):
        confirmed = [f for f in all_findings() if f.critic_verdict == "confirmed"]
    assert all(f.id in text for f in confirmed)
    assert "scale" in text  # artefact reasons are listed
    assert "+351" not in text.replace("strip_prefix_plus351", "")  # no raw phone numbers


def test_report_in_portuguese_and_partial(phase5: dict) -> None:
    copy = _copy_run(phase5["ctx"], "report_pt")
    spec = yaml.safe_load(copy.spec_path.read_text())
    spec["report_language"], spec["phase"] = "pt", "critique"
    copy.spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    text = build_report(copy, run_result={"status": "failed", "issues": ["token_budget_exhausted"]})
    assert "# Relatório Jellybean" in text and "## Achados por veredicto" in text
    assert "PARCIAL" in text and "token_budget_exhausted" in text


def test_write_report_tool_respects_limits(phase5: dict) -> None:
    from jellybean.runtime.agent_runner import Runtime
    from jellybean.skills.orquestrador import tools as ot

    copy = _copy_run(phase5["ctx"], "report_tool")
    with Runtime(ctx=copy, llm=FakeLLM({})).active():
        result = ot.write_report()
    assert result.status == "ok" and len(result.to_json()) <= 4000
    assert (copy.run_dir / "report.md").exists()


# ------------------------------------------------------------------ evaluation
def test_evaluation_against_ground_truth(phase5: dict) -> None:
    result = evaluate(phase5["ctx"], phase5["truth"])
    assert result["join"]["keys_and_normalization_correct"] and result["join"]["false_candidate_rejected"]
    assert all(result["spec"].values())
    assert result["recall"]["D"] is None  # orders is not part of this run
    assert sum(r >= 0.6 for r in result["recall"].values() if r is not None) >= 4
    assert result["traps"]["units_error"]["pass"]
    assert all(result["quality_detected"].values()), result["quality_detected"]
    assert set(result["where_lost"]) == set("ABCDEF")
    assert result["precision"]["entity_precision"] is not None and "top100_by_score" in result["precision"]
    assert all(result["acceptance"].values()), result["acceptance"]
    assert "Acceptance criteria" in format_evaluation(result)


def test_evaluate_and_report_cli(cli_env: RunContext) -> None:
    result = runner.invoke(app, ["evaluate", cli_env.run_id])
    assert result.exit_code == 0, result.output
    assert "Recall per anomaly" in result.output and (cli_env.run_dir / "evaluation.json").exists()
    shown = runner.invoke(app, ["report", cli_env.run_id, "--show"])
    assert shown.exit_code == 0 and "# Jellybean report" in shown.output
    assert runner.invoke(app, ["report", "no_such_run"]).exit_code == 2


# -------------------------------------------- feedback -> known in a second run
def test_feedback_registers_pattern_and_second_run_knows_it(cli_env: RunContext) -> None:
    with use_run(cli_env):
        target = next(
            f
            for f in all_findings()
            if f.critic_verdict == "confirmed" and f.type != "temporal" and f.human_feedback is None
        )
    result = runner.invoke(app, ["feedback", cli_env.run_id, target.id, "confirmed", "--note", "real pattern"])
    assert result.exit_code == 0, result.output
    assert "Pattern PAT-" in result.output
    with use_run(cli_env):
        assert load_finding(target.id).registered_pattern
    decision = [d for d in load_spec(cli_env.spec_path).decisions if d.topic == f"finding_review:{target.id}"][-1]
    assert decision.by == "human" and decision.evidence == {"note": "real pattern"}

    # A second run over the same data: the same finding is now "known".
    second = _copy_run(cli_env, "second_run")
    from jellybean.skills.critico import tools as cr

    with use_run(second):
        assert cr.match_pattern_registry(target.id).summary["novelty"] == "known"
    assert runner.invoke(app, ["feedback", cli_env.run_id, target.id, "maybe"]).exit_code == 2


# ---------------------------------------------------------------------- resume
def test_resume_continues_from_the_spec_phase(cli_env: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    copy = _copy_run(cli_env, "resume_me")
    llm = FakeLLM(
        {"orquestrador": [call("get_spec"), call("write_report"), call("finish", status="done", summary="resumed")]}
    )
    monkeypatch.setattr("jellybean.runtime.llm_client.build_llm_client", lambda settings: llm)
    result = runner.invoke(app, ["resume", copy.run_id, "--auto-approve"])
    assert result.exit_code == 0, result.output
    first = llm.calls[0].messages[0].content
    assert f"Resume run {copy.run_id}" in first and "The spec phase is '" in first
    assert json.loads((copy.run_dir / "run_result.json").read_text())["status"] == "done"
    assert (copy.run_dir / "report.md").exists()


# ------------------------------------------------- token budget -> partial report
def test_run_stops_cleanly_with_a_partial_report(
    cli_env: RunContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "tiny.yaml"
    config.write_text("limits:\n  token_budget_per_run: 300\n")
    monkeypatch.setenv("JELLYBEAN_CONFIG_FILE", str(config))
    reset_settings()
    llm = FakeLLM(
        {"orquestrador": [call("init_run", instruction="analyse traffic")] + [call("list_datasets") for _ in range(10)]}
    )
    monkeypatch.setattr("jellybean.runtime.llm_client.build_llm_client", lambda settings: llm)
    monkeypatch.setattr("jellybean.runtime.llm_client.build_chat_model", lambda settings: None)
    result = runner.invoke(app, ["run", "analyse traffic", "--auto-approve", "--run-id", "budget_stop"])
    assert result.exit_code == 1, result.output  # failed, but cleanly
    assert "▶ orquestrador" in result.output and "· init_run(" in result.output  # live progress
    run_dir = cli_env.settings.workspace_root / "budget_stop"
    report = (run_dir / "report.md").read_text()
    assert "PARTIAL" in report and "token_budget_exhausted" in report


# -------------------------------------------------------------- prompt caching
def _settings(provider: str, caching: bool = True) -> Settings:
    config = AppConfig.model_validate({"llm": {"prompt_caching": caching}})
    return Settings(env=EnvSettings(provider=provider, model="m"), config=config)


@pytest.mark.parametrize(
    ("provider", "caching", "expected"),
    [
        ("anthropic", True, {"cache_control": {"type": "ephemeral"}}),
        ("bedrock", True, {"cache_control": {"type": "ephemeral"}}),
        ("openai", True, {}),  # OpenAI caches automatically
        ("anthropic", False, {}),
    ],
)
def test_prompt_caching_options(provider: str, caching: bool, expected: dict) -> None:
    client = LangChainLLMClient(_settings(provider, caching), chat_model=object())
    assert client._invoke_kwargs() == expected


def test_cached_tokens_count_less_in_the_budget() -> None:
    plain = LLMResponse(text="", input_tokens=1000, output_tokens=100)
    cached = LLMResponse(text="", input_tokens=1000, output_tokens=100, cache_read_tokens=900)
    assert budget_tokens(plain) == 1100
    assert budget_tokens(cached) == 1000 - 900 + 90 + 100


def test_quiet_run_prints_no_progress(cli_env: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeLLM({"orquestrador": [call("finish", status="done", summary="nothing to do")]})
    monkeypatch.setattr("jellybean.runtime.llm_client.build_llm_client", lambda settings: llm)
    monkeypatch.setattr("jellybean.runtime.llm_client.build_chat_model", lambda settings: None)
    result = runner.invoke(app, ["run", "x", "--auto-approve", "--run-id", "quiet_run", "--quiet"])
    assert result.exit_code == 0, result.output
    assert "▶" not in result.output and "Status: done" in result.output


def test_outputs_command_and_report_section(cli_env: RunContext) -> None:
    listed = runner.invoke(app, ["outputs", cli_env.run_id])
    assert listed.exit_code == 0, listed.output
    # The phase-5 fixture calls tools directly (no agent), so its artifacts are attributed to "system".
    assert "system:" in listed.output and "enriched_traffic_v1 (table, " in listed.output
    assert "features_scaled_v1 (table" in listed.output and "document(s)" in listed.output
    as_json = json.loads(runner.invoke(app, ["outputs", cli_env.run_id, "--type", "table", "--json"]).output)
    assert as_json["outputs"] and all(o["type"] == "table" and Path(o["path"]).exists() for o in as_json["outputs"])
    report = build_report(cli_env)
    assert (
        "## Outputs (datasets and other artifacts)" in report and "](artifacts/enriched_traffic_v1.parquet)" in report
    )


def test_run_result_has_outputs_by_agent(cli_env: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeLLM({"orquestrador": [call("init_run", instruction="x"), call("finish", status="done", summary="ok")]})
    monkeypatch.setattr("jellybean.runtime.llm_client.build_llm_client", lambda settings: llm)
    monkeypatch.setattr("jellybean.runtime.llm_client.build_chat_model", lambda settings: None)
    result = runner.invoke(app, ["run", "x", "--auto-approve", "--run-id", "outputs_run"])
    assert result.exit_code == 0, result.output
    payload = json.loads((cli_env.settings.workspace_root / "outputs_run" / "run_result.json").read_text())
    assert "outputs_by_agent" in payload and "Outputs (" in result.output


# ------------------------------------------------------------------- overview plots
def test_cluster_map_and_findings_overview(phase5: dict) -> None:
    from jellybean.skills._shared.artifacts import ArtifactStore
    from jellybean.skills.detector import tools as de

    ctx = phase5["ctx"]
    with use_run(ctx):
        plots = [r for r in ArtifactStore(ctx).list("plot") if r.params.get("kind") == "clusters"]
        assert plots, "run_clustering should draw the cluster map"
        assert (ctx.run_dir / plots[0].path).stat().st_size > 10_000
        hdbscan_run = plots[0].params["run"]
        again = de.plot_cluster_run(hdbscan_run)
        assert again.status == "ok" and again.summary["plot"].startswith("plot_clusters_")
        assert "not a clustering run" in de.plot_cluster_run(phase5["runs"][0]).summary["error"]
    text = build_report(ctx)
    assert "![Findings on the PCA map](report_findings_overview.png)" in text
    assert (ctx.run_dir / "report_findings_overview.png").exists()
    assert "![Cluster map:" in text


def test_report_shows_filters(phase5: dict) -> None:
    copy = _copy_run(phase5["ctx"], "report_filters")
    spec = yaml.safe_load(copy.spec_path.read_text())
    spec["filters"] = {"traffic": "(\"date\" >= '2026-10-01')"}
    copy.spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    assert "**Row filters:** `traffic`: (\"date\" >= '2026-10-01')" in build_report(copy)


def test_report_shows_context_notes(phase5: dict) -> None:
    copy = _copy_run(phase5["ctx"], "report_notes")
    spec = yaml.safe_load(copy.spec_path.read_text())
    spec["context_notes"] = "traffic.duration_sec: call duration in seconds.\ncharge: amount in EUR."
    copy.spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    text = build_report(copy)
    assert "**Context notes (from the datasets' README.md):**" in text
    assert "> traffic.duration_sec: call duration in seconds.\n> charge: amount in EUR." in text


def test_evaluation_reports_entity_and_filters(phase5: dict) -> None:
    result = evaluate(phase5["ctx"], phase5["truth"])
    assert result["entity"] == "traffic.msisdn" and result["filters"] == {}
    copy = _copy_run(phase5["ctx"], "eval_filters")
    spec = yaml.safe_load(copy.spec_path.read_text())
    spec["filters"] = {"traffic": '("hour" < 12)'}
    copy.spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    filtered = evaluate(copy, phase5["truth"])
    assert filtered["filters"] and "row filters" in filtered["note"]
    assert "Filters:" in format_evaluation(filtered)


def test_anomalies_of_another_entity_are_not_applicable(phase5: dict) -> None:
    copy = _copy_run(phase5["ctx"], "eval_entity")
    spec = yaml.safe_load(copy.spec_path.read_text())
    spec["entity"] = {"column": "imei", "alternatives": [], "source": "user", "dataset": "traffic"}
    copy.spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    result = evaluate(copy, phase5["truth"])
    assert result["recall"]["A"] is None and "this run analyses 'imei'" in result["where_lost"]["A"]
