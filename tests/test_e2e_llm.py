# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""End-to-end run with a real LLM: the whole flow, the report and the evaluation.

Runs only with ``uv run pytest -m llm`` and a configured ``.env`` in the project
root (JELLYBEAN_PROVIDER, JELLYBEAN_MODEL and credentials). It uses the session
dummy data, never ./data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext
from jellybean.runtime.human import AutoApproveHuman
from jellybean.runtime.llm_client import LLMConfigurationError, build_llm_client
from jellybean.settings import AppConfig, EnvSettings, Settings, load_env_file
from jellybean.spec import load_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "analisa os dados em traffic e enriquece-os com os valores de cust_segmentation"


@pytest.mark.llm
def test_real_run_full_flow(dummy_base: Path, tmp_path: Path) -> None:
    load_env_file(PROJECT_ROOT / ".env")
    env = EnvSettings(
        _env_file=PROJECT_ROOT / ".env",
        data_root=dummy_base / "data",
        workspace_root=tmp_path / "ws",
        ground_truth_root=dummy_base / "truth",
    )
    settings = Settings(env=env, config=AppConfig())
    try:
        llm = build_llm_client(settings)
    except LLMConfigurationError as exc:
        pytest.skip(f"LLM not configured: {exc}")
    if env.provider == "bedrock" and not (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
        pytest.skip("AWS_REGION not set for Bedrock")

    ctx = RunContext.create(settings, seed=42, run_id="e2e_llm")
    runtime = Runtime(ctx=ctx, llm=llm, human=AutoApproveHuman(), options={"instruction": INSTRUCTION})
    result = runtime.run_agent("orquestrador", INSTRUCTION)
    assert result.status == "done", result

    spec = load_spec(ctx.spec_path)
    expected = yaml.safe_load((dummy_base / "truth" / "expected_spec.yaml").read_text())
    assert spec.report_language == "pt"  # the instruction is in Portuguese
    confirmed = [j for j in spec.joins if j.status == "confirmed"]
    assert [[list(k) for k in j.keys] for j in confirmed] == [expected["joins"][0]["keys"]]
    assert confirmed[0].key_normalization == expected["joins"][0]["key_normalization"]
    assert spec.entity.column == expected["entity"]["column"]
    assert spec.time_column == expected["time_column"]

    # Phase 4: control 2 answered (auto-approve), cleaning applied, enriched table and features built.
    assert any(d.topic == "cleaning_approval" for d in spec.decisions)
    assert spec.cleaning_plan.status == "applied"
    assert spec.enriched_artifact and spec.features.table and spec.features.scaled
    assert spec.phase == "report"

    # Phase 5: findings judged by isolated critics; the unit-error trap is an artifact.
    from jellybean.runtime.context import use_run
    from jellybean.skills._shared.findings import all_findings

    with use_run(ctx):
        findings = all_findings()
    assert findings and all(f.critic_verdict for f in findings)
    assert any(f.critic_verdict == "confirmed" for f in findings)
    trap = [f for f in findings if f.type == "temporal" and f.context.get("measure") == "duration_sec"]
    assert all(f.critic_verdict == "artifact" for f in trap)
    assert all(f.description for f in findings if f.critic_verdict == "confirmed")

    # Phase 6: report and evaluation against the ground truth.
    from jellybean.evaluation import evaluate, format_evaluation
    from jellybean.reporting import write_report_file

    report = Path(write_report_file(ctx)).read_text()
    assert "# Relatório Jellybean" in report  # Portuguese instruction -> Portuguese report
    evaluation = evaluate(ctx, dummy_base / "truth")
    print(format_evaluation(evaluation))
    acceptance = evaluation["acceptance"]
    assert acceptance["join_discovered_with_normalisation_and_region_code_rejected"], evaluation["join"]
    assert acceptance["unit_error_is_artifact"], evaluation["traps"]
    assert acceptance["four_of_six_anomalies_recall_0_6"], evaluation["where_lost"]

    events = [json.loads(line) for line in ctx.trace_path.read_text().splitlines()]
    tools = [e for e in events if e["kind"] == "tool_call"]
    assert max(e["output_chars"] for e in tools) <= 4000
    assert max(e["raw_rows"] for e in tools) <= 20
