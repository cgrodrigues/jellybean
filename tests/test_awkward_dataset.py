# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""End to end on a deliberately awkward dataset: every tool must answer without an error.

One event table with upper-case column names, no unique key, a single day of
data, a category with one value and no enrichment table. The deterministic
tool chain of a run (no LLM) must reach the report; an error result is what the
console shows as a red line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest

from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext
from jellybean.runtime.human import AutoApproveHuman
from jellybean.runtime.llm_client import FakeLLM
from jellybean.runtime.tool_registry import ToolResult
from jellybean.settings import Settings
from jellybean.skills._shared.findings import all_findings
from jellybean.skills._shared.spec_tools import read_spec
from jellybean.skills.analista import tools as an
from jellybean.skills.critico import tools as cr
from jellybean.skills.detector import tools as dt
from jellybean.skills.feature_engineer import tools as fe
from jellybean.skills.orquestrador import tools as ot
from jellybean.skills.preparer import tools as pr
from jellybean.skills.profiler import integrity
from jellybean.skills.profiler import tools as pt

DATASET = "SUBSCRIBER_EVENTS"


def _write(root: Path) -> None:
    path = root / DATASET / "part-0.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    # Every event has a twin that differs only in its duration (two legs of one call), so no
    # column or pair of columns is a key; 300 rows are repeated exactly. One subscriber
    # (index 7) has a burst of long events.
    con.execute(
        f"""COPY (
            WITH base AS (
                SELECT '26801' || lpad((i % 400)::VARCHAR, 10, '0') AS "IMSI",
                       '35' || lpad((i % 380)::VARCHAR, 13, '0') AS "IMEI",
                       '+3519' || lpad((i % 400)::VARCHAR, 8, '0') AS "A_NUMBER",
                       '+3519' || lpad(((i * 7) % 900)::VARCHAR, 8, '0') AS "B_NUMBER",
                       'MO' AS "EVENT_TYPE",
                       TIMESTAMP '2026-10-01 00:00:00' + INTERVAL ((i % 600) * 2) MINUTE AS "EVENT_START_DATE",
                       CASE WHEN i % 400 = 7 THEN 500.0 ELSE (i % 5) * 10.0 END AS "EVENT_DURATION"
                FROM range(6000) t(i))
            SELECT * FROM base
            UNION ALL SELECT * REPLACE ("EVENT_DURATION" + 10 AS "EVENT_DURATION") FROM base
            UNION ALL SELECT * FROM base LIMIT 12300
        ) TO '{path}' (FORMAT parquet)"""
    )
    con.close()


@pytest.fixture
def run(settings: Settings) -> Runtime:
    _write(settings.data_root)
    ctx = RunContext.create(settings, seed=7, run_id="awkward")
    runtime = Runtime(ctx=ctx, llm=FakeLLM({}), human=AutoApproveHuman(), options={"instruction": "analyze"})
    with runtime.active():
        yield runtime


def test_awkward_dataset_reaches_the_report_without_errors(run: Runtime) -> None:
    results: list[tuple[str, ToolResult]] = []

    def step(name: str, result: ToolResult) -> ToolResult:
        results.append((name, result))
        return result

    # --- discovery
    step("init_run", ot.init_run(instruction="analyze the subscriber events"))
    step("datasets", ot.update_spec(patch={"datasets": [{"name": DATASET, "path": "p", "role": "primary"}]}))
    profile = step("profile_table", pt.profile_table(DATASET))
    types = step("infer_semantic_types", pt.infer_semantic_types(DATASET))
    grain = step("infer_grain", pt.infer_grain(DATASET))
    entities = step("suggest_entities", pt.suggest_entities(DATASET))
    temporal = step("check_temporal_coverage", pt.check_temporal_coverage(DATASET, "EVENT_START_DATE"))
    step("correlation_summary", pt.correlation_summary(DATASET))
    nulls = step("classify_nulls", integrity.classify_nulls(DATASET))
    dups = step("find_duplicate_keys", integrity.find_duplicate_keys(DATASET))
    invariants = step("discover_arithmetic_invariants", integrity.discover_arithmetic_invariants(DATASET))
    step("discover_self_references", integrity.discover_self_references(DATASET))
    issues = step(
        "compile_quality_issues",
        pt.compile_quality_issues(
            [a.artifact_id for a in (profile, grain, temporal, nulls, dups, invariants) if a.artifact_id]
        ),
    )
    step(
        "propose_spec",
        pt.propose_spec(
            [],
            grain.artifact_id,
            entities.artifact_id,
            [types.artifact_id],
            temporal=temporal.artifact_id,
            quality_issues=issues.artifact_id,
            checks=[a.artifact_id for a in (nulls, invariants, dups) if a.artifact_id],
        ),
    )
    step("update_spec entity", ot.update_spec(patch={"entity": {"column": "IMSI"}}))
    confirmation = step("prepare_spec_confirmation", ot.prepare_spec_confirmation())
    assert confirmation.summary["ready"], confirmation.summary
    grain_item = next(i for i in confirmation.summary["items"] if i["item"] == "grain")
    assert grain_item["value"] == "no unique key"
    step("ask_human spec", ot.ask_human(**confirmation.summary["question"]))
    step("confirm_spec", ot.confirm_spec())

    # --- cleaning (no join to materialise)
    plan = step("propose_cleaning_plan", pr.propose_cleaning_plan(issues.artifact_id))
    approval = step("prepare_cleaning_approval", ot.prepare_cleaning_approval())
    step("ask_human cleaning", ot.ask_human(**approval.summary["question"]))
    step("approve_cleaning_plan", ot.approve_cleaning_plan())
    step("apply_cleaning_plan", pr.apply_cleaning_plan())
    assert plan.artifact_id
    spec = read_spec()
    table = spec.enriched_artifact or spec.prepared.get(DATASET) or DATASET

    # --- features
    features = step(
        "build_entity_features",
        fe.build_entity_features(table, "IMSI", "EVENT_START_DATE", windows=["all", "day", "hour"]),
    )
    graph = step("build_graph", fe.build_graph(table, "IMSI", ["IMEI"], features=features.artifact_id))
    scaled = step("scale_encode", fe.scale_encode(graph.summary["features_table"] or features.artifact_id))
    embedding = step("reduce_dimensions", fe.reduce_dimensions(scaled.artifact_id, "pca", 10))
    step("feature_summary", fe.feature_summary(read_spec().features.table))
    sources = {d["source"] for d in _dictionary()}
    assert {"A_NUMBER", "B_NUMBER", "EVENT_TYPE", "IMEI"} <= sources

    # --- detection
    runs = [
        step(f"outlier {m}", dt.run_outlier_detector(scaled.artifact_id, m)).artifact_id
        for m in ("isolation_forest", "ecod", "lof")
    ]
    runs.append(step("clustering", dt.run_clustering(embedding.artifact_id, "hdbscan")).artifact_id)
    runs.append(step("graph", dt.run_graph_analysis(read_spec().features.graph, "components")).artifact_id)
    runs.append(
        step("timeseries count", dt.run_timeseries_anomaly(table, "EVENT_START_DATE", "count", "IMEI")).artifact_id
    )
    runs.append(
        step(
            "timeseries duration",
            dt.run_timeseries_anomaly(table, "EVENT_START_DATE", "EVENT_DURATION", None, "day"),
        ).artifact_id
    )
    step("consolidate_findings", dt.consolidate_findings([r for r in runs if r]))

    # --- critic: every finding ends with a verdict
    for f in all_findings():
        step(f"stability {f.id}", cr.stability_check(f.id, n_runs=2))
        step(f"ensemble {f.id}", cr.ensemble_agreement(f.id))
        step(f"artifact {f.id}", cr.artifact_check(f.id))
        step(f"explain {f.id}", cr.explain_by_reference(f.id))
        step(f"registry {f.id}", cr.match_pattern_registry(f.id))
    for f in all_findings():
        step(f"verdict {f.id}", cr.set_verdict(f.id, _verdict(f), {"policy": "test"}))
    assert all(f.critic_verdict for f in all_findings())

    # --- analyst and report
    for f in all_findings():
        if f.critic_verdict == "confirmed":
            step(f"describe {f.id}", an.describe_group(f.id))
            if f.type != "temporal":
                step(f"rules {f.id}", an.fit_surrogate_rules(f.id))
            step(f"exemplars {f.id}", an.get_exemplars(f.id, 3))
            step(f"write {f.id}", an.write_finding(f.id, "Description written by the test with tool numbers only."))
    step("review_findings", ot.review_findings())
    report = step("write_report", ot.write_report())

    errors = [(name, r.summary.get("error")) for name, r in results if r.status != "ok"]
    assert not errors, errors
    text = (run.ctx.run_dir / "report.md").read_text(encoding="utf-8")
    assert report.status == "ok" and "no unique key" in text


def _dictionary() -> list[dict[str, Any]]:
    from jellybean.skills._shared.artifacts import ArtifactStore

    return ArtifactStore().load_json(read_spec().features.dictionary)["features"]


def _verdict(f: Any) -> str:
    """The critic's rules, applied in order (as its SKILL.md states them)."""
    if f.checks["artifact"]["suspected"]:
        return "artifact"
    stability = f.checks["stability"].get("stability")
    if stability is None:
        return "weak"
    if stability < 0.6:
        return "unstable"
    if (f.checks.get("explanation") or {}).get("explained"):
        return "explained"
    if (f.checks.get("explanation") or {}).get("attributed_to"):
        return "weak"
    if f.checks["artifact"].get("low_support_share", 0.0) >= 0.5:
        return "weak"
    return "confirmed"
