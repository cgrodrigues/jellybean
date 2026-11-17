# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Unknown-domain test: the system must work on data it was never designed for.

The data comes from ``scripts/generate_unknown_domain.py`` (industrial machine
telemetry), a generator independent of jellybean's telecom simulator. The test
runs the whole deterministic tool chain (no LLM), taking every decision the way
the agents' SKILL.md files prescribe and only from tool results, then compares
the outcome with the private ground truth.

Nothing in jellybean knows about machines, sensors or firmware; failures here
reveal hidden assumptions of the telecom scenario.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext, use_run
from jellybean.runtime.human import AutoApproveHuman
from jellybean.runtime.llm_client import FakeLLM
from jellybean.settings import AppConfig, EnvSettings, Settings
from jellybean.skills._shared import data_tools as dt
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.findings import all_findings, finding_entities
from jellybean.skills.critico import tools as cr
from jellybean.skills.detector import tools as de
from jellybean.skills.feature_engineer import tools as fe
from jellybean.skills.orquestrador import tools as ot
from jellybean.skills.preparer import tools as pr
from jellybean.skills.profiler import tools as pt
from tests.conftest import verdict_policy

GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "generate_unknown_domain.py"


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("generate_unknown_domain", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("unknown")
    truth = _load_generator().generate(base / "data", base / "truth")
    settings = Settings(
        env=EnvSettings(
            data_root=base / "data",
            workspace_root=base / "ws",
            ground_truth_root=base / "truth",
            config_file=base / "none.yaml",
        ),
        config=AppConfig(),
    )
    ctx = RunContext.create(settings, seed=42, run_id="unknown")
    out: dict[str, Any] = {"ctx": ctx, "truth": truth}
    runtime = Runtime(
        ctx=ctx,
        llm=FakeLLM({}),
        human=AutoApproveHuman(),
        options={"instruction": "analyse telemetry enriched with machines"},
    )
    with runtime.active():
        # Orchestrator: datasets and roles.
        ot.init_run(instruction="analyse telemetry enriched with machines")
        ot.update_spec(
            patch={
                "datasets": [
                    {"name": "telemetry", "path": "p", "role": "primary"},
                    {"name": "machines", "path": "p", "role": "enrichment"},
                ]
            }
        )
        # Profiler procedure.
        prof = {d: pt.profile_table(d) for d in ("telemetry", "machines")}
        types = {d: pt.infer_semantic_types(d) for d in ("telemetry", "machines")}
        grain, entities = pt.infer_grain("telemetry"), pt.suggest_entities("telemetry")
        partitions = set(dt.get_schema("telemetry").summary["partition_columns"])  # SKILL: not a partition column
        time_col = next(c for c in types["telemetry"].summary["time_column_candidates"] if c not in partitions)
        temporal = pt.check_temporal_coverage("telemetry", time_col)
        cand = pt.discover_join_candidates("telemetry", "machines")
        out["candidates"] = cand
        best = cand.summary["best"]
        join = pt.test_join(
            "telemetry",
            "machines",
            [[best["left_column"], best["right_column"]]],
            [best["normalization"]],
            candidates_artifact=cand.artifact_id,
        )
        issues = pt.compile_quality_issues(
            [
                prof["telemetry"].artifact_id,
                prof["machines"].artifact_id,
                grain.artifact_id,
                temporal.artifact_id,
                join.artifact_id,
            ]
        )
        out["issues"] = ArtifactStore(ctx).load_json(issues.artifact_id)["issues"]
        pt.propose_spec(
            [join.artifact_id],
            grain.artifact_id,
            entities.artifact_id,
            [types[d].artifact_id for d in types],
            temporal=temporal.artifact_id,
            quality_issues=issues.artifact_id,
        )
        # Control 1.
        out["control_1"] = ot.prepare_spec_confirmation()
        if not out["control_1"].summary.get("auto_confirmable"):
            ot.ask_human(**out["control_1"].summary["question"])
        ot.confirm_spec()
        # Preparer + control 2 (recommended option).
        pr.propose_cleaning_plan(issues.artifact_id)
        ot.ask_human(**ot.prepare_cleaning_approval().summary["question"])
        ot.approve_cleaning_plan()
        pr.apply_cleaning_plan()
        enriched = pr.materialize_join(0)
        # Feature engineer.
        spec = ot.read_spec()
        feats = fe.build_entity_features(enriched.artifact_id, spec.entity.column, spec.time_column)
        table = feats.artifact_id
        if feats.summary["counterparty_columns"]:  # SKILL: graph only with shared-resource identifiers
            graph = fe.build_graph(
                enriched.artifact_id, spec.entity.column, feats.summary["counterparty_columns"], features=table
            )
            table = graph.summary.get("features_table") or table
        scaled = fe.scale_encode(table)
        fe.reduce_dimensions(scaled.artifact_id, "pca", 10)
        # Detector procedure.
        spec = ot.read_spec()
        runs = [
            de.run_outlier_detector(spec.features.scaled, m).artifact_id for m in ("isolation_forest", "ecod", "lof")
        ]
        runs.append(de.run_clustering(spec.features.embedding, "hdbscan").artifact_id)
        if spec.features.graph:
            runs.append(de.run_graph_analysis(spec.features.graph, "components").artifact_id)
        raw = pd.read_parquet(ArtifactStore(ctx).path_of(spec.features.table))
        attrs = [
            c
            for c in raw.columns
            if c.startswith("attr_") and not pd.api.types.is_numeric_dtype(raw[c]) and 2 <= raw[c].nunique() <= 10
        ]
        runs += [de.run_contextual_outliers(spec.features.table, a).artifact_id for a in attrs[:2]]
        for measure in feats.summary["measure_columns"]:
            ts = de.run_timeseries_anomaly(spec.enriched_artifact, spec.time_column, measure, None, "day")
            if ts.status == "ok":
                runs.append(ts.artifact_id)
        out["consolidated"] = de.consolidate_findings([r for r in runs if r])
        # Critic: one set of checks per finding, then the SKILL.md verdict policy.
        for f in all_findings():
            cr.stability_check(f.id, n_runs=3)
            cr.ensemble_agreement(f.id)
            cr.artifact_check(f.id)
            cr.match_pattern_registry(f.id)
            cr.explain_by_reference(f.id)
        for f in all_findings():
            cr.set_verdict(f.id, verdict_policy(f), {"policy": "SKILL.md"})
        findings = all_findings()
        out["findings"] = findings
        out["members"] = {
            f.id: set(finding_entities(f)[finding_entities(f).columns[0]].astype(str))
            for f in findings
            if f.type != "temporal"
        }
    return out


def _confirmed_entities(run: dict[str, Any]) -> set[str]:
    return set().union(
        *(run["members"][f.id] for f in run["findings"] if f.critic_verdict == "confirmed" and f.id in run["members"])
    )


def _trail(run: dict[str, Any]) -> list:
    """Where each anomaly went (shown on failure)."""
    planted = run["truth"]["anomalies"]
    return [
        (
            f.id,
            f.type,
            f.size,
            f.critic_verdict,
            f.methods,
            {
                k: len(set(v) & run["members"].get(f.id, set()))
                for k, v in planted.items()
                if set(v) & run["members"].get(f.id, set())
            },
        )
        for f in run["findings"]
    ]


# ------------------------------------------------------------------------- discovery
def test_join_with_a_hyphenated_prefix_is_discovered(run: dict[str, Any]) -> None:
    best = run["candidates"].summary["best"]
    expected = run["truth"]["join"]
    assert (best["left_column"], best["right_column"]) == (expected["left"][1], expected["right"][1])
    assert best["normalization"] == f"strip_prefix:{expected['prefix']}"
    assert run["control_1"].summary["auto_confirmable"] is True


def test_spec_is_inferred(run: dict[str, Any]) -> None:
    with use_run(run["ctx"]):
        spec = ot.read_spec()
    assert spec.grain.columns == run["truth"]["grain"]
    assert spec.entity.column == run["truth"]["entity"]
    assert spec.time_column == run["truth"]["time_column"]


def test_quality_problems_are_detected(run: dict[str, Any]) -> None:
    found = {(i["kind"], i.get("column")) for i in run["issues"]}
    assert ("duplicate_rows", None) in found
    assert ("nulls", "vibration_mm_s") in found
    assert ("inconsistent_categories", "status") in found
    shift = next(i for i in run["issues"] if i["kind"] == "scale_shift_by_file")
    assert shift["column"] == "pressure_bar" and run["truth"]["quality"]["kpa_day"] in shift["detail"]["file"]


# ------------------------------------------------------------------------- detection
@pytest.mark.parametrize(("anomaly", "minimum"), [("S1", 0.9), ("S2", 0.9), ("S5", 0.9), ("S4", 0.6)])
def test_planted_anomalies_are_confirmed(run: dict[str, Any], anomaly: str, minimum: float) -> None:
    planted = set(run["truth"]["anomalies"][anomaly])
    recall = len(planted & _confirmed_entities(run)) / len(planted)
    assert recall >= minimum, (anomaly, recall, _trail(run))


def test_power_surge_day_is_found(run: dict[str, Any]) -> None:
    day = run["truth"]["S3"]["date"]
    hits = [
        f
        for f in run["findings"]
        if f.type == "temporal"
        and f.critic_verdict == "confirmed"
        and any(w["start"].startswith(day) for w in f.context.get("windows", []))
    ]
    assert hits, _trail(run)


def test_unit_error_day_is_an_artifact(run: dict[str, Any]) -> None:
    day = run["truth"]["quality"]["kpa_day"]
    trap = [
        f
        for f in run["findings"]
        if f.type == "temporal" and any(w["start"].startswith(day) for w in f.context.get("windows", []))
    ]
    assert trap and all(f.critic_verdict == "artifact" for f in trap)


def test_precision_floor(run: dict[str, Any]) -> None:
    confirmed = _confirmed_entities(run)
    planted = set().union(*(set(v) for k, v in run["truth"]["anomalies"].items() if k != "decoy"))
    precision = len(confirmed & planted) / len(confirmed)
    assert precision >= 0.4, (precision, _trail(run))


def test_tool_outputs_stay_within_limits(run: dict[str, Any]) -> None:
    events = [json.loads(line) for line in run["ctx"].trace_path.read_text().splitlines()]
    tools = [e for e in events if e["kind"] == "tool_call"]
    assert max(e["output_chars"] for e in tools) <= 4000 and max(e["raw_rows"] for e in tools) <= 20
    assert not [e for e in tools if e["status"] == "error"]


# ------------------------------------------------------------------- known limitation
@pytest.mark.xfail(
    strict=True,
    reason="The explained verdict finds causes in reference tables only: legitimate "
    "maintenance is a real, stable pattern and gets confirmed.",
)
def test_legitimate_maintenance_is_not_confirmed(run: dict[str, Any]) -> None:
    decoy = set(run["truth"]["anomalies"]["decoy"])
    assert not decoy & _confirmed_entities(run)
