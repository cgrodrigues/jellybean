# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures: isolated settings, small Parquet fixtures, an active run."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from jellybean.runtime.context import RunContext, use_run
from jellybean.settings import AppConfig, EnvSettings, Settings, reset_settings


@pytest.fixture(autouse=True)
def _isolate_global_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Never let tests read the developer's real .env or data folders."""
    monkeypatch.chdir(tmp_path)
    for var in ("JELLYBEAN_DATA_ROOT", "JELLYBEAN_WORKSPACE_ROOT", "JELLYBEAN_GROUND_TRUTH_ROOT"):
        monkeypatch.delenv(var, raising=False)
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at temporary data/workspace/ground-truth folders."""
    env = EnvSettings(
        data_root=tmp_path / "data",
        workspace_root=tmp_path / "workspace",
        ground_truth_root=tmp_path / "ground_truth",
        config_file=tmp_path / "missing-config.yaml",
    )
    for folder in (env.data_root, env.workspace_root, env.ground_truth_root):
        folder.mkdir(parents=True, exist_ok=True)
    return Settings(env=env, config=AppConfig())


def _copy(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO '{path}' (FORMAT parquet)")


@pytest.fixture
def small_data(settings: Settings) -> Path:
    """Write three small datasets plus a secret ground-truth file.

    * ``events``    partitioned by ``date=``, with phone numbers (PII) and amounts;
    * ``customers`` one row per customer, with e-mails and a README;
    * ``wide``      300 rows x 30 long text columns, to stress the output limits.
    """
    root = settings.data_root
    con = duckdb.connect()
    for day in ("2026-10-01", "2026-10-02"):
        _copy(
            con,
            f"""SELECT 'e' || lpad(range::VARCHAR, 5, '0') || '{day[-2:]}' AS event_id,
                       '+35191' || lpad((range % 50)::VARCHAR, 7, '0') AS phone,
                       (range % 97) * 1.5 AS amount,
                       TIMESTAMP '{day} 00:00:00' + INTERVAL (range) MINUTE AS ts
                FROM range(500)""",
            root / "events" / f"date={day}" / "part-0.parquet",
        )
    _copy(
        con,
        """SELECT '91' || lpad(range::VARCHAR, 7, '0') AS customer_id,
                  'user' || range || '@example.com' AS contact,
                  CASE WHEN range % 3 = 0 THEN 'Business' ELSE 'Consumer' END AS segment
           FROM range(50)""",
        root / "customers" / "part-0.parquet",
    )
    (root / "customers" / "README.md").write_text("# customers\n\nOne row per customer.\n")
    cols = ", ".join(f"repeat('x', 200) || range AS col_{i}" for i in range(30))
    _copy(con, f"SELECT range AS id, {cols} FROM range(300)", root / "wide" / "part-0.parquet")
    _copy(con, "SELECT 'TOP SECRET' AS label", settings.ground_truth_root / "anomalies.parquet")
    con.close()
    return root


@pytest.fixture
def run_ctx(settings: Settings, small_data: Path) -> Iterator[RunContext]:
    """An active run over the small datasets."""
    ctx = RunContext.create(settings, seed=7, run_id="test_run")
    with use_run(ctx):
        yield ctx


@pytest.fixture(scope="session")
def dummy_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dummy data generated once per session (full population, fewer days/calls for speed).

    The section-8 join thresholds assume the default population of 20 000 customers.
    """
    from jellybean.datagen import GeneratorConfig, generate

    base = tmp_path_factory.mktemp("dummy")
    generate(GeneratorConfig(out=base / "data", truth=base / "truth", seed=42, customers=20_000, days=14, scale=0.6))
    return base


def dummy_settings(base: Path, workspace: Path) -> Settings:
    """Settings pointing at the session dummy data and a private workspace."""
    return Settings(
        env=EnvSettings(
            data_root=base / "data",
            workspace_root=workspace,
            ground_truth_root=base / "truth",
            config_file=base / "none.yaml",
        ),
        config=AppConfig(),
    )


@pytest.fixture(scope="session")
def phase4_pipeline(dummy_base: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """Deterministic tool chain up to the features (no LLM), shared by phase 4 and 5 tests."""
    from jellybean.runtime.agent_runner import Runtime
    from jellybean.runtime.llm_client import FakeLLM
    from jellybean.skills.feature_engineer import tools as fe
    from jellybean.skills.orquestrador import tools as ot
    from jellybean.skills.preparer import tools as pr
    from jellybean.skills.profiler import tools as pt

    settings = dummy_settings(dummy_base, tmp_path_factory.mktemp("ws4"))
    ctx = RunContext.create(settings, seed=42, run_id="phase4")
    runtime = Runtime(ctx=ctx, llm=FakeLLM({}), options={"instruction": "traffic + cust_segmentation"})
    out: dict = {"ctx": ctx, "truth": dummy_base / "truth"}
    with runtime.active():
        ot.init_run(instruction="traffic + cust_segmentation")
        ot.update_spec(
            patch={
                "datasets": [
                    {"name": "traffic", "path": "p", "role": "primary"},
                    {"name": "cust_segmentation", "path": "p", "role": "enrichment"},
                ]
            }
        )
        prof = {n: pt.profile_table(n).artifact_id for n in ("traffic", "cust_segmentation")}
        types = [pt.infer_semantic_types(n).artifact_id for n in ("traffic", "cust_segmentation")]
        grain, entities = pt.infer_grain("traffic").artifact_id, pt.suggest_entities("traffic").artifact_id
        temporal = pt.check_temporal_coverage("traffic", "start_time").artifact_id
        cand = pt.discover_join_candidates("traffic", "cust_segmentation")
        best = cand.summary["best"]
        join = pt.test_join(
            "traffic",
            "cust_segmentation",
            [[best["left_column"], best["right_column"]]],
            [best["normalization"]],
            candidates_artifact=cand.artifact_id,
        ).artifact_id
        # Also test the false candidate and compile its issues too (a real LLM did this):
        # the cleaning plan must ignore issues of joins that are not in the spec.
        false_join = pt.test_join("traffic", "cust_segmentation", [["cell_id", "region_code"]]).artifact_id
        issues = pt.compile_quality_issues([*prof.values(), grain, temporal, join, false_join]).artifact_id
        pt.propose_spec([join], grain, entities, types, temporal=temporal, quality_issues=issues)
        assert ot.confirm_spec().status == "ok"
        out["plan"] = pr.propose_cleaning_plan(issues)
        approval = ot.prepare_cleaning_approval().summary["question"]
        ot.ask_human(**approval)  # auto-approve -> recommended option
        out["approval"] = ot.approve_cleaning_plan()
        out["applied"] = pr.apply_cleaning_plan()
        out["join"] = pr.materialize_join(0)
        features = fe.build_entity_features(out["join"].artifact_id, "msisdn", "start_time")
        out["features"] = features
        out["graph"] = fe.build_graph(out["join"].artifact_id, "msisdn", ["imei"], features=features.artifact_id)
        out["scaled"] = fe.scale_encode(out["graph"].summary["features_table"])
        out["embedding"] = fe.reduce_dimensions(out["scaled"].artifact_id, "pca", 10)
    # Outside the ``with``: a session fixture that yields inside it would leave this run
    # active for every later test of the session (tests activate it with ``use_run``).
    yield out


def verdict_policy(f: Any) -> str:
    """The verdict rules of the critic's SKILL.md, applied deterministically."""
    artifact, stability = f.checks["artifact"], f.checks["stability"]["stability"]
    if artifact["suspected"]:
        return "artifact"
    if stability < 0.6:
        return "unstable"
    if (f.checks.get("explanation") or {}).get("explained"):
        return "explained"
    if (f.checks.get("explanation") or {}).get("attributed_to"):
        return "weak"
    if artifact.get("low_support_share", 0.0) >= 0.5:
        return "weak"
    ensemble = f.checks.get("ensemble", {})
    structural = any(m.startswith(("hdbscan", "graph", "contextual", "timeseries")) for m in f.methods)
    if f.type == "outlier_group" and not structural and ensemble.get("mean_votes", 0) < 2:
        return "weak"
    return "confirmed"


@pytest.fixture(scope="session")
def phase5(phase4_pipeline: dict) -> dict:
    """Phase 5 on top of the phase-4 run: detection, critic checks, verdict policy, analyst."""
    from jellybean.runtime.context import use_run
    from jellybean.skills._shared.findings import all_findings
    from jellybean.skills._shared.spec_tools import read_spec
    from jellybean.skills.analista import tools as an
    from jellybean.skills.critico import tools as cr
    from jellybean.skills.detector import tools as dt

    ctx: RunContext = phase4_pipeline["ctx"]
    with use_run(ctx):
        spec = read_spec()
        feats = spec.features
        runs = [dt.run_outlier_detector(feats.scaled, m).artifact_id for m in ("isolation_forest", "ecod", "lof")]
        runs.append(dt.run_clustering(feats.embedding, "hdbscan").artifact_id)
        runs.append(dt.run_graph_analysis(feats.graph, "components").artifact_id)
        runs.append(dt.run_contextual_outliers(feats.table, "attr_segment").artifact_id)
        runs.append(dt.run_timeseries_anomaly(spec.enriched_artifact, "start_time", "count", "cell_id").artifact_id)
        runs.append(
            dt.run_timeseries_anomaly(spec.enriched_artifact, "start_time", "duration_sec", None, "day").artifact_id
        )
        consolidated = dt.consolidate_findings(runs)
        for f in all_findings():
            cr.stability_check(f.id, n_runs=3)
            cr.ensemble_agreement(f.id)
            cr.artifact_check(f.id)
            cr.match_pattern_registry(f.id)
            cr.explain_by_reference(f.id)
        verdicts = {}
        for f in all_findings():
            result = cr.set_verdict(f.id, verdict_policy(f), {"policy": "test"})
            verdicts[f.id] = result.summary.get("verdict")
        for f in all_findings():
            if f.critic_verdict == "confirmed":
                an.describe_group(f.id)
                if f.type != "temporal":
                    an.fit_surrogate_rules(f.id)
                an.get_exemplars(f.id, 3)
                an.write_finding(f.id, "Description written by the phase 5 test with tool numbers only.")
    return {
        "ctx": ctx,
        "truth": phase4_pipeline["truth"],
        "runs": runs,
        "consolidated": consolidated,
        "verdicts": verdicts,
    }
