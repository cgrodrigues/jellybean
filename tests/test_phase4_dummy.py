"""Phase 4 verification on the dummy data, without an LLM.

Runs the deterministic tool chain profiler -> control 1 -> cleaning plan ->
control 2 (recommended) -> apply -> join -> features -> graph, and checks the
enriched table and the feature table against the ground truth.
"""

from __future__ import annotations

import json

import duckdb
import pandas as pd
import pytest

from jellybean.runtime.context import RunContext
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.spec import load_spec


@pytest.fixture
def pipeline(phase4_pipeline: dict) -> dict:
    """The shared phase-4 tool chain (see conftest.py)."""
    return phase4_pipeline


def _sql(ctx: RunContext, artifact_id: str, sql: str) -> list:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{ArtifactStore(ctx).path_of(artifact_id)}')")
    return con.execute(sql).fetchall()


def test_plan_follows_policies(pipeline: dict) -> None:
    truth = {i["id"]: i for i in json.loads((pipeline["truth"] / "quality_issues.json").read_text())}
    plan = ArtifactStore(pipeline["ctx"]).load_json(pipeline["plan"].artifact_id)
    by_op = {(o["dataset"], o["operation"]["op"]): o for o in plan["ops"]}
    assert ("traffic", "dedupe") in by_op and ("traffic", "drop_rows") in by_op
    assert by_op[("cust_segmentation", "dedupe")]["operation"]["keep"] == "latest"
    assert ("cust_segmentation", "normalize") in by_op
    fix = by_op[("traffic", "fix_units")]
    assert fix["requires_explicit_approval"] and fix["operation"]["factor"] == 1000
    assert truth["duration_in_ms"]["file"].endswith(fix["operation"]["file"])
    assert not any(o["operation"]["op"] == "impute" for o in plan["ops"])  # identifiers never imputed
    # The false candidate's duplicated region_code keys are ignored, never deduplicated.
    assert all(o["operation"].get("keys") != ["region_code"] for o in plan["ops"])
    # Recommended approval leaves the inferred unit fix out.
    assert pipeline["approval"].summary["excluded_ops"] == [fix["id"]]


def test_cleaned_tables(pipeline: dict) -> None:
    ctx = pipeline["ctx"]
    spec = load_spec(ctx.spec_path)
    assert spec.cleaning_plan.status == "applied"
    traffic, seg = spec.prepared["traffic"], spec.prepared["cust_segmentation"]
    rows, ids, negatives, null_cells = _sql(
        ctx,
        traffic,
        "SELECT count(*), count(DISTINCT call_id), "
        "count(*) FILTER (WHERE duration_sec < 0), count(*) - count(cell_id) FROM t",
    )[0]
    assert rows == ids and negatives == 0 and null_cells > 0  # dedupe + drop, nulls kept
    seg_rows, seg_keys, segments = _sql(
        ctx, seg, "SELECT count(*), count(DISTINCT customer_id), list(DISTINCT segment ORDER BY segment) FROM t"
    )[0]
    assert seg_rows == seg_keys and segments == ["Business", "Consumer", "Premium", "Youth"]
    # The unit error is still present (not approved): the critic must catch it later.
    assert _sql(ctx, traffic, "SELECT max(duration_sec) FROM t")[0][0] > 100_000
    log = ArtifactStore(ctx).load_json  # noqa: F841 - logs are parquet; check they exist
    results = {r["dataset"]: r for r in pipeline["applied"].summary["results"]}
    assert results["traffic"]["transform_log"] and results["cust_segmentation"]["transform_log"]


def test_enriched_table(pipeline: dict) -> None:
    ctx, join = pipeline["ctx"], pipeline["join"]
    clean_rows = _sql(ctx, load_spec(ctx.spec_path).prepared["traffic"], "SELECT count(*) FROM t")[0][0]
    assert join.summary["rows"] == clean_rows  # no row lost or duplicated
    assert 0.95 < join.summary["coverage"] < 0.99 and join.summary["unmatched_rows"] > 0
    # The planted 3% unknown customers (among customers present in traffic) are unmatched.
    unmatched, total = _sql(
        ctx,
        join.artifact_id,
        "SELECT count(DISTINCT msisdn) FILTER (WHERE NOT matched_cust_segmentation), count(DISTINCT msisdn) FROM t",
    )[0]
    assert unmatched / total == pytest.approx(0.03, abs=0.003)
    assert "segment" in join.summary["added_columns"]


def test_feature_table_matches_ground_truth(pipeline: dict) -> None:
    ctx, truth = pipeline["ctx"], pipeline["truth"]
    table = pd.read_parquet(ArtifactStore(ctx).path_of(pipeline["graph"].summary["features_table"]))
    entities = _sql(ctx, pipeline["join"].artifact_id, "SELECT count(DISTINCT msisdn) FROM t")[0][0]
    assert len(table) == entities and table["msisdn"].is_unique  # one row per entity present in the data
    gt = pd.read_parquet(truth / "anomalies.parquet")
    group = {a: table[table.msisdn.isin(gt[gt.anomaly_id == a].entity_id)] for a in "ABCF"}
    normal = table[~table.msisdn.isin(gt.entity_id)]
    assert group["A"]["time_share_night"].median() > 5 * normal["time_share_night"].median()
    assert group["B"]["duration_sec_p50"].median() <= 3 < normal["duration_sec_p50"].median()
    assert (group["C"]["call_type_share_outgoing"] == 1).all()
    consumer_basic = normal[(normal.attr_segment == "Consumer") & (normal.attr_tariff_plan == "basic")]
    assert (group["F"]["attr_segment"] == "Consumer").all()
    assert group["F"]["time_share_work_hours"].median() > 1.5 * consumer_basic["time_share_work_hours"].median()


def test_graph_recovers_coordinated_group(pipeline: dict) -> None:
    ctx, truth = pipeline["ctx"], pipeline["truth"]
    graph = pipeline["graph"].summary
    assert graph["largest_components"][:3] == sorted(graph["largest_components"][:3], reverse=True)
    assert sum(graph["largest_components"]) == 40
    nodes = pd.read_parquet(ArtifactStore(ctx).path_of(graph["graph_features"]))
    linked = set(nodes[nodes.graph_degree > 0].msisdn)
    gt = pd.read_parquet(truth / "anomalies.parquet")
    assert linked == set(gt[gt.anomaly_id == "C"].entity_id)


def test_scaled_table_is_model_ready(pipeline: dict) -> None:
    ctx = pipeline["ctx"]
    scaled = pd.read_parquet(ArtifactStore(ctx).path_of(pipeline["scaled"].artifact_id))
    assert len(scaled) == pipeline["features"].summary["entities"]
    assert not scaled.drop(columns=["msisdn"]).isna().any().any()
    spec = load_spec(ctx.spec_path)
    assert spec.features.scaled == pipeline["scaled"].artifact_id and spec.phase == "features"
