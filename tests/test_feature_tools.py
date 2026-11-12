"""Unit tests for feature engineer tools over the small fixtures."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from jellybean.runtime.context import RunContext
from jellybean.skills._shared import data_tools as dt
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills.feature_engineer import tools as fe
from jellybean.spec import load_spec


@pytest.fixture
def spec_ready(run_ctx: RunContext) -> RunContext:
    spec = {
        "run_id": run_ctx.run_id,
        "instruction": "x",
        "seed": 7,
        "datasets": [{"name": "events", "path": "p", "role": "primary"}],
        "entity": {"column": "phone"},
        "grain": {"dataset": "events", "columns": ["event_id"]},
        "time_column": "ts",
    }
    run_ctx.spec_path.write_text(yaml.safe_dump(spec))
    return run_ctx


@pytest.fixture
def enriched(spec_ready: RunContext) -> str:
    """Events plus a per-entity constant attribute and a shared 'device' column."""
    return dt.run_sql(
        """SELECT *, CASE WHEN right(phone, 1) IN ('0', '1') THEN 'Business' ELSE 'Consumer' END AS segment,
                  'D' || (CAST(right(phone, 2) AS INT) // 5) AS device
           FROM events""",
        save_as="enriched",
    ).artifact_id


def _table(ctx: RunContext, artifact_id: str) -> pd.DataFrame:
    return pd.read_parquet(ArtifactStore(ctx).path_of(artifact_id))


def test_build_entity_features(spec_ready: RunContext, enriched: str) -> None:
    result = fe.build_entity_features(enriched, "phone", "ts")
    assert result.status == "ok", result.summary
    assert result.summary["entities"] == 50
    assert "segment" in result.summary["attributes"]
    table = _table(spec_ready, result.artifact_id)
    assert table["n_events"].sum() == 1000 and table["phone"].is_unique
    for column in (
        "amount_sum",
        "amount_max_daily_sum",
        "time_share_night",
        "time_gap_mean_s",
        "attr_segment",
        "attr_device",
        "day_max_events",
    ):
        assert column in table.columns, column
    assert "event_id_distinct" not in table.columns  # grain excluded
    spec = load_spec(spec_ready.spec_path)
    assert spec.features.table == result.artifact_id and spec.features.dictionary
    names = {d["name"] for d in ArtifactStore(spec_ready).load_json(spec.features.dictionary)["features"]}
    assert set(table.columns) - {"phone"} <= names


def test_build_entity_features_errors(spec_ready: RunContext, enriched: str) -> None:
    assert fe.build_entity_features(enriched, "nope").status == "error"
    assert fe.build_entity_features("missing_v1", "phone").status == "error"


def test_graph_scale_reduce_summary(spec_ready: RunContext, enriched: str) -> None:
    features = fe.build_entity_features(enriched, "phone", "ts").artifact_id
    graph = fe.build_graph(enriched, "phone", ["device"], max_entities_per_value=10, features=features)
    assert graph.status == "ok", graph.summary
    assert graph.summary["largest_components"][0] == 5  # 5 phones per device
    extended = graph.summary["features_table"]
    table = _table(spec_ready, extended)
    assert {"graph_degree", "graph_component_size"} <= set(table.columns)

    scaled = fe.scale_encode(extended)
    assert scaled.status == "ok"
    scaled_df = _table(spec_ready, scaled.artifact_id)
    assert not scaled_df.drop(columns=["phone"]).isna().any().any()
    assert any(c.startswith("attr_segment_") for c in scaled_df.columns)

    pca = fe.reduce_dimensions(scaled.artifact_id, "pca", 3)
    assert pca.status == "ok" and len(pca.summary["explained_variance"]) == 3
    assert _table(spec_ready, pca.artifact_id).shape == (50, 4)

    summary = fe.feature_summary(extended)
    assert summary.status == "ok" and summary.summary["entities"] == 50
    spec = load_spec(spec_ready.spec_path)
    assert (spec.features.table, spec.features.scaled, spec.features.embedding) == (
        extended,
        scaled.artifact_id,
        pca.artifact_id,
    )


def test_graph_ignores_too_common_values(spec_ready: RunContext, enriched: str) -> None:
    graph = fe.build_graph(enriched, "phone", ["segment"], max_entities_per_value=5)
    assert graph.summary["edges"] == 0
    assert graph.summary["link_columns"][0]["values_ignored_too_common"] == 2


def test_non_feature_tables_are_rejected(spec_ready: RunContext, enriched: str) -> None:
    assert fe.scale_encode(enriched).status == "error"
    assert fe.feature_summary("missing_v1").status == "error"
