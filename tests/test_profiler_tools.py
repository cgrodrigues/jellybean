# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiler tools over the small fixtures (see conftest.py).

Fixture recap: ``events.phone`` is ``+35191`` + 7 digits (50 distinct values) and
``customers.customer_id`` is ``91`` + 7 digits, so stripping ``+351`` makes them join.
"""

from __future__ import annotations

import pytest
import yaml

from jellybean.runtime.context import RunContext
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills.profiler import tools as pt


def test_profile_table_reports_columns_and_artifact(run_ctx: RunContext) -> None:
    result = pt.profile_table("events")
    assert result.status == "ok"
    assert result.summary["rows"] == 1000
    columns = {c["name"]: c for c in result.summary["columns"]}
    assert columns["phone"]["masked"] is True
    assert all(str(v).startswith("masked:") for v, _ in columns["phone"]["top"])
    full = ArtifactStore().load_json(result.artifact_id)
    assert "histograms" in full and "amount" in full["histograms"]


def test_profile_table_detects_category_variants(run_ctx: RunContext) -> None:
    from jellybean.skills._shared import data_tools as dt

    saved = dt.run_sql(
        "SELECT * FROM (VALUES ('Business'), ('business'), (' BUSINESS  '), ('Consumer')) t(segment)", save_as="segs"
    )
    result = pt.profile_table(saved.artifact_id)
    kinds = {i["kind"] for i in result.summary["issues"]}
    assert "inconsistent_categories" in kinds


def test_infer_semantic_types(run_ctx: RunContext) -> None:
    result = pt.infer_semantic_types("events")
    types = {k: v[0] for k, v in result.summary["columns"].items()}
    assert types == {
        "event_id": "identifier",
        "phone": "identifier",
        "amount": "measure",
        "ts": "timestamp",
        "date": "timestamp",
    }
    assert "ts" in result.summary["time_column_candidates"]


def test_infer_grain(run_ctx: RunContext) -> None:
    result = pt.infer_grain("events")
    assert result.summary["recommended"] == ["event_id"]
    assert result.summary["exact_duplicate_rows"] == 0


def test_suggest_entities_and_join_bonus(run_ctx: RunContext) -> None:
    result = pt.suggest_entities("events")
    assert result.summary["recommended"] == "phone"
    assert not result.summary["candidates"][0]["join_key_bonus"]
    spec = {
        "run_id": run_ctx.run_id,
        "instruction": "x",
        "seed": 7,
        "datasets": [
            {"name": "events", "path": "p", "role": "primary"},
            {"name": "customers", "path": "p", "role": "enrichment"},
        ],
        "joins": [{"left": "events", "right": "customers", "keys": [["phone", "customer_id"]]}],
    }
    run_ctx.spec_path.write_text(yaml.safe_dump(spec))
    assert pt.suggest_entities("events").summary["candidates"][0]["join_key_bonus"]


def test_check_temporal_coverage(run_ctx: RunContext) -> None:
    result = pt.check_temporal_coverage("events", "ts")
    assert result.status == "ok" and result.summary["days"] == 2 and result.summary["gaps"] == 0
    assert pt.check_temporal_coverage("events", "nope").status == "error"


def test_join_discovery_and_test_join(run_ctx: RunContext) -> None:
    cand = pt.discover_join_candidates("events", "customers")
    best = cand.summary["best"]
    assert (best["left_column"], best["right_column"], best["normalization"]) == (
        "phone",
        "customer_id",
        "strip_prefix_plus351",
    )
    joined = pt.test_join(
        "events",
        "customers",
        [["phone", "customer_id"]],
        ["strip_prefix_plus351"],
        candidates_artifact=cand.artifact_id,
    )
    metrics = joined.summary["metrics"]
    assert metrics["coverage"] == 1.0 and metrics["cardinality"] == "N:1" and metrics["fanout"] == 1.0
    raw = pt.test_join("events", "customers", [["phone", "customer_id"]])
    assert raw.summary["metrics"]["coverage"] == 0.0


def test_test_join_errors(run_ctx: RunContext) -> None:
    assert pt.test_join("events", "customers", [["nope", "customer_id"]]).status == "error"
    assert pt.test_join("events", "customers", [["phone", "customer_id"]], ["explode"]).status == "error"
    assert pt.test_join("events", "customers", [["phone", "customer_id"]], candidates_artifact="x_v1").status == "error"
    assert pt.test_join("events", "missing", [["phone", "customer_id"]]).status == "error"


def test_normalize_key(run_ctx: RunContext) -> None:
    result = pt.normalize_key("events", "phone", match_against="customers.customer_id")
    assert result.summary["recommended"] == "strip_prefix_plus351"
    assert result.summary["value_shapes"][0][0] == "+999999999999"
    assert pt.normalize_key("events", "phone", strategy="explode").status == "error"
    assert pt.normalize_key("events", "phone", match_against="customers").status == "error"


def test_correlation_summary(run_ctx: RunContext) -> None:
    result = pt.correlation_summary("events")
    assert result.status == "ok"
    assert pt.correlation_summary("missing").status == "error"


def test_compile_quality_issues(run_ctx: RunContext) -> None:
    profile = pt.profile_table("events")
    joined = pt.test_join("events", "customers", [["phone", "customer_id"]])
    merged = pt.compile_quality_issues([profile.artifact_id, joined.artifact_id])
    assert merged.status == "ok" and merged.artifact_id == "quality_issues_v1"
    assert any(i["kind"] == "unmatched_rows" for i in merged.summary["issues"])
    assert pt.compile_quality_issues(["nope_v1"]).status == "error"


def test_scale_shift_of_a_whole_partition_is_one_issue() -> None:
    """With date/hour partitions, a day whose files are all shifted is reported once."""
    shifted = [
        {
            "column": "d",
            "file": f"date=2026-10-17/hour={h:02d}/part-0.parquet",
            "rows": 10,
            "file_median": 70000.0,
            "typical_median": 70.0,
            "factor": 1000.0,
        }
        for h in range(24)
    ]
    other = {
        "column": "d",
        "file": "date=2026-10-18/hour=05/part-0.parquet",
        "rows": 3,
        "file_median": 900.0,
        "typical_median": 70.0,
        "factor": 12.9,
    }
    files = [s["file"] for s in shifted] + [f"date=2026-10-18/hour={h:02d}/part-0.parquet" for h in range(24)]
    out = pt._collapse_by_partition([*shifted, other], files)
    assert out[0]["file"] == "date=2026-10-17/" and out[0]["files"] == 24 and out[0]["rows"] == 240
    assert out[1]["file"] == "date=2026-10-18/hour=05/part-0.parquet"  # a single shifted file stays as is


def test_timezone_aware_timestamps_are_profiled(run_ctx: RunContext, settings) -> None:
    """TIMESTAMPTZ values reach Python through pytz (a declared dependency)."""
    import duckdb

    folder = settings.data_root / "tz_events"
    folder.mkdir()
    con = duckdb.connect()
    con.execute(
        f"""COPY (SELECT range AS id, (range % 7) * 2.5 AS amount,
                         (TIMESTAMP '2026-10-01 00:00:00' + INTERVAL (range * 20) MINUTE) AT TIME ZONE 'UTC' AS ts
                  FROM range(600)) TO '{folder / "part-0.parquet"}' (FORMAT parquet)"""
    )
    assert con.execute(f"SELECT typeof(ts) FROM '{folder / 'part-0.parquet'}' LIMIT 1").fetchone()[0] == (
        "TIMESTAMP WITH TIME ZONE"
    )
    con.close()

    profile = pt.profile_table("tz_events")
    assert profile.status == "ok", profile.summary
    assert {c["name"]: c for c in profile.summary["columns"]}["ts"]["type"] == "TIMESTAMP WITH TIME ZONE"
    coverage = pt.check_temporal_coverage("tz_events", "ts")
    assert coverage.status == "ok", coverage.summary


def test_generated_data_has_a_timezone_aware_column(tmp_path) -> None:
    import duckdb

    from jellybean.datagen.telemetry import generate_telemetry

    generate_telemetry(tmp_path / "d", tmp_path / "t", machines=100, days=8, seed=1)
    kind = duckdb.sql(
        f"SELECT typeof(register_synced_at) FROM '{tmp_path / 'd' / 'machines' / 'part-0.parquet'}' LIMIT 1"
    ).fetchall()[0][0]
    assert kind == "TIMESTAMP WITH TIME ZONE"


def _single_day_dataset(settings, name: str, partitioned: bool) -> None:
    import duckdb

    folder = settings.data_root / name / ("date=2026-10-01" if partitioned else "")
    folder.mkdir(parents=True)
    con = duckdb.connect()
    con.execute(
        f"""COPY (SELECT range AS id, TIMESTAMP '2026-10-01 00:00:00' + INTERVAL (range * 2) MINUTE AS ts
                  FROM range(700)) TO '{folder / "part-0.parquet"}' (FORMAT parquet)"""
    )
    con.close()


@pytest.mark.parametrize("partitioned", [False, True])
def test_temporal_coverage_of_a_single_day(run_ctx: RunContext, settings, partitioned: bool) -> None:
    """One day (or one date partition) has no other day to compare its hourly profile with."""
    name = "one_day_part" if partitioned else "one_day"
    _single_day_dataset(settings, name, partitioned)
    result = pt.check_temporal_coverage(name, "ts")
    assert result.status == "ok", result.summary
    kinds = {i["kind"] for i in ArtifactStore().load_json(result.artifact_id)["issues"]}
    assert not kinds & {"hour_profile_shift", "possible_timezone_shift"}
