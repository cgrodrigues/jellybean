# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Artifact ids built from column, dataset or method names accept any spelling of those names."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from jellybean.runtime.context import RunContext, use_run
from jellybean.settings import Settings
from jellybean.skills._shared.artifacts import ArtifactStore, prefix_part
from jellybean.skills.detector import tools as dt
from jellybean.skills.profiler import integrity
from jellybean.skills.profiler import tools as pt


@pytest.fixture
def odd_names(settings: Settings) -> RunContext:
    path = Path(settings.data_root) / "Call_Events" / "part-0.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        f"""COPY (
            SELECT 's' || (i % 40) AS "IMSI", 'p' || ((i * 7) % 40) AS "Other.Party",
                   CASE WHEN i % 2 = 0 THEN 'MO' ELSE 'MT' END AS "EVENT_TYPE",
                   TIMESTAMP '2026-10-01 00:00:00' + INTERVAL (i * 9) MINUTE AS "Event Start",
                   CASE WHEN i BETWEEN 1500 AND 1505 THEN 90000.0 ELSE 30.0 + (i % 11) END AS "EVENT_DURATION",
                   (i % 5) * 1.0 AS "amount.usd", (i % 5) * 2.0 AS "Amount Usd"
            FROM range(3200) t(i)
        ) TO '{path}' (FORMAT parquet)"""
    )
    con.close()
    ctx = RunContext.create(settings, seed=7, run_id="odd_names")
    with use_run(ctx):
        yield ctx


def test_prefix_part_makes_any_name_valid() -> None:
    assert prefix_part("EVENT_DURATION") == "event_duration"
    assert prefix_part("Amount.USD (net)") == "amount_usd_net"
    assert prefix_part("  ") == "x" and prefix_part("123") == "123" and prefix_part("ÁÉ") == "x"
    assert len(prefix_part("a" * 200)) <= 40


@pytest.mark.parametrize("measure", ["EVENT_DURATION", "amount.usd", "Amount Usd"])
def test_timeseries_accepts_any_measure_name(odd_names: RunContext, measure: str) -> None:
    result = dt.run_timeseries_anomaly("Call_Events", "Event Start", measure, granularity="hour")
    assert result.status == "ok", result.summary
    record = ArtifactStore().get(result.artifact_id)
    assert record.params["measure"] == measure  # the original name is kept
    assert result.summary["measure"] == measure


def test_names_that_differ_only_by_case_do_not_overwrite_each_other(odd_names: RunContext) -> None:
    first = dt.run_timeseries_anomaly("Call_Events", "Event Start", "amount.usd", granularity="hour")
    second = dt.run_timeseries_anomaly("Call_Events", "Event Start", "Amount Usd", granularity="hour")
    assert first.artifact_id != second.artifact_id
    store = ArtifactStore()
    assert {store.get(a).params["measure"] for a in (first.artifact_id, second.artifact_id)} == {
        "amount.usd",
        "Amount Usd",
    }


def test_profiler_tools_accept_odd_dataset_and_column_names(odd_names: RunContext) -> None:
    for result in (
        pt.profile_table("Call_Events"),
        pt.infer_semantic_types("Call_Events"),
        pt.infer_grain("Call_Events"),
        pt.suggest_entities("Call_Events"),
        pt.check_temporal_coverage("Call_Events", "Event Start"),
        pt.correlation_summary("Call_Events"),
        integrity.classify_nulls("Call_Events"),
        integrity.find_duplicate_keys("Call_Events"),
        integrity.discover_arithmetic_invariants("Call_Events"),
        integrity.infer_sign_convention("Call_Events", "EVENT_DURATION", "EVENT_TYPE"),
    ):
        assert result.status == "ok", result.summary


def test_a_series_without_anomalies_says_so(odd_names: RunContext) -> None:
    quiet = dt.run_timeseries_anomaly("Call_Events", "Event Start", "count", granularity="hour")
    assert quiet.status == "ok" and quiet.summary["anomalous_points"] == 0
    assert "no anomalous" in quiet.summary["result"]
    spike = dt.run_timeseries_anomaly("Call_Events", "Event Start", "EVENT_DURATION", granularity="hour")
    assert spike.summary["anomalous_points"] >= 1 and "anomalous" in spike.summary["result"]
    path = Path(odd_names.settings.data_root) / "short" / "part-0.parquet"
    path.parent.mkdir(parents=True)
    duckdb.connect().execute(
        f"COPY (SELECT TIMESTAMP '2026-10-01' + INTERVAL (i) HOUR AS ts FROM range(72) t(i)) "
        f"TO '{path}' (FORMAT parquet)"
    )
    short = dt.run_timeseries_anomaly("short", "ts", "count", granularity="day")
    assert short.status == "ok" and any("too few" in w for w in short.warnings)
