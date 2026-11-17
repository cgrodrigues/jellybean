# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared data tools over small fixtures."""

from __future__ import annotations

from jellybean.runtime.context import RunContext
from jellybean.skills._shared import data_tools as dt


def test_list_datasets(run_ctx: RunContext) -> None:
    result = dt.list_datasets()
    assert result.status == "ok"
    by_name = {d["name"]: d for d in result.summary["datasets"]}
    assert set(by_name) == {"events", "customers", "wide"}
    assert by_name["events"]["rows"] == 1000
    assert by_name["events"]["files"] == 2
    assert by_name["events"]["partition_columns"] == ["date"]
    assert by_name["customers"]["has_readme"] is True


def test_list_datasets_empty(settings) -> None:
    ctx = RunContext.create(settings, seed=1, run_id="empty")
    from jellybean.runtime.context import use_run

    with use_run(ctx):
        result = dt.list_datasets()
    assert result.summary["datasets"] == [] and result.warnings


def test_resolve_dataset_exact_and_fuzzy(run_ctx: RunContext) -> None:
    exact = dt.resolve_dataset(" Events ")
    assert exact.summary["match"] == "events"
    fuzzy = dt.resolve_dataset("customer")
    assert fuzzy.summary["match"] is None
    assert "customers" in fuzzy.summary["candidates"]
    unknown = dt.resolve_dataset("zzz")
    assert unknown.summary["match"] is None and unknown.summary["available"]


def test_get_schema_dataset_and_artifact(run_ctx: RunContext) -> None:
    schema = dt.get_schema("events")
    names = [c["name"] for c in schema.summary["columns"]]
    assert names == ["event_id", "phone", "amount", "ts", "date"]
    assert schema.summary["partition_columns"] == ["date"]

    saved = dt.run_sql("SELECT phone, sum(amount) AS total FROM events GROUP BY phone", save_as="totals")
    artifact_schema = dt.get_schema(saved.artifact_id)
    assert artifact_schema.summary["kind"] == "artifact"
    assert [c["name"] for c in artifact_schema.summary["columns"]] == ["phone", "total"]

    assert dt.get_schema("nope").status == "error"


def test_run_sql_preview_and_count(run_ctx: RunContext) -> None:
    result = dt.run_sql("SELECT event_id, amount FROM events ORDER BY event_id")
    assert result.status == "ok"
    assert result.summary["row_count"] == 1000
    assert len(result.summary["rows"]) == 20
    assert result.summary["columns"] == ["event_id", "amount"]
    assert result.warnings  # "Showing 20 of 1000 rows."


def test_run_sql_save_as_creates_artifact_with_lineage(run_ctx: RunContext) -> None:
    result = dt.run_sql("SELECT * FROM events WHERE amount > 100", save_as="big_events")
    assert result.artifact_id == "big_events_v1"
    info = dt.get_artifact_info("big_events_v1")
    assert info.summary["parents"] == ["events"]
    assert info.summary["created_by"] == {"agent": "system", "tool": "run_sql"}
    assert info.summary["row_count"] == result.summary["row_count"]
    # The artifact is now queryable by its id.
    again = dt.run_sql("SELECT count(*) AS n FROM big_events_v1")
    assert again.summary["rows"][0][0] == result.summary["row_count"]


def test_run_sql_masks_pii_in_preview(run_ctx: RunContext) -> None:
    result = dt.run_sql("SELECT phone, amount FROM events LIMIT 3")
    assert result.summary["masked_columns"] == ["phone"]
    assert all(str(r[0]).startswith("masked:") for r in result.summary["rows"])


def test_sample_rows_masks_and_is_deterministic(run_ctx: RunContext) -> None:
    first = dt.sample_rows("customers", n=5)
    second = dt.sample_rows("customers", n=5)
    assert first.summary["rows"] == second.summary["rows"]
    assert set(first.summary["masked_columns"]) == {"customer_id", "contact"}
    other_seed = dt.sample_rows("customers", n=5, seed=123)
    assert other_seed.summary["rows"] != first.summary["rows"]


def test_sample_rows_where_and_bounds(run_ctx: RunContext) -> None:
    result = dt.sample_rows("events", n=20, where="amount = 0")
    assert result.status == "ok"
    assert all(r[2] == 0 for r in result.summary["rows"])
    assert dt.sample_rows("events", n=50).status == "error"  # n is capped at 20 by validation


def test_read_dataset_notes(run_ctx: RunContext) -> None:
    notes = dt.read_dataset_notes("customers")
    assert "One row per customer" in notes.summary["notes"]
    missing = dt.read_dataset_notes("events")
    assert missing.summary["notes"] is None and missing.warnings
    assert dt.read_dataset_notes("nope").status == "error"


def test_get_artifact_info_unknown(run_ctx: RunContext) -> None:
    assert dt.get_artifact_info("nope_v1").status == "error"
