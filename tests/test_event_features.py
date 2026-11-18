# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Features of event tables: grain columns are analysed, counts per type, hourly window."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext
from jellybean.runtime.llm_client import FakeLLM
from jellybean.settings import Settings
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills.feature_engineer import tools as fe
from jellybean.skills.orquestrador import tools as ot
from tests.test_no_key_grain import write_events

KEY = ["A_NUMBER", "B_NUMBER", "EVENT_START_DATE", "EVENT_TYPE", "IMSI", "IMEI"]


@pytest.fixture
def events_run(settings: Settings) -> Iterator[Runtime]:
    write_events(settings.data_root)
    ctx = RunContext.create(settings, seed=7, run_id="features")
    runtime = Runtime(ctx=ctx, llm=FakeLLM({}), options={"instruction": "analyze the events"})
    with runtime.active():
        ot.init_run(instruction="analyze the events")
        ot.update_spec(patch={"datasets": [{"name": "events", "path": "p", "role": "primary"}]})
        yield runtime


def _sources(result) -> set[str]:
    dictionary = ArtifactStore().load_json(result.summary["dictionary"])["features"]
    return {d["source"] for d in dictionary}


@pytest.mark.parametrize("unique", [True, False])
def test_grain_columns_are_not_dropped_from_the_features(events_run: Runtime, unique: bool) -> None:
    ot.update_spec(patch={"grain": {"dataset": "events", "columns": KEY, "unique": unique}})
    result = fe.build_entity_features("events", "IMSI", "EVENT_START_DATE")
    assert result.status == "ok", result.summary
    assert {"A_NUMBER", "B_NUMBER", "EVENT_TYPE", "IMEI"} <= _sources(result)
    assert "IMSI" not in _sources(result)  # the entity itself is never a feature


def test_a_technical_row_id_is_still_excluded(run_ctx: RunContext) -> None:
    runtime = Runtime(ctx=run_ctx, llm=FakeLLM({}), options={"instruction": "i"})
    with runtime.active():
        ot.init_run(instruction="i")
        ot.update_spec(
            patch={
                "datasets": [{"name": "events", "path": "p", "role": "primary"}],
                "grain": {"dataset": "events", "columns": ["event_id"]},
            }
        )
        result = fe.build_entity_features("events", "phone", "ts")
        assert result.status == "ok" and "event_id" not in _sources(result)
        assert result.summary["excluded_row_ids"] == ["event_id"]


# ------------------------------------------------ counts and the hourly window
def _burst_events(root) -> None:
    """300 subscribers with 48 events over two days (half MO, half MT); one of them has
    8 extra MO events inside a single hour, which barely moves its MO share."""
    import duckdb

    con = duckdb.connect()
    path = root / "calls" / "part-0.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""COPY (
            SELECT 's' || lpad(s::VARCHAR, 4, '0') AS sub, CASE WHEN h % 2 = 0 THEN 'MO' ELSE 'MT' END AS kind,
                   TIMESTAMP '2026-10-01 00:10:00' + INTERVAL (h) HOUR AS ts, 60.0 + (s % 7) AS seconds
            FROM range(300) t(s), range(48) u(h)
            UNION ALL
            SELECT 's0007', 'MO', TIMESTAMP '2026-10-02 02:00:00' + INTERVAL (m) MINUTE, 61.0 FROM range(8) v(m)
        ) TO '{path}' (FORMAT parquet)"""
    )
    con.close()


def test_hourly_counts_per_type_separate_a_burst_that_shares_do_not(events_run: Runtime) -> None:
    import pandas as pd

    _burst_events(events_run.ctx.settings.data_root)
    result = fe.build_entity_features("calls", "sub", "ts", windows=["all", "day", "hour"])
    assert result.status == "ok", result.summary
    store = ArtifactStore()
    table = pd.read_parquet(store.path_of(result.artifact_id)).set_index("sub")
    dictionary = {d["name"]: d for d in store.load_json(result.summary["dictionary"])["features"]}

    assert (table["n_events"].drop("s0007") == 48).all() and table.loc["s0007", "n_events"] == 56
    # Counts per level, next to the shares.
    assert table.loc["s0007", "kind_count_mo"] == 32 and table.loc["s0001", "kind_count_mo"] == 24
    assert dictionary["kind_count_mo"]["source"] == "kind" and dictionary["kind_count_mo"]["template"] == "category"
    # The overall share hardly moves (0.57 against 0.50)...
    assert abs(table.loc["s0007", "kind_share_mo"] - 0.5) < 0.08
    # ...but the hourly count of that type stands far out from every other subscriber.
    hourly = table["kind_mo_max_per_hour"]
    assert hourly.loc["s0007"] == 9 and hourly.drop("s0007").max() == 1
    assert dictionary["kind_mo_max_per_hour"]["template"] == "window:hour"
    assert "clock hour" in dictionary["kind_mo_max_per_hour"]["formula"]
    assert table.loc["s0001", "kind_mo_mean_per_active_hour"] == pytest.approx(0.5)
    # Daily window: maximum and mean count per day of each level.
    assert table.loc["s0007", "kind_mo_max_per_day"] == 20 and table.loc["s0001", "kind_mo_max_per_day"] == 12
    assert table.loc["s0001", "kind_mo_mean_per_day"] == pytest.approx(12.0)


def test_the_hourly_window_is_opt_in_and_refuses_too_many_rows(events_run: Runtime, monkeypatch) -> None:
    _burst_events(events_run.ctx.settings.data_root)
    default = fe.build_entity_features("calls", "sub", "ts")
    names = {d["name"] for d in ArtifactStore().load_json(default.summary["dictionary"])["features"]}
    assert not {"kind_count_mo", "kind_mo_max_per_hour", "kind_mo_max_per_day"} & names and "kind_share_mo" in names
    monkeypatch.setattr(events_run.ctx.settings.config.features, "window_max_rows", 1000)
    capped = fe.build_entity_features("calls", "sub", "ts", windows=["all", "hour"])
    assert capped.status == "ok" and any("hour" in w and "window_max_rows" in w for w in capped.warnings)
    names = {d["name"] for d in ArtifactStore().load_json(capped.summary["dictionary"])["features"]}
    assert "kind_mo_max_per_hour" not in names and "time_max_events_per_hour" in names
