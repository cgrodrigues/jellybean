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
