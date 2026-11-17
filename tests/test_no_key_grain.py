# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Event tables without a unique key: "no key" is a valid grain, not an error."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
import yaml

from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext
from jellybean.runtime.llm_client import FakeLLM
from jellybean.settings import Settings
from jellybean.skills._shared.spec_tools import read_spec
from jellybean.skills.orquestrador import tools as ot
from jellybean.skills.preparer import tools as pr
from jellybean.skills.profiler import tools as pt
from jellybean.spec import GrainSpec, load_spec


def write_events(root: Path, name: str = "events") -> None:
    """Call events with no event id: several rows per subscriber and second, and exact repeats."""
    con = duckdb.connect()
    path = root / name / "part-0.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""COPY (
            SELECT '26801' || lpad((i % 400)::VARCHAR, 10, '0') AS IMSI,
                   '35' || lpad((i % 380)::VARCHAR, 13, '0') AS IMEI,
                   '+3519' || lpad((i % 400)::VARCHAR, 8, '0') AS A_NUMBER,
                   '+3519' || lpad(((i * 7) % 900)::VARCHAR, 8, '0') AS B_NUMBER,
                   CASE WHEN i % 3 = 0 THEN 'MO' WHEN i % 3 = 1 THEN 'MT' ELSE 'SMS' END AS EVENT_TYPE,
                   TIMESTAMP '2026-10-01 00:00:00' + INTERVAL ((i % 3000) * 37) SECOND AS EVENT_START_DATE,
                   (i % 113) * 1.5 AS EVENT_DURATION
            FROM (SELECT range AS i FROM range(6000) UNION ALL SELECT range FROM range(700))
            UNION ALL
            -- The other leg of the same call: identical but for the type (a legitimate second row).
            SELECT '26801' || lpad((i % 400)::VARCHAR, 10, '0'), '35' || lpad((i % 380)::VARCHAR, 13, '0'),
                   '+3519' || lpad((i % 400)::VARCHAR, 8, '0'), '+3519' || lpad(((i * 7) % 900)::VARCHAR, 8, '0'),
                   'MT', TIMESTAMP '2026-10-01 00:00:00' + INTERVAL ((i % 3000) * 37) SECOND, (i % 113) * 1.5
            FROM (SELECT range AS i FROM range(6000)) WHERE i % 3 = 0
        ) TO '{path}' (FORMAT parquet)"""
    )
    con.close()


@pytest.fixture
def events_run(settings: Settings) -> Iterator[Runtime]:
    write_events(settings.data_root)
    ctx = RunContext.create(settings, seed=7, run_id="nokey")
    runtime = Runtime(ctx=ctx, llm=FakeLLM({}), options={"instruction": "analyze the events"})
    with runtime.active():
        ot.init_run(instruction="analyze the events")
        ot.update_spec(patch={"datasets": [{"name": "events", "path": "p", "role": "primary"}]})
        yield runtime


def _propose() -> tuple:
    grain = pt.infer_grain("events")
    entities = pt.suggest_entities("events")
    types = pt.infer_semantic_types("events")
    temporal = pt.check_temporal_coverage("events", "EVENT_START_DATE")
    proposal = pt.propose_spec(
        [], grain.artifact_id, entities.artifact_id, [types.artifact_id], temporal=temporal.artifact_id
    )
    return grain, proposal


def test_no_key_table_goes_through_discovery_in_one_pass(events_run: Runtime) -> None:
    grain, proposal = _propose()
    assert grain.status == "ok"
    assert grain.summary["unique_key"] is False and grain.summary["exact_duplicate_rows"] == 700
    assert proposal.status == "ok", proposal.summary
    spec = read_spec()
    assert spec.grain == GrainSpec(dataset="events", columns=[], unique=False)
    assert spec.entity is not None and spec.columns and spec.time_column == "EVENT_START_DATE"
    confirmation = ot.prepare_spec_confirmation()
    assert confirmation.status == "ok" and confirmation.summary["ready"] is True
    item = next(i for i in confirmation.summary["items"] if i["item"] == "grain")
    assert item["value"] == "no unique key" and item["auto"] is False  # a human confirms it
    assert item["evidence"]["exact_duplicate_rows"] == 700
    assert "grain" in confirmation.summary["question"]["question"]


def test_propose_spec_refuses_an_artifact_that_is_not_a_grain(events_run: Runtime) -> None:
    entities = pt.suggest_entities("events")
    types = pt.infer_semantic_types("events")
    duplicates = pt.find_duplicate_keys("events")
    wrong = pt.propose_spec([], duplicates.artifact_id, entities.artifact_id, [types.artifact_id])
    assert wrong.status == "error" and "infer_grain" in wrong.summary["error"]


def test_specs_without_the_unique_field_still_load(events_run: Runtime, tmp_path: Path) -> None:
    data = yaml.safe_load(events_run.ctx.spec_path.read_text())
    data["grain"] = {"dataset": "events", "columns": ["IMSI", "EVENT_START_DATE"]}  # as written before the field
    old = tmp_path / "spec.yaml"
    old.write_text(yaml.safe_dump(data))
    assert load_spec(old).grain.unique is True
    with pytest.raises(ValueError, match="at least one column"):
        GrainSpec(dataset="events", columns=[])
    with pytest.raises(ValueError):
        GrainSpec(dataset="events", columns=[], unique=False, extra_field=1)


def test_a_non_unique_grain_is_never_used_as_dedupe_keys(events_run: Runtime) -> None:
    _propose()
    keys = ["A_NUMBER", "B_NUMBER", "EVENT_START_DATE", "EVENT_TYPE", "IMSI", "IMEI"]
    ot.update_spec(patch={"grain": {"dataset": "events", "columns": keys, "unique": False}})
    refused = pr.apply_transform("events", [{"op": "dedupe", "keys": keys}])
    assert refused.status == "error" and "not unique" in refused.summary["error"]
    assert pr.apply_transform("events", [{"op": "dedupe", "keys": list(reversed(keys))}]).status == "error"
    exact = pr.apply_transform("events", [{"op": "dedupe"}])  # exact duplicate rows: still allowed
    assert exact.status == "ok"


def test_report_and_summary_show_no_unique_key(events_run: Runtime) -> None:
    from jellybean.reporting import build_report

    _propose()
    assert read_spec().compact_summary()["grain"] == {"dataset": "events", "columns": [], "unique": False}
    assert "no unique key" in build_report(events_run.ctx)


def test_cleaning_plan_does_not_recommend_removing_repeated_events(events_run: Runtime) -> None:
    from jellybean.skills._shared.artifacts import ArtifactStore

    grain, _ = _propose()
    issues = pt.compile_quality_issues([grain.artifact_id])
    plan = pr.propose_cleaning_plan(issues.artifact_id)
    assert plan.status == "ok", plan.summary
    ops = ArtifactStore().load_json(plan.artifact_id)["ops"]
    dedupes = [o for o in ops if o["operation"]["op"] == "dedupe"]
    # Exact-row dedupe is offered, never recommended, and never keyed on anything.
    assert dedupes and all(o["requires_explicit_approval"] and not o["operation"].get("keys") for o in dedupes)


# ------------------------------------------------- shapes and messages (Task 3)
def test_update_spec_errors_say_what_was_sent_and_what_is_expected(events_run: Runtime) -> None:
    wrong = ot.update_spec(patch={"grain": ["IMSI", "EVENT_START_DATE"], "columns": {"IMSI": "identifier"}})
    assert wrong.status == "error"
    text = wrong.summary["error"]
    assert "pydantic.dev" not in text and "grain" in text and "columns.IMSI" in text
    assert "['IMSI', 'EVENT_START_DATE']" in text  # what was sent
    assert '"unique": false' in text  # the expected shape, with the no-key form
    assert '{"dataset": "events", "semantic_type": "identifier"}' in text


def test_every_documented_spec_example_is_valid(events_run: Runtime) -> None:
    from jellybean.skills._shared.spec_tools import SPEC_FIELD_EXAMPLES
    from jellybean.spec import apply_patch

    ot.update_spec(
        patch={
            "datasets": [
                {"name": "events", "path": "p", "role": "primary"},
                {"name": "customers", "path": "p", "role": "enrichment"},
            ]
        }
    )
    spec = read_spec()
    assert {"grain", "entity", "columns", "joins", "time_column"} <= set(SPEC_FIELD_EXAMPLES)
    for field, examples in SPEC_FIELD_EXAMPLES.items():
        for example in examples:
            apply_patch(spec, {field: example})  # raises when an example drifts from the model
    from jellybean.runtime.tool_registry import TOOL_SPEC_ATTR

    description = getattr(ot.update_spec, TOOL_SPEC_ATTR).description
    assert '"unique": false' in description and '"semantic_type"' in description  # shown to the agent


def test_a_human_change_to_no_unique_key_is_one_update(events_run: Runtime) -> None:
    _propose()
    ot.update_spec(patch={"grain": {"columns": ["IMSI", "EVENT_START_DATE"], "unique": True}})
    changed = ot.update_spec(patch={"grain": {"columns": [], "unique": False}})
    assert changed.status == "ok" and read_spec().grain == GrainSpec(dataset="events", columns=[], unique=False)
    assert ot.prepare_spec_confirmation().summary["ready"] is True


def test_profiler_evidence_is_readable_and_json_artifacts_are_not_tables(events_run: Runtime) -> None:
    from jellybean.skills._shared import data_tools

    types = pt.infer_semantic_types("events")
    read = data_tools.read_json_artifact(types.artifact_id)
    assert read.status == "ok" and "IMSI" in str(read.summary)
    assert "sample" not in str(read.summary).lower()  # the tool promises samples are never returned
    profile = pt.profile_table("events")
    assert data_tools.read_json_artifact(profile.artifact_id).status == "error"  # profiles hold sample values
    schema = data_tools.get_schema(types.artifact_id)
    assert schema.status == "error"
    assert "read_json_artifact" in schema.summary["error"] and "Datasets:" not in schema.summary["error"]
