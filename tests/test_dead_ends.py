# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Mandatory steps have an exit when their input is absent, empty or could not be computed.

One test per gate fixed in docs/dead-end-audit.md.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import pytest

from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext
from jellybean.runtime.human import AutoApproveHuman
from jellybean.runtime.llm_client import FakeLLM
from jellybean.settings import Settings
from jellybean.skills._shared import data_tools, detection
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.spec_tools import read_spec
from jellybean.skills.detector import tools as dt
from jellybean.skills.orquestrador import tools as ot
from jellybean.skills.preparer import tools as pr
from jellybean.skills.profiler import tools as pt
from tests.test_no_key_grain import write_events
from tests.test_runtime import FINISH, call


@pytest.fixture
def events_run(settings: Settings) -> Iterator[Runtime]:
    write_events(settings.data_root)
    ctx = RunContext.create(settings, seed=7, run_id="gates")
    runtime = Runtime(ctx=ctx, llm=FakeLLM({}), human=AutoApproveHuman(), options={"instruction": "analyze"})
    with runtime.active():
        ot.init_run(instruction="analyze the events")
        ot.update_spec(patch={"datasets": [{"name": "events", "path": "p", "role": "primary"}]})
        yield runtime


# ------------------------------------------------------------------ discovery
def test_no_entity_candidate_is_escalated_not_looped(events_run: Runtime) -> None:
    grain, types = pt.infer_grain("events"), pt.infer_semantic_types("events")
    none = ArtifactStore().save_json("entities_events", {"dataset": "events", "candidates": []})
    proposal = pt.propose_spec([], grain.artifact_id, none.id, [types.artifact_id])
    assert proposal.status == "ok" and any("entity" in w for w in proposal.warnings)
    assert read_spec().entity is None and read_spec().columns  # the rest of the proposal was written
    check = ot.prepare_spec_confirmation()
    assert check.status == "ok" and check.summary["ready"] is False
    assert any("--entity" in p and "needs_human" in p for p in check.summary["problems"])


def test_dataset_folders_that_cannot_be_used_are_reported(events_run: Runtime) -> None:
    bad = Path(events_run.ctx.settings.data_root) / "my events"
    bad.mkdir()
    duckdb.connect().execute(f"COPY (SELECT 1 AS x) TO '{bad / 'p.parquet'}' (FORMAT parquet)")
    listed = data_tools.list_datasets()
    assert listed.status == "ok" and any("my events" in w and "rename" in w for w in listed.warnings)


# -------------------------------------------------------------------- cleaning
def test_no_quality_issues_gives_an_empty_plan_that_can_be_approved_and_applied(events_run: Runtime) -> None:
    plan = pr.propose_cleaning_plan()
    assert plan.status == "ok" and plan.summary["ops"] == []
    approval = ot.prepare_cleaning_approval()
    assert approval.status == "ok"
    ot.ask_human(**approval.summary["question"])
    assert ot.approve_cleaning_plan().status == "ok"
    assert pr.apply_cleaning_plan().status == "ok"


def test_materialize_join_without_joins_is_a_skip(events_run: Runtime) -> None:
    result = pr.materialize_join(0)
    assert result.status == "ok" and result.summary["skipped"] is True
    assert "feature" in result.summary["next"]


# ------------------------------------------------------------------- detection
@pytest.mark.parametrize("method", ["hdbscan", "kmeans", "gmm"])
def test_clustering_a_handful_of_entities_gives_labels_not_an_error(method: str) -> None:
    X = np.random.default_rng(0).normal(size=(5, 3))
    labels = detection.cluster_labels(X, method, {}, 1)
    assert labels.shape == (5,)


def test_timeseries_without_any_timestamp_says_so(events_run: Runtime) -> None:
    path = Path(events_run.ctx.settings.data_root) / "undated" / "p.parquet"
    path.parent.mkdir()
    duckdb.connect().execute(
        f"COPY (SELECT range AS id, CAST(NULL AS TIMESTAMP) AS ts FROM range(50)) TO '{path}' (FORMAT parquet)"
    )
    result = dt.run_timeseries_anomaly("undated", "ts", "count")
    assert result.status == "error" and "no row has a value" in result.summary["error"]


# ---------------------------------------------------------------- agent runner
def test_an_agent_repeating_the_same_error_is_stopped_with_the_blocker(run_ctx: RunContext) -> None:
    script = {
        "profiler": [call("get_schema", dataset_or_artifact="nope"), call("get_schema", dataset_or_artifact="nope")]
        + [call("list_datasets")] * 5
        + [FINISH]
    }
    llm = FakeLLM(script)
    result = Runtime(ctx=run_ctx, llm=llm).run_agent("profiler", "t")
    assert result.status == "failed" and result.iterations == 2  # stopped at the second identical error
    assert any(i.startswith("blocked:get_schema") for i in result.issues)
    assert "Unknown dataset" in result.summary


def test_too_many_consecutive_errors_stop_the_agent(run_ctx: RunContext) -> None:
    run_ctx.settings.config.limits.max_consecutive_tool_errors = 3
    names = ["a", "b", "c", "d", "e"]
    script = {"profiler": [call("get_schema", dataset_or_artifact=n) for n in names] + [FINISH]}
    result = Runtime(ctx=run_ctx, llm=FakeLLM(script)).run_agent("profiler", "t")
    assert result.status == "failed" and result.iterations == 3
    assert "blocked:consecutive_errors" in result.issues
    events = [json.loads(line) for line in run_ctx.trace_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["kind"] == "agent_end" and "3 tool calls in a row" in events[-1]["reason"]
