# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Model fits in a worker process: same results, killed at the timeout, nothing left behind."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jellybean.runtime.context import RunContext
from jellybean.runtime.isolated import IsolatedWorkError, run_isolated
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.skills._shared import detection
from jellybean.skills._shared.artifacts import ArtifactStore

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def helpers(monkeypatch: pytest.MonkeyPatch) -> str:
    """Make tests/isolated_helpers.py importable by the worker processes."""
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    return "tests.isolated_helpers"


def _workers() -> list[str]:
    out = subprocess.run(["pgrep", "-f", "jellybean.runtime.isolated"], capture_output=True, text=True).stdout
    return out.split()


def test_worker_returns_arrays_and_reports_errors(run_ctx: RunContext, helpers: str) -> None:
    X = np.arange(12.0).reshape(4, 3)
    out = run_isolated(f"{helpers}:double", {"X": X}, {"factor": 2.0})
    assert np.array_equal(out["result"], X * 2) and int(out["rows"]) == 4
    with pytest.raises(IsolatedWorkError, match="no good") as raised:
        run_isolated(f"{helpers}:fail", {"X": X}, {})
    assert raised.value.kind == "ValueError"
    assert not list(run_ctx.tmp_dir.glob("work_*"))  # scratch folders are removed


def test_caller_regains_control_at_the_timeout_from_work_that_holds_the_gil(run_ctx: RunContext, helpers: str) -> None:
    before = set(_workers())

    @tool(timeout_s=1.0)
    def slow_fit() -> ToolResult:
        """A fit that would hold the interpreter lock for about a minute if it ran in-process."""
        run_isolated(f"{helpers}:hold_gil", {"X": np.zeros(1)}, {"n": 3_000_000_000})
        ArtifactStore().save_json("should_not_exist", {})
        return ToolResult.ok({})

    started = time.monotonic()
    result = slow_fit()
    elapsed = time.monotonic() - started
    assert result.status == "error" and result.summary["timed_out"] is True
    assert elapsed < 3.0, f"the caller waited {elapsed:.1f}s for a 1s limit"
    time.sleep(0.5)
    assert set(_workers()) <= before  # the worker was killed
    assert not ArtifactStore().list() and not list(run_ctx.tmp_dir.glob("work_*"))


def test_a_timed_out_call_leaves_no_artifact_even_when_its_work_finishes_later(run_ctx: RunContext) -> None:
    done: list[str] = []

    @tool(timeout_s=0.3)
    def lingering() -> ToolResult:
        """Pure-Python work that cannot be interrupted and then tries to save its result."""
        time.sleep(1.0)
        try:
            ArtifactStore().save_json("late_result", {"x": 1})
            done.append("saved")
        except BaseException as exc:
            done.append(type(exc).__name__)
            raise
        return ToolResult.ok({})

    assert lingering().summary["timed_out"] is True
    time.sleep(1.5)
    assert done == ["ToolCancelledError"] and not ArtifactStore().list()
    assert not list((run_ctx.run_dir / "artifacts").glob("late_result*"))


def test_one_tool_can_have_its_own_timeout(run_ctx: RunContext) -> None:
    @tool
    def patient() -> ToolResult:
        """Sleeps longer than the global limit."""
        time.sleep(0.6)
        return ToolResult.ok({})

    run_ctx.settings.config.limits.tool_timeout_s = 0.2
    assert patient().status == "error"
    run_ctx.settings.config.limits.tool_timeouts = {"patient": 5}
    assert patient().status == "ok"


@pytest.mark.parametrize("method", ["isolation_forest", "lof", "ecod", "hbos"])
def test_outlier_scores_are_identical_in_a_worker(run_ctx: RunContext, method: str) -> None:
    X = np.random.default_rng(4).normal(size=(600, 6))
    X[:4] += 7
    local = detection.outlier_scores(X, method, {}, 11, 300)
    run_ctx.settings.config.limits.isolate_above_rows = 100
    assert np.array_equal(detection.outlier_scores(X, method, {}, 11, 300), local)


@pytest.mark.parametrize("method", ["hdbscan", "kmeans", "gmm"])
def test_cluster_labels_and_projection_are_identical_in_a_worker(run_ctx: RunContext, method: str) -> None:
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(c, 0.3, size=(200, 4)) for c in (0, 4, 9)])
    local = detection.cluster_labels(X, method, {"n_clusters": 3, "n_components": 3}, 3, max_rows=10_000)
    pca = detection.reduce_matrix(X, "pca", 2, 3)
    run_ctx.settings.config.limits.isolate_above_rows = 100
    assert np.array_equal(detection.cluster_labels(X, method, {"n_clusters": 3, "n_components": 3}, 3, 10_000), local)
    again = detection.reduce_matrix(X, "pca", 2, 3)
    assert np.array_equal(again["Z"], pca["Z"]) and np.array_equal(again["ratios"], pca["ratios"])


def test_errors_of_the_method_come_back_as_detection_errors(run_ctx: RunContext) -> None:
    run_ctx.settings.config.limits.isolate_above_rows = 10
    X = pd.DataFrame(np.zeros((50, 2))).to_numpy()
    with pytest.raises(detection.DetectionError, match="Unknown outlier method"):
        detection.outlier_scores(X, "nope", {}, 1, 100)
