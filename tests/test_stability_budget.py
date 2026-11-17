# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""stability_check always produces an answer, and a finding always gets a verdict."""

from __future__ import annotations

import time
from typing import Any

import pytest

from jellybean.runtime.context import RunContext
from jellybean.skills._shared.findings import all_findings, load_finding
from jellybean.skills.critico import tools as cr
from tests.test_pipeline_resources import run_copy  # noqa: F401  (fixture: a private copy of the phase-5 run)


def _entity_finding() -> Any:
    return next(f for f in all_findings() if f.type != "temporal" and not f.checks["artifact"]["suspected"])


@pytest.fixture
def tight_budget(run_copy: RunContext, monkeypatch: pytest.MonkeyPatch) -> RunContext:  # noqa: F811
    """A 4 s tool limit, of which the stability check may use 1 s."""
    monkeypatch.setattr(run_copy.settings.config.limits, "tool_timeouts", {"stability_check": 4.0})
    monkeypatch.setattr(run_copy.settings.config.critic, "stability_budget_share", 0.25)
    cr.clear_redetect_cache()
    return run_copy


def _slow_redetect(seconds: float, monkeypatch: pytest.MonkeyPatch) -> None:
    real = cr._redetect

    def slow(*args: Any, **kwargs: Any) -> set[Any]:
        time.sleep(seconds)
        return real(*args, **kwargs)

    monkeypatch.setattr(cr, "_redetect", slow)


def test_partial_stability_when_the_budget_runs_out(run_copy: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    f = _entity_finding()
    monkeypatch.setattr(run_copy.settings.config.limits, "tool_timeouts", {"stability_check": 14.0})
    monkeypatch.setattr(run_copy.settings.config.critic, "stability_budget_share", 0.5)  # 7 s for the repetitions
    cr.clear_redetect_cache()

    def one_second(finding: Any, sample: set[Any], **kwargs: Any) -> set[Any]:
        time.sleep(1.0)  # stands for the model fits of one repetition
        return sample

    monkeypatch.setattr(cr, "_redetect", one_second)
    started = time.monotonic()
    result = cr.stability_check(f.id, n_runs=10)
    assert result.status == "ok" and time.monotonic() - started < 12
    summary = result.summary
    assert 1 <= summary["runs_completed"] < summary["runs_requested"] == 10
    assert summary["stability"] == 1.0 and summary["stopped_by"] in ("time budget", "timeout")
    assert load_finding(f.id).checks["stability"]["runs_completed"] == summary["runs_completed"]
    assert any("repetitions" in w for w in result.warnings)


def test_unmeasurable_stability_is_recorded_and_allows_a_weak_verdict(
    tight_budget: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = _entity_finding()
    _slow_redetect(30.0, monkeypatch)  # one repetition is slower than the whole budget
    started = time.monotonic()
    result = cr.stability_check(f.id)
    assert time.monotonic() - started < 3.5
    assert result.status == "ok" and result.summary["stability"] is None
    assert result.summary["reason"] == "timeout" and result.summary["runs_completed"] == 0
    assert any("weak" in w for w in result.warnings)
    assert load_finding(f.id).checks["stability"]["stability"] is None

    refused = cr.set_verdict(f.id, "confirmed", {"x": 1})
    assert refused.status == "error" and "not measured" in refused.summary["error"]
    assert cr.set_verdict(f.id, "explained", {"x": 1}).status == "error"
    recorded = cr.set_verdict(f.id, "weak", {"note": "checks pass"})
    assert recorded.status == "ok", recorded.summary
    evidence = load_finding(f.id).critic_evidence
    assert load_finding(f.id).critic_verdict == "weak"
    assert evidence["evidence"]["stability"] == "not measured (timeout)"


def test_a_check_killed_by_the_tool_timeout_still_counts_as_attempted(
    run_copy: RunContext,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    f = _entity_finding()
    monkeypatch.setattr(run_copy.settings.config.limits, "tool_timeouts", {"stability_check": 0.5})
    monkeypatch.setattr(cr, "read_features", lambda *a, **k: time.sleep(3))  # stuck before any repetition
    assert cr.stability_check(f.id).summary.get("timed_out") is True
    check = load_finding(f.id).checks["stability"]
    assert check["stability"] is None and check["runs_completed"] == 0
    assert cr.set_verdict(f.id, "unstable", {"note": "stability check timed out"}).status == "ok"


def test_large_populations_are_checked_on_a_capped_sample(run_copy: RunContext, monkeypatch) -> None:  # noqa: F811
    f = _entity_finding()
    monkeypatch.setattr(run_copy.settings.config.critic, "stability_max_rows", 2000)
    cr.clear_redetect_cache()
    result = cr.stability_check(f.id, n_runs=2)
    assert result.status == "ok" and result.summary["runs_completed"] == 2
    assert 2000 <= result.summary["sample_rows"] <= 2000 + f.size  # the sample, plus the finding's members
    assert result.summary["stability"] is not None


def test_report_says_when_stability_was_not_measured(tight_budget: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    from jellybean.reporting import build_report

    f = next(x for x in all_findings() if x.critic_verdict == "confirmed" and x.type != "temporal")
    _slow_redetect(30.0, monkeypatch)
    cr.stability_check(f.id)
    assert "stability not measured" in build_report(tight_budget)
