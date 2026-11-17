# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Phase 5 verification on the dummy data, without an LLM.

Runs detector -> critic checks -> verdicts -> analyst -> pattern registry with
the deterministic tools, on top of the shared phase-4 pipeline, and checks the
acceptance criteria of spec section 10 that do not need an LLM:

* at least 4 of the planted anomalies have recall >= 0.6 in confirmed findings;
* the unit-error trap gets the verdict "artifact";
* anomaly F is found thanks to the enrichment (contextual) and not by global methods
  (its full recall target is checked on the default data by evaluate_run, phase 6);
* a confirmed pattern with positive feedback is classified "known" afterwards.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from jellybean.runtime.context import RunContext, use_run
from jellybean.skills._shared.findings import all_findings, finding_entities, load_finding, save_finding
from jellybean.skills.analista import tools as an
from jellybean.skills.critico import tools as cr


def _members(ctx: RunContext, f: Any) -> set[str]:
    ids = finding_entities(f, ctx)
    return set(ids[ids.columns[0]].astype(str))


def test_every_finding_has_a_verdict(phase5: dict) -> None:
    with use_run(phase5["ctx"]):
        findings = all_findings()
    assert 3 <= len(findings) <= 12
    assert all(f.critic_verdict for f in findings)
    assert phase5["consolidated"].status == "ok"


def test_recall_of_planted_anomalies(phase5: dict) -> None:
    ctx = phase5["ctx"]
    gt = pd.read_parquet(phase5["truth"] / "anomalies.parquet")
    with use_run(ctx):
        confirmed = [f for f in all_findings() if f.critic_verdict == "confirmed"]
        found: set[str] = set()
        for f in confirmed:
            found |= _members(ctx, f)
    recall = {a: len(set(block.entity_id) & found) / len(block) for a, block in gt.groupby("anomaly_id")}
    traffic = {a: r for a, r in recall.items() if a != "D"}  # D lives in orders (another run)
    # On failure, show where each anomaly was lost (spec section 10).
    with use_run(ctx):
        trail = [
            (
                f.id,
                f.type,
                f.size,
                f.critic_verdict,
                f.methods,
                gt[gt.entity_id.isin(_members(ctx, f))].anomaly_id.value_counts().to_dict(),
                f.checks.get("artifact", {}).get("reasons", [])[:1],
                f.checks.get("artifact", {}).get("low_support_share"),
                f.checks.get("stability", {}).get("stability"),
            )
            for f in all_findings()
        ]
    assert sum(r >= 0.6 for r in traffic.values()) >= 4, (recall, trail)


def test_unit_error_trap_is_an_artifact(phase5: dict) -> None:
    with use_run(phase5["ctx"]):
        trap = [f for f in all_findings() if f.type == "temporal" and f.context.get("measure") == "duration_sec"]
    assert trap, "the daily duration spike should be a candidate finding"
    assert all(f.critic_verdict == "artifact" for f in trap)
    assert any("scale" in r for f in trap for r in f.checks["artifact"]["reasons"])


def test_no_confirmed_finding_is_suspected(phase5: dict) -> None:
    with use_run(phase5["ctx"]):
        for f in all_findings():
            if f.critic_verdict == "confirmed":
                assert not f.checks["artifact"]["suspected"]
                assert f.checks["artifact"].get("low_support_share", 0.0) < 0.5
                assert f.description and f.top_features


def test_f_needs_the_enrichment(phase5: dict) -> None:
    """F is found by the contextual (segment) method, not by global outlier methods."""
    ctx = phase5["ctx"]
    gt = pd.read_parquet(phase5["truth"] / "anomalies.parquet")
    f_ids = set(gt[gt.anomaly_id == "F"].entity_id)
    with use_run(ctx):
        contextual = [f for f in all_findings() if f.context.get("contextual_only") and f.critic_verdict == "confirmed"]
        global_only = [f for f in all_findings() if not any(m.startswith("contextual") for m in f.methods)]
        via_context = set().union(*(_members(ctx, f) for f in contextual)) if contextual else set()
        via_global = set().union(*(_members(ctx, f) for f in global_only)) if global_only else set()
    # F's signal (working-hours and international shares) needs enough calls per
    # customer: on the full default data (30 days) contextual recall is ~0.9; on this
    # reduced test data (14 days, 60% volume) it is ~0.5. What matters here is that F
    # is found through the enrichment and not by the global methods.
    context_recall = len(via_context & f_ids) / len(f_ids)
    global_recall = len(via_global & f_ids) / len(f_ids)
    assert context_recall >= 0.4 and context_recall >= 2 * global_recall, (context_recall, global_recall)


def test_verdict_guards(phase5: dict) -> None:
    with use_run(phase5["ctx"]):
        artifact = next(f for f in all_findings() if f.critic_verdict == "artifact")
        refused = cr.set_verdict(artifact.id, "confirmed", {})
        assert refused.status == "error" and "artifact" in refused.summary["error"]
        weak = [f for f in all_findings() if f.checks["artifact"].get("low_support_share", 0) >= 0.5]
        for f in weak:
            assert cr.set_verdict(f.id, "confirmed", {}).status == "error"


def test_registry_known_after_positive_feedback(phase5: dict) -> None:
    ctx = phase5["ctx"]
    with use_run(ctx):
        f = next(f for f in all_findings() if f.critic_verdict == "confirmed" and f.type != "temporal")
        assert an.register_pattern(f.id).status == "error"  # no feedback yet
        f = load_finding(f.id)
        f.human_feedback = "confirmed"
        save_finding(f)
        registered = an.register_pattern(f.id)
        assert registered.status == "ok"
        again = cr.match_pattern_registry(f.id)
        assert again.summary["novelty"] == "known" and again.summary["best_pattern"] == registered.summary["pattern"]
        other = next(
            (g for g in all_findings() if g.id != f.id and g.type == f.type and g.critic_verdict == "confirmed"), None
        )
        if other is not None:
            assert cr.match_pattern_registry(other.id).summary["novelty"] in ("new", "variant")


def test_analyst_outputs(phase5: dict) -> None:
    ctx = phase5["ctx"]
    with use_run(ctx):
        f = next(f for f in all_findings() if f.critic_verdict == "confirmed" and f.type != "temporal")
        assert f.rules and 0 < f.rules[0]["precision"] <= 1
        assert f.exemplars and all(str(next(iter(e.values()))).startswith("masked:") for e in f.exemplars)
        plot = an.make_plot(f.id, "feature_distributions")
        assert plot.status == "ok" and (ctx.run_dir / plot.summary["path"]).exists()
        assert an.make_plot(f.id, "timeseries").status == "error"
        temporal = next((g for g in all_findings() if g.type == "temporal"), None)
        if temporal is not None:
            assert an.make_plot(temporal.id, "timeseries").status == "ok"
        weak = next((g for g in all_findings() if g.critic_verdict != "confirmed"), None)
        if weak is not None:
            assert an.write_finding(weak.id, "x" * 30).status == "error"


def test_phase5_trace_hygiene(phase5: dict) -> None:
    import json

    events = [json.loads(line) for line in phase5["ctx"].trace_path.read_text().splitlines()]
    tools = [e for e in events if e["kind"] == "tool_call"]
    assert max(e["output_chars"] for e in tools) <= 4000
    assert max(e["raw_rows"] for e in tools) <= 20
