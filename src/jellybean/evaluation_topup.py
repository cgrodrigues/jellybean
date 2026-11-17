# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Evaluation of a top-up run against its ground truth (``requirements_2.md`` section 11).

One analysis studies one entity, so a run measures the anomalies of its entity
and the discovery criteria; :func:`jellybean.evaluation.combine` merges the runs
of several entities (and a run without ``promotions`` for T8).

Per run:

* discovery: the four enrichment joins and the self-reference, the balance
  invariant, R1-R5 found by the profiler's issues (recall and precision of the
  best issue, from its SQL predicate), the sign error (TR2) as a systematic
  block apart from R1, no structural-null column imputed;
* anomalies of the run's entity: recall in confirmed findings, and where the
  others were lost;
* traps: TR1 (campaign day) and TR3 (retail chains) explained, TR2 fixed by the
  cleaning plan or ruled an artefact;
* T3 at the promotion level, and never as a confirmed group of its customers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from jellybean.reporting import trace_stats
from jellybean.runtime.context import RunContext, use_run
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.catalog import connect
from jellybean.skills._shared.findings import Finding, all_findings, finding_entities
from jellybean.spec import AnalysisSpec, load_spec

ANOMALIES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
BALANCE = {"amount", "balance_before", "balance_after"}


def _members(ctx: RunContext, f: Finding, entity: str | None) -> set[str]:
    """Entities of a finding; for a temporal finding grouped by the entity, its groups."""
    if f.type == "temporal":
        if f.context.get("group_by") == entity:
            return {str(w["group"]) for w in f.context.get("windows", [])}
        return set()
    ids = finding_entities(f, ctx)
    return set(ids[ids.columns[0]].astype(str))


def _query(ctx: RunContext, sql: str, params: list[Any] | None = None) -> list[tuple]:
    with use_run(ctx):
        con = connect()
        try:
            return con.execute(sql, params or []).fetchall()
        finally:
            con.close()


def _sum_terms(expression: str) -> set[str] | None:
    """{a, b, c} of "a + b = c" (order does not matter), else None."""
    left, _, right = expression.partition("=")
    terms = [t.strip() for t in left.split("+")] + [right.strip()]
    return set(terms) if len(terms) == 3 and all(terms) else None


def _discovery(ctx: RunContext, spec: AnalysisSpec, expected: dict[str, Any]) -> dict[str, Any]:
    datasets = {d.name for d in spec.datasets}
    joins = {}
    for exp in expected["joins"]:
        if exp.get("self_reference") or exp["left"] != "topup" or exp["right"] not in datasets:
            continue
        keys = [list(k) for k in exp["keys"]]
        norm = [n for n in exp["key_normalization"] if n != "none"]
        found = [
            j
            for j in spec.joins
            if j.right == exp["right"] and j.status == "confirmed" and [list(k) for k in j.keys] == keys
        ]
        joins[exp["right"]] = bool(found) and found[0].key_normalization == norm
    refs = {(r.dataset, r.column, r.ref_dataset, r.ref_column) for r in spec.references}
    self_ref = ("topup", "ref_movement_id", "topup", "movement_id") in refs
    invariant = any(_sum_terms(i.expression) == BALANCE for i in spec.invariants if i.dataset == "topup")
    return {"joins": joins, "self_reference": self_ref, "balance_invariant": invariant}


def _reconciliation(
    ctx: RunContext, issues: list[dict[str, Any]], recon: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    """Best issue (by recall + precision) for each reconciliation problem, from its predicate."""
    rows: dict[str, set[str]] = {}
    for issue in issues:
        predicate = issue.get("detail", {}).get("predicate")
        if predicate and issue["dataset"] == "topup":
            try:
                rows[issue["id"]] = {
                    r[0] for r in _query(ctx, f"SELECT DISTINCT movement_id FROM topup x WHERE {predicate}")
                }
            except Exception:
                rows[issue["id"]] = set()
    out = {}
    kinds = {i["id"]: (i["kind"], i.get("detail", {}).get("invariant")) for i in issues}
    for issue_id, block in recon.groupby("issue_id"):
        truth = set(block["movement_id"])
        best = {"recall": 0.0, "precision": 0.0, "issue": None}
        for qid, found in rows.items():
            hit = len(found & truth)
            recall, precision = hit / len(truth), (hit / len(found) if found else 0.0)
            if recall + precision > best["recall"] + best["precision"]:
                best = {"recall": round(recall, 4), "precision": round(precision, 4), "issue": qid, "kind": kinds[qid]}
        out[issue_id] = best
    return out, rows


def evaluate_topup(ctx: RunContext, truth: Path) -> dict[str, Any]:
    """Every top-up metric for one run."""
    spec = load_spec(ctx.spec_path)
    store = ArtifactStore(ctx)
    expected = yaml.safe_load((truth / "expected_spec_topup.yaml").read_text(encoding="utf-8"))
    gt = pd.read_parquet(truth / "anomalies.parquet")
    gt = gt[gt["dataset"] == "topup"]
    recon = pd.read_parquet(truth / "reconciliation_issues.parquet")
    traps = {
        t["id"]: t for t in json.loads((truth / "traps.json").read_text(encoding="utf-8")) if t["dataset"] == "topup"
    }
    datasets = sorted(d.name for d in spec.datasets)
    entity = spec.entity.column if spec.entity else None
    out: dict[str, Any] = {
        "scenario": "topup",
        "run_id": ctx.run_id,
        "phase": spec.phase,
        "entity": f"topup.{entity}" if entity else None,
        "datasets": datasets,
        "filters": spec.filters,
        "opaque_codes": bool(expected.get("opaque_codes")),
        "readme_notes": bool(spec.context_notes),
    }
    if spec.filters:
        out["note"] = "The run used row filters; anomalies outside the filtered rows cannot be found."

    # ---- discovery
    out["discovery"] = _discovery(ctx, spec, expected)
    issues_id = spec.evidence_artifacts.get("quality_issues")
    issues = store.load_json(issues_id)["issues"] if issues_id else []
    out["reconciliation"], _ = _reconciliation(ctx, issues, recon)
    tr2 = traps.get("TR2_sign_error", {})
    systematic = [
        i
        for i in issues
        if i["kind"] == "invariant_violation_systematic"
        and set(i["detail"].get("dates", [])) == set(tr2.get("dates", []))
    ]
    r1_issue = out["reconciliation"].get("R1", {}).get("issue")
    out["sign_error"] = {
        "reported_as_systematic": bool(systematic),
        "segment": systematic[0]["detail"].get("segment") if systematic else None,
        "separate_from_r1": bool(systematic) and r1_issue not in {i["id"] for i in systematic},
    }
    plan = store.load_json(spec.cleaning_plan.artifact_id) if spec.cleaning_plan.artifact_id else {"ops": []}
    structural = set(expected.get("structural_nulls", {}))
    imputed = {o["operation"].get("column") for o in plan["ops"] if o["operation"]["op"] == "impute"}
    out["structural_nulls_imputed"] = sorted(imputed & structural)
    flips = [o["id"] for o in plan["ops"] if o["operation"]["op"] == "flip_sign"]
    applied = spec.cleaning_plan.status == "applied"
    out["sign_error"]["fixed_by_cleaning"] = applied and any(o in spec.cleaning_plan.approved_ops for o in flips)

    # ---- findings
    findings = all_findings(ctx)
    members = {f.id: _members(ctx, f, entity) for f in findings}
    confirmed = [f for f in findings if f.critic_verdict == "confirmed"]
    confirmed_entities = set().union(*(members[f.id] for f in confirmed)) if confirmed else set()
    recall, trail = {}, {}
    for anomaly in ANOMALIES:
        block = gt[(gt["anomaly_id"] == anomaly) & (gt["entity_column"] != "movement_id")]
        column = block["entity_column"].iloc[0] if len(block) else None
        if column != entity:
            recall[anomaly] = None
            trail[anomaly] = f"not applicable: anomaly entity is '{column}', this run analyses '{entity}'"
            continue
        ids = set(block["entity_id"].astype(str))
        recall[anomaly] = round(len(ids & confirmed_entities) / len(ids), 4)
        hits = {f.id: (len(ids & members[f.id]), f.critic_verdict) for f in findings if ids & members[f.id]}
        trail[anomaly] = (
            f"confirmed (recall {recall[anomaly]})"
            if recall[anomaly] >= 0.6
            else f"in findings {hits}"
            if hits
            else "not in any finding"
        )
    out["recall"], out["where_lost"] = recall, trail
    planted = set(gt.loc[gt["entity_column"] == entity, "entity_id"].astype(str))
    out["precision"] = {
        "confirmed_entities": len(confirmed_entities),
        "entity_precision": round(len(confirmed_entities & planted) / len(confirmed_entities), 4)
        if confirmed_entities
        else None,
        "finding_precision": round(
            sum(len(members[f.id] & planted) >= 0.5 * len(members[f.id]) for f in confirmed) / len(confirmed), 4
        )
        if confirmed
        else None,
    }

    # ---- T3 at the promotion level, not as its customers
    t3_moves = gt.loc[(gt["anomaly_id"] == "T3") & (gt["entity_column"] == "movement_id"), "entity_id"].tolist()
    t3 = {"promotion_level": None, "customer_group_confirmed": None}
    if entity == "promo_id":
        t3["promotion_level"] = any("P017" in members[f.id] for f in confirmed)
    if entity == "msisdn" and t3_moves:
        customers = {
            r[0]
            for r in _query(
                ctx, "SELECT DISTINCT msisdn FROM topup WHERE movement_id IN (SELECT unnest(?))", [t3_moves]
            )
        }
        t3["customer_group_confirmed"] = any(
            members[f.id] and len(members[f.id] & customers) >= 0.5 * len(members[f.id]) for f in confirmed
        )
    out["t3"] = t3

    # ---- traps
    trap_results: dict[str, Any] = {}
    tr1 = traps.get("TR1_legitimate_campaign")
    if tr1 and entity == "promo_id":
        hit = [
            f
            for f in findings
            if f.type == "temporal"
            and "P009" in members[f.id]
            and any(str(w["start"]).startswith(tr1["date"]) for w in f.context.get("windows", []))
        ]
        verdicts = [f.critic_verdict for f in hit]
        trap_results["TR1"] = {
            "expected": "explained",
            "verdicts": verdicts,
            "pass": all(v == "explained" for v in verdicts),
            "surfaced": bool(hit),
        }
    tr3 = traps.get("TR3_retail_chains")
    if tr3 and entity == "dealer_id":
        chains = set(tr3["entity_ids"])
        hit = [f for f in findings if members[f.id] and len(members[f.id] & chains) >= 0.5 * len(members[f.id])]
        verdicts = [f.critic_verdict for f in hit]
        # Not surfacing the chains at all is also a pass: they are normal for their type.
        trap_results["TR3"] = {
            "expected": "explained",
            "verdicts": verdicts,
            "pass": all(v == "explained" for v in verdicts),
            "surfaced": bool(hit),
        }
    if tr2:
        dates = tr2["dates"]
        affected = (
            {
                r[0]
                for r in _query(
                    ctx,
                    "SELECT DISTINCT msisdn FROM topup WHERE upper(trim(channel)) = 'USSD' AND date::VARCHAR IN "
                    "(SELECT unnest(?)) AND balance_after <> balance_before + amount",
                    [dates],
                )
            }
            if entity == "msisdn"
            else set()
        )
        hit = [
            f
            for f in findings
            if affected and members[f.id] and len(members[f.id] & affected) >= 0.5 * len(members[f.id])
        ]
        verdicts = [f.critic_verdict for f in hit]
        fixed = out["sign_error"]["fixed_by_cleaning"]
        trap_results["TR2"] = {
            "expected": "artifact or fixed by the preparer",
            "fixed_by_cleaning": fixed,
            "verdicts": verdicts,
            "pass": fixed or all(v == "artifact" for v in verdicts),
        }
    out["traps"] = trap_results
    out["hygiene"] = trace_stats(ctx)

    # ---- acceptance (per run; combine() adds the criteria that need several runs)
    rec = out["reconciliation"]
    disc = out["discovery"]
    out["acceptance"] = {
        "joins_and_self_reference_discovered": all(disc["joins"].values()) and disc["self_reference"]
        if disc["joins"]
        else None,
        "balance_invariant_discovered": disc["balance_invariant"],
        "r1_to_r5_recall_0_9": all(rec.get(r, {}).get("recall", 0) >= 0.9 for r in ("R1", "R2", "R3", "R4", "R5")),
        "sign_error_systematic_and_apart_from_r1": out["sign_error"]["reported_as_systematic"]
        and out["sign_error"]["separate_from_r1"],
        "no_structural_null_imputed": not out["structural_nulls_imputed"],
        "t3_at_promotion_level": t3["promotion_level"],
        "t3_not_reported_as_customers": None
        if t3["customer_group_confirmed"] is None
        else not t3["customer_group_confirmed"],
        "tr1_tr3_explained": all(trap_results[t]["pass"] for t in ("TR1", "TR3") if t in trap_results)
        if {"TR1", "TR3"} & set(trap_results)
        else None,
        "tr2_artifact_or_fixed": trap_results.get("TR2", {}).get("pass"),
        "tool_outputs_within_limits": out["hygiene"].get("max_tool_output_chars", 0) <= 4000
        and out["hygiene"].get("max_raw_rows", 0) <= 20,
    }
    return out


def combine_topup(results: list[dict[str, Any]], best: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Acceptance criteria that need several top-up runs (one per entity, plus one without promotions)."""
    applicable = {a: b["recall"] for a, b in best.items() if a in ANOMALIES and b["recall"] is not None}
    with_promotions = [r for r in results if "promotions" in r["datasets"] and r["recall"].get("T8") is not None]
    without = [r for r in results if "promotions" not in r["datasets"] and r["recall"].get("T8") is not None]
    t8 = None
    if with_promotions and without:
        t8 = max(r["recall"]["T8"] for r in with_promotions) >= 0.6 and all(r["recall"]["T8"] < 0.6 for r in without)
    acceptance: dict[str, Any] = {}
    for name in results[0]["acceptance"]:
        values = [r["acceptance"][name] for r in results if r["acceptance"].get(name) is not None]
        acceptance[name] = all(values) if values else None
    acceptance["five_of_eight_anomalies_recall_0_6"] = (
        sum(v >= 0.6 for v in applicable.values()) >= 5 if len(applicable) == len(ANOMALIES) else None
    )
    acceptance["t8_needs_promotions"] = t8
    return {
        "acceptance": acceptance,
        "anomalies_evaluated": sorted(applicable),
        "anomalies_missing": sorted(set(ANOMALIES) - set(applicable)),
    }


def format_topup(result: dict[str, Any]) -> str:
    """Human-readable evaluation of one top-up run."""
    lines = [
        f"Evaluation of run {result['run_id']} (top-up, phase {result['phase']}, entity {result['entity']})",
        f"Datasets: {', '.join(result['datasets'])}; opaque codes: {result['opaque_codes']}; "
        f"README notes: {result['readme_notes']}",
        "",
    ]
    if result.get("filters"):
        lines += [f"Filters: {result['filters']} - {result.get('note', '')}", ""]
    d = result["discovery"]
    lines.append(
        f"Joins: {d['joins']}; self-reference: {d['self_reference']}; balance invariant: {d['balance_invariant']}"
    )
    lines.append("Reconciliation problems (best issue):")
    for issue, r in result["reconciliation"].items():
        lines.append(f"  {issue}: recall {r['recall']}, precision {r['precision']} ({r.get('kind')})")
    lines.append(f"Sign error: {result['sign_error']}")
    lines.append(f"Structural-null columns imputed: {result['structural_nulls_imputed'] or 'none'}")
    lines.append("Recall per anomaly (confirmed findings):")
    for anomaly, value in result["recall"].items():
        lines.append(f"  {anomaly}: {value if value is not None else 'n/a'}  <- {result['where_lost'][anomaly]}")
    lines.append(f"Precision: {result['precision']}")
    lines.append(f"T3: {result['t3']}")
    lines.append(f"Traps: {result['traps']}")
    lines.append(f"Hygiene: {result['hygiene']}")
    lines.append("Acceptance criteria:")
    for name, ok in result["acceptance"].items():
        lines.append(f"  [{'n/a' if ok is None else ('x' if ok else ' ')}] {name}")
    return "\n".join(lines)
