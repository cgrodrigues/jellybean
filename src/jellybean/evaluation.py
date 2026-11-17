# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Evaluation of a run against the dummy-data ground truth (spec section 10).

``scripts/evaluate_run.py <run_id>`` (or ``jellybean evaluate <run_id>``)
compares a run with ``./ground_truth`` and reports: join and spec correctness,
recall per planted anomaly in confirmed findings, precision of confirmed
findings and precision in the top 100 by score, trap verdicts, planted quality
problems detected by the profiler, and run hygiene. For every anomaly it also
shows **where it was lost** (not flagged, flagged but not a finding, a finding
with another verdict, or confirmed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from jellybean.reporting import trace_stats
from jellybean.runtime.context import RunContext
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.findings import Finding, all_findings, finding_entities
from jellybean.spec import load_spec

# Planted quality problems -> (issue kind(s), column) as reported by the profiler.
_QUALITY_MAP: dict[str, tuple[tuple[str, ...], str | None]] = {
    "null_cell_id": (("nulls",), "cell_id"),
    "null_imei": (("nulls",), "imei"),
    "negative_duration": (("negative_values",), "duration_sec"),
    "timezone_shift": (("possible_timezone_shift", "timestamp_outside_partition"), "start_time"),
    "duration_in_ms": (("scale_shift_by_file",), "duration_sec"),
    "duplicate_rows": (("duplicate_rows",), None),
    "segment_spelling": (("inconsistent_categories",), "segment"),
    "duplicate_keys": (("duplicate_keys",), "customer_id"),
}


def _members(ctx: RunContext, f: Finding) -> set[str]:
    ids = finding_entities(f, ctx)
    return set(ids[ids.columns[0]].astype(str))


def _temporal_hits(f: Finding, details: dict[str, Any]) -> bool:
    """A temporal finding matches anomaly E when a window covers its cell and time."""
    e = details.get("E")
    if not e or f.context.get("group_by") != "cell_id":
        return False
    start = pd.Timestamp(f"{e['date']} {e['start_hour']:02d}:00")
    end = start + pd.Timedelta(hours=e["hours"])
    return any(
        str(w["group"]) == str(e["cell_id"]) and pd.Timestamp(w["start"]) < end and pd.Timestamp(w["end"]) > start
        for w in f.context.get("windows", [])
    )


def evaluate(ctx: RunContext, truth: Path) -> dict[str, Any]:
    """Compute every evaluation metric for one run (the top-up scenario has its own metrics)."""
    spec = load_spec(ctx.spec_path)
    if spec.primary() and spec.primary().name == "topup":
        from jellybean.evaluation_topup import evaluate_topup

        return evaluate_topup(ctx, truth)
    store = ArtifactStore(ctx)
    expected = yaml.safe_load((truth / "expected_spec.yaml").read_text(encoding="utf-8"))
    gt = pd.read_parquet(truth / "anomalies.parquet")
    details = json.loads((truth / "anomaly_details.json").read_text(encoding="utf-8"))
    traps = json.loads((truth / "traps.json").read_text(encoding="utf-8"))
    planted_issues = json.loads((truth / "quality_issues.json").read_text(encoding="utf-8"))
    datasets = {d.name for d in spec.datasets}
    primary_name = spec.primary().name if spec.primary() else None
    entity = f"{spec.entity.dataset or primary_name}.{spec.entity.column}" if spec.entity else None
    out: dict[str, Any] = {"run_id": ctx.run_id, "phase": spec.phase, "entity": entity, "filters": spec.filters}
    if spec.filters:
        # The ground truth covers all the data: recall is measured on the whole of it.
        out["note"] = "The run used row filters; anomalies outside the filtered rows cannot be found."

    # ---- join and spec
    exp_join = expected["joins"][0]
    confirmed_joins = [j for j in spec.joins if j.status == "confirmed"]
    join_ok = any(
        [list(k) for k in j.keys] == exp_join["keys"] and j.key_normalization == exp_join["key_normalization"]
        for j in confirmed_joins
    )
    rejected_ok = not any("region_code" in [b for _, b in j.keys] for j in confirmed_joins)
    join_applicable = exp_join["left"] in datasets and exp_join["right"] in datasets
    out["join"] = {
        "keys_and_normalization_correct": join_ok if join_applicable else None,
        "false_candidate_rejected": rejected_ok if join_applicable else None,
        "applicable": join_applicable,
    }
    primary_name = spec.primary().name if spec.primary() else None
    if primary_name == "orders":
        orders = expected.get("orders", {})
        exp_grain, exp_entity, exp_time = ["order_id"], orders.get("entity"), orders.get("time_column")
    else:
        exp_grain = expected["grain"]["columns"]
        exp_entity, exp_time = expected["entity"]["column"], expected["time_column"]
    out["spec"] = {
        "grain": bool(spec.grain) and spec.grain.columns == exp_grain,
        "entity": bool(spec.entity) and spec.entity.column == exp_entity,
        "time_column": spec.time_column == exp_time,
    }

    # ---- findings
    findings = all_findings(ctx)
    members = {f.id: _members(ctx, f) for f in findings if f.type != "temporal"}
    confirmed = [f for f in findings if f.critic_verdict == "confirmed"]
    confirmed_entities = set().union(*(members[f.id] for f in confirmed if f.id in members)) if confirmed else set()
    primary = spec.primary().name if spec.primary() else None

    # Detection runs: which entities were flagged by anything.
    runs = [r for r in store.list("table") if r.params.get("kind") in ("outlier", "contextual", "cluster", "graph")]
    flagged: set[str] = set()
    for record in runs:
        table = pd.read_parquet(store.path_of(record.id))
        entity = record.params.get("entity")
        if entity in table.columns and "flagged" in table.columns:
            flagged |= set(table.loc[table["flagged"], entity].astype(str))

    recall, trail = {}, {}
    for anomaly, block in gt.groupby("anomaly_id"):
        # The top-up truth also lists every movement involved (entity_column = movement_id).
        block = block[block["entity_column"] != "movement_id"]
        dataset = block["dataset"].iloc[0]
        if dataset != primary:
            recall[anomaly] = None
            trail[anomaly] = f"not applicable: lives in '{dataset}', not the primary dataset of this run"
            continue
        ids = set(block["entity_id"].astype(str))
        anomaly_entity = block["entity_column"].iloc[0]
        run_entity = spec.entity.column if spec.entity else None
        if anomaly != "E" and run_entity and anomaly_entity != run_entity:
            # One analysis per entity: this anomaly is about another entity.
            recall[anomaly] = None
            trail[anomaly] = f"not applicable: anomaly entity is '{anomaly_entity}', this run analyses '{run_entity}'"
            continue
        if anomaly == "E":
            hit = [f for f in findings if f.type == "temporal" and _temporal_hits(f, details)]
            recall[anomaly] = 1.0 if any(f.critic_verdict == "confirmed" for f in hit) else 0.0
            trail[anomaly] = (
                "confirmed"
                if recall[anomaly]
                else f"temporal finding(s) {[f'{f.id}:{f.critic_verdict}' for f in hit]}"
                if hit
                else "no temporal window on the planted cell"
            )
            continue
        recall[anomaly] = round(len(ids & confirmed_entities) / len(ids), 4)
        in_findings = {
            f.id: (len(ids & members[f.id]), f.critic_verdict)
            for f in findings
            if f.id in members and ids & members[f.id]
        }
        flagged_share = round(len(ids & flagged) / len(ids), 4)
        if recall[anomaly] >= 0.6:
            trail[anomaly] = f"confirmed (recall {recall[anomaly]})"
        elif in_findings:
            trail[anomaly] = f"in findings {in_findings} (flagged share {flagged_share})"
        elif flagged_share:
            trail[anomaly] = f"flagged by detectors (share {flagged_share}) but not consolidated into a finding"
        else:
            trail[anomaly] = "not flagged by any detector"
    out["recall"] = recall
    out["where_lost"] = trail

    planted = set(gt[gt["dataset"] == primary]["entity_id"].astype(str))
    out["precision"] = {
        "confirmed_entities": len(confirmed_entities),
        "entity_precision": round(len(confirmed_entities & planted) / len(confirmed_entities), 4)
        if confirmed_entities
        else None,
        "finding_precision": round(
            sum(
                (len(members[f.id] & planted) / len(members[f.id]) >= 0.5)
                if f.id in members
                else _temporal_hits(f, details)
                for f in confirmed
            )
            / len(confirmed),
            4,
        )
        if confirmed
        else None,
    }
    outlier_runs = [r for r in runs if r.params["kind"] == "outlier"]
    if outlier_runs:
        ranks = None
        for record in outlier_runs:
            table = pd.read_parquet(store.path_of(record.id)).set_index(record.params["entity"])
            r = table["score"].rank(pct=True)
            ranks = r if ranks is None else ranks.add(r, fill_value=0)
        top = set(ranks.sort_values(ascending=False).index[:100].astype(str))
        out["precision"]["top100_by_score"] = round(len(top & planted) / 100, 4)

    # ---- traps
    trap_results = {}
    for trap in traps:
        if trap["dataset"] not in datasets:
            trap_results[trap["id"]] = {"expected": trap["expected_verdict"], "applicable": False, "pass": None}
            continue
        if trap["id"] == "units_error":
            date = trap["file"].split("date=")[1].split("/")[0]
            hits = [
                f
                for f in findings
                if f.type == "temporal"
                and f.context.get("measure") == trap["column"]
                and any(str(w["start"])[:10] <= date <= str(w["end"])[:10] for w in f.context.get("windows", []))
            ]
            verdicts = [f.critic_verdict for f in hits]
            trap_results[trap["id"]] = {
                "expected": trap["expected_verdict"],
                "verdicts": verdicts,
                "pass": bool(verdicts) and all(v == trap["expected_verdict"] for v in verdicts),
            }
        elif trap["id"] == "imputation_artifact":
            trap_ids = set(map(str, trap["entity_ids"]))
            hits = [
                f
                for f in findings
                if f.id in members and members[f.id] and len(members[f.id] & trap_ids) / len(members[f.id]) >= 0.5
            ]
            verdicts = [f.critic_verdict for f in hits]
            trap_results[trap["id"]] = {
                "expected": trap["expected_verdict"],
                "verdicts": verdicts,
                # Not surfacing a false cluster at all is also a pass: nothing was imputed.
                "pass": all(v != "confirmed" for v in verdicts),
                "surfaced": bool(hits),
            }
    out["traps"] = trap_results

    # ---- quality issues detected by the profiler
    issues_id = spec.evidence_artifacts.get("quality_issues")
    found = store.load_json(issues_id)["issues"] if issues_id else []
    quality = {}
    for planted_issue in planted_issues:
        if planted_issue["dataset"] not in datasets:
            continue
        kinds, column = _QUALITY_MAP.get(planted_issue["id"], ((), None))
        quality[planted_issue["id"]] = any(
            i["kind"] in kinds and (column is None or i.get("column") == column) for i in found
        )
    out["quality_detected"] = quality

    # ---- hygiene
    out["hygiene"] = trace_stats(ctx)

    # ---- acceptance criteria (section 10)
    applicable = {a: r for a, r in recall.items() if r is not None}
    out["acceptance"] = {
        # None = not applicable to this run (e.g. a run over orders only).
        "join_discovered_with_normalisation_and_region_code_rejected": (join_ok and rejected_ok)
        if join_applicable
        else None,
        "duplicate_keys_flagged_before_join": quality.get("duplicate_keys") if join_applicable else None,
        "unmatched_rows_flagged": bool(spec.enriched_artifact) if join_applicable else None,
        "four_of_six_anomalies_recall_0_6": sum(r >= 0.6 for r in applicable.values()) >= min(4, len(applicable)),
        "unit_error_is_artifact": trap_results.get("units_error", {}).get("pass"),
        "tool_outputs_within_limits": out["hygiene"].get("max_tool_output_chars", 0) <= 4000
        and out["hygiene"].get("max_raw_rows", 0) <= 20,
    }
    return out


def format_evaluation(result: dict[str, Any]) -> str:
    """Human-readable evaluation table."""
    if result.get("scenario") == "topup":
        from jellybean.evaluation_topup import format_topup

        return format_topup(result)
    lines = [f"Evaluation of run {result['run_id']} (phase {result['phase']}, entity {result.get('entity')})", ""]
    if result.get("filters"):
        lines += [f"Filters: {result['filters']} - {result.get('note', '')}", ""]
    lines.append(f"Join: {result['join']}")
    lines.append(f"Spec: {result['spec']}")
    lines.append("Recall per anomaly (confirmed findings):")
    for anomaly, value in result["recall"].items():
        lines.append(f"  {anomaly}: {value if value is not None else 'n/a'}  <- {result['where_lost'][anomaly]}")
    lines.append(f"Precision: {result['precision']}")
    lines.append(f"Traps: {result['traps']}")
    lines.append(f"Quality problems detected: {result['quality_detected']}")
    lines.append(f"Hygiene: {result['hygiene']}")
    lines.append("Acceptance criteria:")
    for name, ok in result["acceptance"].items():
        mark = "n/a" if ok is None else ("x" if ok else " ")
        lines.append(f"  [{mark}] {name}")
    return "\n".join(lines)


def combine(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine the evaluations of several runs (one analysis per entity).

    Each anomaly keeps its best recall over the runs where it is applicable;
    the acceptance criteria are recomputed on those best values.
    """
    best: dict[str, dict[str, Any]] = {}
    for result in results:
        for anomaly, value in result["recall"].items():
            if value is None:
                best.setdefault(anomaly, {"recall": None, "run": None, "where": result["where_lost"][anomaly]})
                continue
            if best.get(anomaly, {}).get("recall") is None or value > best[anomaly]["recall"]:
                best[anomaly] = {
                    "recall": value,
                    "run": result["run_id"],
                    "entity": result.get("entity"),
                    "where": result["where_lost"][anomaly],
                }
    if results[0].get("scenario") == "topup":
        from jellybean.evaluation_topup import combine_topup

        extra = combine_topup(results, best)
        return {
            "runs": [r["run_id"] for r in results],
            "entities": {r["run_id"]: r.get("entity") for r in results},
            "best_recall": best,
            **extra,
        }
    applicable = {a: b["recall"] for a, b in best.items() if b["recall"] is not None}
    acceptance: dict[str, Any] = {}
    for name in results[0]["acceptance"]:
        values = [r["acceptance"][name] for r in results if r["acceptance"].get(name) is not None]
        acceptance[name] = all(values) if values else None
    acceptance["four_of_six_anomalies_recall_0_6"] = (
        sum(v >= 0.6 for v in applicable.values()) >= min(4, len(applicable)) if applicable else None
    )
    return {
        "runs": [r["run_id"] for r in results],
        "entities": {r["run_id"]: r.get("entity") for r in results},
        "best_recall": best,
        "acceptance": acceptance,
    }


def format_combined(combined: dict[str, Any]) -> str:
    lines = [f"Combined evaluation of {len(combined['runs'])} runs", ""]
    for run, entity in combined["entities"].items():
        lines.append(f"  {run}: entity {entity}")
    lines.append("Best recall per anomaly:")
    for anomaly, b in combined["best_recall"].items():
        value = "n/a" if b["recall"] is None else f"{b['recall']}"
        lines.append(
            f"  {anomaly}: {value}" + (f"  (run {b['run']}, {b['entity']})" if b.get("run") else f"  <- {b['where']}")
        )
    lines.append("Acceptance criteria (all runs):")
    for name, ok in combined["acceptance"].items():
        lines.append(f"  [{'n/a' if ok is None else ('x' if ok else ' ')}] {name}")
    return "\n".join(lines)
