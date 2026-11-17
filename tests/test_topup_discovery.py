# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Discovery on the top-up ledger (requirements_2.md, phase 2 acceptance criteria).

Runs the profiler's deterministic tool chain (no LLM) on generated top-up data,
taking each decision the way ``profiler/SKILL.md`` prescribes and only from tool
results, then checks the discovery criteria of section 11 against the ground
truth: joins (sparse keys included) and the self-reference, the balance
invariant without README, R1-R5 found with recall >= 0.9, the sign error (TR2)
reported as a systematic block apart from R1, and no structural-null column
imputed by the cleaning plan. The same checks run with ``--opaque-codes``.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd
import pytest
import yaml

from jellybean.datagen import GeneratorConfig, generate
from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext, use_run
from jellybean.runtime.human import AutoApproveHuman
from jellybean.runtime.llm_client import FakeLLM
from jellybean.settings import AppConfig, EnvSettings, Settings
from jellybean.skills._shared import data_tools as dt
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.catalog import connect
from jellybean.skills.orquestrador import tools as ot
from jellybean.skills.preparer import tools as pr
from jellybean.skills.profiler import tools as pt

ENRICHMENTS = ["promotions", "dealers", "voucher_batches", "cust_segmentation"]


def _discover(ctx: RunContext, enrichments: tuple[str, ...] = tuple(ENRICHMENTS)) -> dict[str, Any]:
    """The profiler procedure of SKILL.md, decisions taken from tool results only."""
    enrichments = list(enrichments)
    out: dict[str, Any] = {}
    runtime = Runtime(ctx=ctx, llm=FakeLLM({}), human=AutoApproveHuman(), options={"instruction": "analyse topup"})
    with runtime.active():
        ot.init_run(instruction="analyse topup enriched with promotions, dealers, voucher_batches, cust_segmentation")
        ot.update_spec(
            patch={
                "datasets": [{"name": "topup", "path": "p", "role": "primary"}]
                + [{"name": e, "path": "p", "role": "enrichment"} for e in enrichments]
            }
        )
        names = ["topup", *enrichments]
        prof = {d: pt.profile_table(d) for d in names}
        types = {d: pt.infer_semantic_types(d) for d in names}
        grain, entities = pt.infer_grain("topup"), pt.suggest_entities("topup")
        partitions = set(dt.get_schema("topup").summary["partition_columns"])
        time_col = next(c for c in types["topup"].summary["time_column_candidates"] if c not in partitions)
        temporal = pt.check_temporal_coverage("topup", time_col)
        # Integrity checks.
        nulls = pt.classify_nulls("topup")
        invariants = pt.discover_arithmetic_invariants("topup")
        duplicates = pt.find_duplicate_keys("topup")
        refs = pt.discover_self_references("topup")
        checks = [invariants, nulls]
        for cand in refs.summary["candidates"]:
            if cand["next"] == "check_referential_integrity":
                checks.append(pt.check_referential_integrity("topup", cand["column"], "topup", cand["target"]))
            else:
                checks.append(pt.check_reciprocity("topup", cand["target"], cand["column"], time_column=time_col))
        signed = [
            c["name"]
            for c in prof["topup"].summary["columns"]
            if "p50" in c and c["min"] is not None and float(c["min"]) < 0 < float(c["max"])
        ]
        type_column = Counter(
            v["determined_by"] for v in nulls.summary["columns"].values() if "determined_by" in v
        ).most_common(1)[0][0]
        signs = [pt.infer_sign_convention("topup", m, type_column) for m in signed]
        # Joins.
        joins = []
        for enrichment in enrichments:
            cand = pt.discover_join_candidates("topup", enrichment)
            best = cand.summary["best"]
            joins.append(
                pt.test_join(
                    "topup",
                    enrichment,
                    [[best["left_column"], best["right_column"]]],
                    [best["normalization"]],
                    candidates_artifact=cand.artifact_id,
                )
            )
        # Reference tables pointing to each other (same column name).
        columns = {e: {c["name"] for c in prof[e].summary["columns"]} for e in enrichments}
        for a in enrichments:
            for b in enrichments:
                if a == b:
                    continue
                for col in sorted(columns[a] & columns[b]):
                    stats = next(c for c in prof[b].summary["columns"] if c["name"] == col)
                    rows = prof[b].summary["rows"]
                    if types[a].summary["columns"].get(col, [None])[0] == "identifier" and stats["distinct"] == rows:
                        checks.append(pt.check_referential_integrity(a, col, b, col))
        integrity = [*checks, duplicates, *signs]
        issues = pt.compile_quality_issues(
            [prof[d].artifact_id for d in names]
            + [grain.artifact_id, temporal.artifact_id]
            + [r.artifact_id for r in integrity]
            + [j.artifact_id for j in joins]
        )
        proposal = pt.propose_spec(
            [j.artifact_id for j in joins],
            grain.artifact_id,
            entities.artifact_id,
            [types[d].artifact_id for d in names],
            temporal=temporal.artifact_id,
            quality_issues=issues.artifact_id,
            checks=[c.artifact_id for c in checks],
        )
        assert proposal.status == "ok", proposal.summary
        out["control_1"] = ot.prepare_spec_confirmation()
        if not out["control_1"].summary.get("auto_confirmable"):
            ot.ask_human(**out["control_1"].summary["question"])
        ot.confirm_spec()
        plan = pr.propose_cleaning_plan(issues.artifact_id)
        out["plan"] = ArtifactStore(ctx).load_json(plan.artifact_id)
        out["issues"] = ArtifactStore(ctx).load_json(issues.artifact_id)["issues"]
        out["spec"] = ot.read_spec()
        out["joins"] = joins
        out["results"] = {"invariants": invariants, "nulls": nulls, "refs": refs, "signs": signs}
    return out


@pytest.fixture(scope="module", params=["readable", "opaque"])
def run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp(f"topup_{request.param}")
    opaque = request.param == "opaque"
    # Opaque codes on a smaller population: the reconciliation sizes scale with it.
    generate(
        GeneratorConfig(
            out=base / "data",
            truth=base / "truth",
            customers=6000 if opaque else 20_000,
            datasets=("cust_segmentation", "topup"),
            opaque_codes=opaque,
        )
    )
    settings = Settings(
        env=EnvSettings(
            data_root=base / "data",
            workspace_root=base / "ws",
            ground_truth_root=base / "truth",
            config_file=base / "none.yaml",
        ),
        config=AppConfig(),
    )
    ctx = RunContext.create(settings, seed=42, run_id=f"discovery_{request.param}")
    out = _discover(ctx)
    out |= {
        "ctx": ctx,
        "truth": base / "truth",
        "expected": yaml.safe_load((base / "truth" / "expected_spec_topup.yaml").read_text()),
        "recon": pd.read_parquet(base / "truth" / "reconciliation_issues.parquet"),
        "traps": {t["id"]: t for t in json.loads((base / "truth" / "traps.json").read_text())},
        "default_size": not opaque,
    }
    return out


def _rows(run: dict[str, Any], predicate: str) -> set[str]:
    with use_run(run["ctx"]):
        con = connect()
        try:
            return {r[0] for r in con.execute(f"SELECT DISTINCT movement_id FROM topup x WHERE {predicate}").fetchall()}
        finally:
            con.close()


# ---------------------------------------------------------------------- joins
def test_enrichment_joins_are_discovered_and_auto_confirmed(run: dict[str, Any]) -> None:
    expected = {(j["right"], tuple(j["keys"][0]), j["key_normalization"][0]) for j in run["expected"]["joins"]}
    for join in run["joins"]:
        s = join.summary
        assert (s["right"], tuple(s["keys"][0]), s["normalization"][0]) in expected, s["keys"]
        # cust_segmentation has 15 duplicated keys whatever the population: below 20,000
        # customers its key uniqueness falls under the threshold and a human confirms it.
        if s["right"] != "cust_segmentation" or run["default_size"]:
            assert s["assessment"]["auto_confirm"], (s["right"], s["assessment"])
    sparse = {j.summary["right"]: j.summary["metrics"] for j in run["joins"]}
    for right in ("promotions", "dealers", "voucher_batches"):
        assert sparse[right]["coverage"] == 1.0 and sparse[right]["null_key_share"] > 0.7
        assert sparse[right]["nulls_structural"] is True
    assert run["control_1"].summary["auto_confirmable"] is run["default_size"]


def test_references_are_discovered(run: dict[str, Any]) -> None:
    refs = {(r.dataset, r.column, r.ref_dataset, r.ref_column, r.kind) for r in run["spec"].references}
    assert ("topup", "ref_movement_id", "topup", "movement_id", "reference") in refs
    assert ("topup", "counterparty_msisdn", "topup", "msisdn", "counterpart") in refs
    assert ("voucher_batches", "dealer_id", "dealers", "dealer_id", "reference") in refs


# ------------------------------------------------------------------ invariant
def test_balance_invariant_is_discovered_without_readme(run: dict[str, Any]) -> None:
    expressions = {i.expression for i in run["spec"].invariants}
    assert "amount + balance_before = balance_after" in expressions
    assert "amount = face_value" in expressions


# ------------------------------------------------------------- reconciliation
@pytest.mark.parametrize("issue_id", ["R1", "R2", "R3", "R4", "R5"])
def test_reconciliation_problems_are_reported(run: dict[str, Any], issue_id: str) -> None:
    truth = set(run["recon"].loc[run["recon"]["issue_id"] == issue_id, "movement_id"])
    best = None
    for issue in run["issues"]:
        predicate = issue["detail"].get("predicate")
        if not predicate:
            continue
        found = _rows(run, predicate)
        recall, precision = len(found & truth) / len(truth), len(found & truth) / len(found) if found else 0.0
        if best is None or recall + precision > best[0] + best[1]:
            best = (recall, precision, issue["kind"], issue["detail"].get("invariant"))
    # Found by one issue of its own (not lumped with another problem).
    assert best and best[0] >= 0.9 and best[1] >= 0.9, (issue_id, best)


def test_sign_error_is_a_systematic_block_apart_from_r1(run: dict[str, Any]) -> None:
    trap = run["traps"]["TR2_sign_error"]
    systematic = [i for i in run["issues"] if i["kind"] == "invariant_violation_systematic"]
    assert len(systematic) == 1
    detail = systematic[0]["detail"]
    assert detail["dates"] == trap["dates"] and detail["sign_flipped"]["column"] == "amount"
    # An R1 row can fall inside the affected segment and be counted with it.
    assert abs(detail["rows"] - trap["rows"]) <= 0.01 * trap["rows"] and len(detail["segment"]["values"]) == 1
    inversions = [i for i in run["issues"] if i["kind"] == "sign_inversion"]
    assert inversions and all(i["detail"]["dates"] == trap["dates"] for i in inversions)


# --------------------------------------------------------------------- nulls
def test_structural_nulls_are_never_imputed(run: dict[str, Any]) -> None:
    structural = {k.split(".", 1)[1] for k in run["spec"].structural_nulls}
    expected = set(run["expected"]["structural_nulls"])
    assert expected <= structural
    imputed = {o["operation"].get("column") for o in run["plan"]["ops"] if o["operation"]["op"] == "impute"}
    assert not imputed & structural
    # Only the unexpected nulls (reason missing in adjustments) remain a null issue.
    null_issues = {i["column"] for i in run["issues"] if i["kind"] == "nulls" and i["dataset"] == "topup"}
    assert null_issues == {"reason_code"}


# ------------------------------------------------------------------- hygiene
def test_tool_outputs_stay_within_limits(run: dict[str, Any]) -> None:
    events = [json.loads(line) for line in run["ctx"].trace_path.read_text().splitlines()]
    tools = [e for e in events if e["kind"] == "tool_call"]
    assert max(e["output_chars"] for e in tools) <= 4000 and max(e["raw_rows"] for e in tools) <= 20
    assert not [e for e in tools if e["status"] == "error"], [e.get("tool") for e in tools if e["status"] == "error"]


# ----------------------------------------------------------------- evaluation
def test_evaluation_reports_the_discovery_criteria(run: dict[str, Any]) -> None:
    """`jellybean evaluate` on a run stopped after discovery and cleaning (no findings yet)."""
    from jellybean.evaluation import evaluate, format_evaluation

    with use_run(run["ctx"]):
        result = evaluate(run["ctx"], run["truth"])
    acceptance = result["acceptance"]
    for criterion in (
        "joins_and_self_reference_discovered",
        "balance_invariant_discovered",
        "r1_to_r5_recall_0_9",
        "sign_error_systematic_and_apart_from_r1",
        "no_structural_null_imputed",
    ):
        assert acceptance[criterion] is True, (criterion, result["reconciliation"], result["discovery"])
    assert result["opaque_codes"] is (not run["default_size"])
    assert result["recall"]["T1"] is None and "not applicable" in result["where_lost"]["T1"]
    assert "Reconciliation problems" in format_evaluation(result)


def test_reciprocity_orients_swapped_columns(run: dict[str, Any]) -> None:
    from jellybean.skills.profiler import tools as pt

    with use_run(run["ctx"]):
        right = pt.check_reciprocity("topup", "msisdn", "counterparty_msisdn", time_column="event_time").summary
        swapped = pt.check_reciprocity("topup", "counterparty_msisdn", "msisdn", time_column="event_time").summary
    assert swapped["without_reciprocal"] == right["without_reciprocal"] and "swapped" in swapped["note"]
