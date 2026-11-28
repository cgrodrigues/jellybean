# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Feature models for typed events and the flip_sign fix.

Runs discovery (as in test_topup_discovery), approves the whole cleaning plan
(the inferred flip_sign included), materialises the joins and builds the
features the way ``feature_engineer/SKILL.md`` prescribes for the customer
entity. Then checks that the sign error is fixed, that the expected responses
are learnt from the data, and that each customer anomaly (T2, T6, T7, T8) is
separated from normal customers by at least one of the new features.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from jellybean.datagen import GeneratorConfig, generate
from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext, use_run
from jellybean.runtime.human import ScriptedHuman
from jellybean.runtime.llm_client import FakeLLM
from jellybean.settings import AppConfig, EnvSettings, Settings
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.catalog import connect
from jellybean.skills.feature_engineer import tools as fe
from jellybean.skills.orquestrador import tools as ot
from jellybean.skills.preparer import tools as pr
from tests.test_topup_discovery import _discover

NEW_PREFIXES = ("cat_", "ratio_", "link_", "seq_", "conf_", "igraph_")
BONUS_RULE = (
    "CASE WHEN ref_face_value >= min_recharge AND CAST(ref_event_time AS DATE) BETWEEN start_date AND end_date "
    "THEN least(round(ref_face_value * bonus_pct / 100, 2), max_bonus) ELSE 0 END"
)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("topup_features")
    generate(GeneratorConfig(out=base / "data", truth=base / "truth", datasets=("cust_segmentation", "topup")))
    settings = Settings(
        env=EnvSettings(
            data_root=base / "data",
            workspace_root=base / "ws",
            ground_truth_root=base / "truth",
            config_file=base / "x",
        ),
        config=AppConfig(),
    )
    ctx = RunContext.create(settings, seed=42, run_id="features")
    out: dict[str, Any] = {"ctx": ctx, "discovery": _discover(ctx)}
    runtime = Runtime(ctx=ctx, llm=FakeLLM({}), human=ScriptedHuman([ot.APPROVE_ALL]), options={"instruction": "x"})
    with runtime.active():
        ot.ask_human(**ot.prepare_cleaning_approval().summary["question"])
        ot.approve_cleaning_plan()
        out["applied"] = pr.apply_cleaning_plan()
        spec = ot.read_spec()
        left = None
        for i in range(len(spec.joins)):
            left = pr.materialize_join(i, left_artifact=left).artifact_id
        enriched, entity, time = left, "msisdn", "event_time"
        # feature_engineer/SKILL.md, step 2b (typed events).
        spec = ot.read_spec()
        type_column = pd.Series(list(spec.structural_nulls.values())).mode()[0]
        measure = spec.invariants[0].expression.split(" ")[0]
        table = fe.build_entity_features(enriched, entity, time).artifact_id
        out["category"] = fe.build_category_features(enriched, entity, type_column, measure, table)
        out["linked"] = fe.build_linked_event_features(enriched, entity, type_column, time, out["category"].artifact_id)
        out["sequence"] = fe.build_sequence_features(enriched, entity, type_column, time, out["linked"].artifact_id)
        counterpart = next(r.column for r in spec.references if r.kind == "counterpart")
        out["graph"] = fe.build_interaction_graph(enriched, entity, counterpart, out["sequence"].artifact_id)
        out["conformity"] = fe.build_conformity_features(
            enriched,
            entity,
            "bonus_rule",
            "amount",
            BONUS_RULE,
            out["graph"].artifact_id,
            where=f"lower(trim({type_column})) = 'bonus_rch'",
        )
        out["rejected"] = [
            fe.build_conformity_features(enriched, entity, "bad", "amount", expr, out["conformity"].artifact_id)
            for expr in ("(SELECT max(amount) FROM topup)", "1;\nDROP TABLE x", f"read_parquet('{base}/truth/x')")
        ]
        out["scaled"] = fe.scale_encode(ot.read_spec().features.table)
        out["table"] = pd.read_parquet(ArtifactStore(ctx).path_of(out["conformity"].artifact_id))
        out["dictionary"] = ArtifactStore(ctx).load_json(ot.read_spec().features.dictionary)["features"]
        out["clean_topup"] = ot.read_spec().prepared["topup"]
    gt = pd.read_parquet(base / "truth" / "anomalies.parquet")
    out["gt"] = gt[(gt["dataset"] == "topup") & (gt["entity_column"] == "msisdn")]
    out["traps"] = {t["id"]: t for t in json.loads((base / "truth" / "traps.json").read_text())}
    return out


# ------------------------------------------------------------------ flip_sign
def test_flip_sign_is_proposed_for_explicit_approval(run: dict[str, Any]) -> None:
    flips = [o for o in run["discovery"]["plan"]["ops"] if o["operation"]["op"] == "flip_sign"]
    assert len(flips) == 1 and flips[0]["requires_explicit_approval"] is True
    op = flips[0]["operation"]
    assert op["column"] == "amount" and op["values"] == ["ussd"] and op["only_if"]
    covered = [n for n in run["discovery"]["plan"]["not_addressed"] if "flip_sign" in n["reason"]]
    assert covered  # the sign_inversion issues are the same rows


def test_flip_sign_restores_the_invariant(run: dict[str, Any]) -> None:
    trap = run["traps"]["TR2_sign_error"]
    ops = {o["op"]: o for r in run["applied"].summary["results"] for o in r["operations"]}
    assert ops["flip_sign"]["affected_rows"] == trap["rows"]
    with use_run(run["ctx"]):
        con = connect()
        try:
            left = con.execute(
                f'SELECT count(*) FROM (SELECT DISTINCT * FROM "{run["clean_topup"]}") '
                "WHERE balance_after <> balance_before + amount"
            ).fetchall()[0][0]
        finally:
            con.close()
    assert left == 300  # only R1, the scattered reconciliation problem


# ------------------------------------------------------------- feature models
def test_expected_responses_are_learnt(run: dict[str, Any]) -> None:
    rules = {
        (r["kind"], r.get("sibling"), r["type"], r["expected"]) for r in run["linked"].summary["expected_responses"]
    }
    assert ("answer", None, "adv_grant", "adv_recovery") in rules
    assert ("after_sibling", "reversal", "bonus_rch", "bonus_clawback") in rules
    pairs = {tuple(p) for p in run["category"].summary["ratio_pairs"]}
    assert ("reversal", "rch_app") in pairs and ("bonus_clawback", "bonus_rch") in pairs


@pytest.mark.parametrize("anomaly", ["T2", "T6", "T7", "T8"])
def test_customer_anomalies_are_separated_by_a_new_feature(run: dict[str, Any], anomaly: str) -> None:
    df, gt = run["table"], run["gt"]
    planted = set(gt.loc[gt["anomaly_id"] == anomaly, "entity_id"])
    members = df["msisdn"].isin(planted)
    normal = df[~df["msisdn"].isin(set(gt["entity_id"]))]
    best = 0.0, ""
    for column in (c for c in df.columns if c.startswith(NEW_PREFIXES)):
        reference = normal[column].astype(float).dropna()
        if reference.empty:
            continue
        values = df.loc[members, column].astype(float)
        separated = max((values > reference.quantile(0.99)).mean(), (values < reference.quantile(0.01)).mean())
        best = max(best, (separated, column))
    assert best[0] >= 0.9, (anomaly, best)


def test_conformity_rule_and_its_formula_are_recorded(run: dict[str, Any]) -> None:
    s = run["conformity"].summary
    assert s["rows_checked"] > 90_000 and 140 <= s["rows_off"] <= 1_700  # T8 (+ T3's 100% bonuses)
    formulas = {d["name"]: d["formula"] for d in run["dictionary"]}
    assert "ref_face_value >= min_recharge" in formulas["conf_bonus_rule_violations"]
    assert all(r.status == "error" for r in run["rejected"])


def test_features_are_model_ready(run: dict[str, Any]) -> None:
    assert run["scaled"].status == "ok"
    names = {d["name"] for d in run["dictionary"]}
    assert {c for c in run["table"].columns if c != "msisdn"} <= names
    assert {"seq_inactivity_after_last_s", "igraph_partner_degree_max", "link_referenced_share"} <= names


def test_extending_an_older_version_does_not_fork_the_table(run: dict[str, Any]) -> None:
    """Two extensions asked in one turn pass the same version: the second must keep the first's columns."""
    with use_run(run["ctx"]):
        old = run["category"].artifact_id  # an ancestor of the latest table
        again = fe.build_sequence_features(
            ot.read_spec().enriched_artifact, "msisdn", "movement_type", "event_time", old, max_pairs=2
        )
        table = pd.read_parquet(ArtifactStore(run["ctx"]).path_of(again.artifact_id))
    assert again.status == "ok" and any("not the latest" in w for w in again.warnings)
    assert {"conf_bonus_rule_violations", "igraph_partners", "link_referenced_share"} <= set(table.columns)


def test_a_family_absent_from_the_table_is_skipped(run: dict[str, Any]) -> None:
    from jellybean.skills.detector import tools as de

    with use_run(run["ctx"]):
        result = de.run_outlier_detector(run["scaled"].artifact_id, "ecod", families=["no_such_family"])
    assert result.status == "ok" and result.summary["skipped"] and result.artifact_id is None
