"""Detection and critique on the top-up ledger, one analysis per entity (requirements_2.md, phase 5).

On top of discovery and cleaning (as in test_topup_features), each entity is
analysed the way the SKILL.md files prescribe: features with the typed-event
models and the conformity rules a reader of the README would write, detection
with the feature families and the per-entity activity series, then the critic's
checks, explain_by_reference included, and its verdict policy. Checks:

* promotions: the misconfigured promotion (T3) is confirmed, the legitimate
  campaign day (TR1) is explained, and no other promotion is confirmed;
* voucher batches: the two batches redeemed before distribution (T4) are confirmed;
* dealers: the four suspicious dealers (T5) are confirmed;
* operators: the three insiders (T1) are confirmed;
* customers: T2, T6 and T8 are confirmed with recall >= 0.6.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from jellybean.datagen import GeneratorConfig, generate
from jellybean.evaluation import combine, evaluate, format_combined
from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext
from jellybean.runtime.human import ScriptedHuman
from jellybean.runtime.llm_client import FakeLLM
from jellybean.settings import AppConfig, EnvSettings, Settings
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.findings import all_findings, finding_entities
from jellybean.skills._shared.spec_tools import patch_spec
from jellybean.skills.critico import tools as cr
from jellybean.skills.detector import tools as de
from jellybean.skills.feature_engineer import tools as fe
from jellybean.skills.orquestrador import tools as ot
from jellybean.skills.preparer import tools as pr
from tests.conftest import verdict_policy
from tests.test_topup_discovery import _discover

pytestmark = pytest.mark.slow  # about 13 minutes: run with `uv run pytest -m slow`

# Rules as an agent would write them from the README notes of the reference tables.
RULES = {
    "bonus_rule": (
        "amount",
        "CASE WHEN ref_face_value >= min_recharge AND CAST(ref_event_time AS DATE) BETWEEN start_date AND end_date "
        "THEN least(round(ref_face_value * bonus_pct / 100, 2), max_bonus) ELSE 0 END",
        "lower(trim(movement_type)) = 'bonus_rch'",
    ),
    "promo_dates": (
        "CASE WHEN CAST(event_time AS DATE) BETWEEN start_date AND end_date THEN 0 ELSE 1 END",
        "0",
        "promo_id IS NOT NULL",
    ),
    "before_distribution": (
        "CASE WHEN CAST(event_time AS DATE) < distribution_date THEN 1 ELSE 0 END",
        "0",
        "voucher_batch_id IS NOT NULL",
    ),
}
FAMILIES = (
    ["timestamp"],
    ["counterparty"],
    ["linked"],
    ["sequence"],
    ["linked", "sequence"],
    ["conformity"],
    ["interaction_graph"],
    ["category_measure", "category_ratio"],
)
ENTITIES = ("promo_id", "voucher_batch_id", "dealer_id", "agent_user", "msisdn")


def _analyse(ctx: RunContext, enriched: str, entity: str) -> list[dict[str, Any]]:
    """Features, detection and critique for one entity; returns the judged findings."""
    patch_spec({"entity": {"column": entity, "alternatives": [], "source": "auto", "dataset": "topup"}})
    spec = ot.read_spec()
    type_column = pd.Series(list(spec.structural_nulls.values())).mode()[0]
    table = fe.build_entity_features(enriched, entity, "event_time").artifact_id
    table = fe.build_category_features(enriched, entity, type_column, "amount", table).artifact_id
    table = fe.build_linked_event_features(enriched, entity, type_column, "event_time", table).artifact_id
    table = fe.build_sequence_features(enriched, entity, type_column, "event_time", table).artifact_id
    counterpart = next((r.column for r in spec.references if r.kind == "counterpart"), None)
    if entity == "msisdn" and counterpart:
        table = fe.build_interaction_graph(enriched, entity, counterpart, table).artifact_id
    for name, (observed, expected, where) in RULES.items():
        result = fe.build_conformity_features(enriched, entity, name, observed, expected, table, where=where)
        if result.status == "ok":  # a rule needs its reference table in the run
            table = result.artifact_id
    scaled = fe.scale_encode(table).artifact_id
    embedding = fe.reduce_dimensions(scaled, "pca", 10).artifact_id
    runs = [de.run_outlier_detector(scaled, m).artifact_id for m in ("isolation_forest", "ecod", "lof")]
    for family in FAMILIES:
        for method in ("ecod", "isolation_forest"):
            result = de.run_outlier_detector(scaled, method, families=family)
            if result.status == "ok" and result.artifact_id:  # a family absent from the table is skipped
                runs.append(result.artifact_id)
    runs.append(de.run_clustering(embedding, "hdbscan").artifact_id)
    raw = pd.read_parquet(ArtifactStore(ctx).path_of(table))
    attrs = [
        c
        for c in raw.columns
        if c.startswith("attr_") and not pd.api.types.is_numeric_dtype(raw[c]) and 2 <= raw[c].nunique() <= 10
    ]
    for attr in attrs[:2]:
        result = de.run_contextual_outliers(table, attr)
        if result.status == "ok":
            runs.append(result.artifact_id)
    if any(j.keys[0][0] == entity for j in spec.joins):  # the entity is the key of a reference table
        runs.append(de.run_timeseries_anomaly(enriched, "event_time", "count", entity, "day").artifact_id)
    assert de.consolidate_findings(runs).status == "ok"
    for f in all_findings():
        for check in (cr.stability_check, cr.ensemble_agreement, cr.artifact_check, cr.explain_by_reference):
            check(f.id)
    judged = []
    for f in all_findings():
        verdict = verdict_policy(f)
        assert cr.set_verdict(f.id, verdict, {"policy": "SKILL.md"}).status == "ok"
        if verdict == "explained":  # the guard: an explained finding cannot be confirmed
            assert cr.set_verdict(f.id, "confirmed", {}).status == "error"
        if verdict == "confirmed" and f.type != "temporal":
            assert cr.set_verdict(f.id, "explained", {}).status == "error"
            cr.set_verdict(f.id, "confirmed", {"policy": "SKILL.md"})
        f = next(x for x in all_findings() if x.id == f.id)
        members = (
            {str(w["group"]) for w in f.context.get("windows", [])}
            if f.type == "temporal"
            else set(finding_entities(f)[f.entity_column].astype(str))
        )
        judged.append({"finding": f, "members": members, "explanation": f.checks.get("explanation", {})})
    return judged


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("topup_detection")
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
    ctx = RunContext.create(settings, seed=42, run_id="detection")
    _discover(ctx)
    out: dict[str, Any] = {}
    runtime = Runtime(ctx=ctx, llm=FakeLLM({}), human=ScriptedHuman([ot.APPROVE_ALL]), options={"instruction": "x"})
    with runtime.active():
        ot.ask_human(**ot.prepare_cleaning_approval().summary["question"])
        ot.approve_cleaning_plan()
        pr.apply_cleaning_plan()
        left = None
        for i in range(len(ot.read_spec().joins)):
            left = pr.materialize_join(i, left_artifact=left).artifact_id
        evaluations = []
        for entity in ENTITIES:
            out[entity] = _analyse(ctx, left, entity)
            evaluations.append(evaluate(ctx, base / "truth") | {"run_id": f"detection:{entity}"})
    # A run without promotions (T8 must need them), customers only.
    bare = RunContext.create(settings, seed=42, run_id="no_promotions")
    _discover(bare, ("dealers", "voucher_batches", "cust_segmentation"))
    runtime = Runtime(ctx=bare, llm=FakeLLM({}), human=ScriptedHuman([ot.APPROVE_ALL]), options={"instruction": "x"})
    with runtime.active():
        ot.ask_human(**ot.prepare_cleaning_approval().summary["question"])
        ot.approve_cleaning_plan()
        pr.apply_cleaning_plan()
        left = None
        for i in range(len(ot.read_spec().joins)):
            left = pr.materialize_join(i, left_artifact=left).artifact_id
        out["bare"] = _analyse(bare, left, "msisdn")
        evaluations.append(evaluate(bare, base / "truth"))
    out["evaluations"] = evaluations
    out["combined"] = combine(evaluations)
    gt = pd.read_parquet(base / "truth" / "anomalies.parquet")
    out["gt"] = gt[(gt["dataset"] == "topup") & (gt["entity_column"] != "movement_id")]
    out["traps"] = {t["id"]: t for t in json.loads((base / "truth" / "traps.json").read_text())}
    out["details"] = json.loads((base / "truth" / "anomaly_details.json").read_text())["topup"]
    return out


def _planted(run: dict[str, Any], anomaly: str) -> set[str]:
    return set(run["gt"].loc[run["gt"]["anomaly_id"] == anomaly, "entity_id"])


def _confirmed(judged: list[dict[str, Any]]) -> set[str]:
    return set().union(*(j["members"] for j in judged if j["finding"].critic_verdict == "confirmed"))


# --------------------------------------------------------------------- promotions
def test_misconfigured_promotion_is_confirmed_and_the_campaign_explained(run: dict[str, Any]) -> None:
    judged = run["promo_id"]
    temporal = [j for j in judged if j["finding"].type == "temporal"]
    t3 = [j for j in temporal if "P017" in j["members"]]
    assert t3 and all(j["finding"].critic_verdict == "confirmed" for j in t3)
    day = run["traps"]["TR1_legitimate_campaign"]["date"]
    tr1 = [
        j
        for j in temporal
        if "P009" in j["members"] and any(w["start"].startswith(day) for w in j["finding"].context["windows"])
    ]
    assert tr1 and all(j["finding"].critic_verdict == "explained" for j in tr1), [j["explanation"] for j in tr1]
    assert "follow every rule of 'promotions'" in tr1[0]["explanation"]["summary"]
    # T3 is one finding at the promotion level; no other promotion is confirmed.
    assert _confirmed(judged) == {"P017"}


# ---------------------------------------------------------------- batches, dealers
def test_vouchers_redeemed_before_distribution_are_confirmed(run: dict[str, Any]) -> None:
    assert _planted(run, "T4") <= _confirmed(run["voucher_batch_id"])


def test_suspicious_dealers_are_confirmed(run: dict[str, Any]) -> None:
    assert _planted(run, "T5") <= _confirmed(run["dealer_id"])


# ---------------------------------------------------------------------- customers
def test_insider_operators_are_confirmed(run: dict[str, Any]) -> None:
    assert _planted(run, "T1") <= _confirmed(run["agent_user"])


@pytest.mark.parametrize("anomaly", ["T2", "T6", "T8"])
def test_customer_anomalies_are_confirmed(run: dict[str, Any], anomaly: str) -> None:
    planted = _planted(run, anomaly)
    recall = len(planted & _confirmed(run["msisdn"])) / len(planted)
    trail = [
        (j["finding"].id, j["finding"].size, j["finding"].critic_verdict, len(planted & j["members"]))
        for j in run["msisdn"]
        if planted & j["members"]
    ]
    assert recall >= 0.6, (anomaly, recall, trail)


def test_explanations_never_cover_planted_anomalies(run: dict[str, Any]) -> None:
    planted = set(run["gt"]["entity_id"])
    for entity in ENTITIES:
        for j in run[entity]:
            if j["finding"].critic_verdict == "explained":
                assert len(j["members"] & planted) <= 0.1 * len(j["members"]), (entity, j["explanation"])


# --------------------------------------------------------------------- evaluation
def test_topup_acceptance_criteria(run: dict[str, Any]) -> None:
    """Section 11 of requirements_2.md, from `jellybean evaluate` over the per-entity runs."""
    combined = run["combined"]
    acceptance = combined["acceptance"]
    print(format_combined(combined))
    for criterion in (
        "joins_and_self_reference_discovered",
        "balance_invariant_discovered",
        "r1_to_r5_recall_0_9",
        "sign_error_systematic_and_apart_from_r1",
        "no_structural_null_imputed",
        "t3_at_promotion_level",
        "t3_not_reported_as_customers",
        "tr1_tr3_explained",
        "tr2_artifact_or_fixed",
        "t8_needs_promotions",
    ):
        assert acceptance[criterion] is True, (criterion, combined)
    assert acceptance["five_of_eight_anomalies_recall_0_6"] is True, combined["best_recall"]
