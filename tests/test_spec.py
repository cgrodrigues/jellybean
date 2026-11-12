"""Tests for the spec.yaml schema, patching and join-confidence rules."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jellybean.settings import JoinConfidenceConfig
from jellybean.spec import AnalysisSpec, Decision, apply_patch, assess_join, load_spec, record_decision, save_spec


def _spec() -> AnalysisSpec:
    return AnalysisSpec(
        run_id="2026-11-10_001",
        instruction="analyse traffic and enrich with cust_segmentation",
        seed=42,
        datasets=[
            {"name": "traffic", "path": "./data/traffic", "role": "primary"},
            {"name": "cust_segmentation", "path": "./data/cust_segmentation", "role": "enrichment"},
        ],
    )


def test_defaults_and_roundtrip(tmp_path: Path) -> None:
    spec = _spec()
    assert spec.phase == "discovery" and spec.report_language == "en"
    assert spec.cleaning_plan.status == "pending"
    path = tmp_path / "spec.yaml"
    save_spec(path, spec)
    assert load_spec(path) == spec


def test_patch_merges_and_validates() -> None:
    patched = apply_patch(
        _spec(),
        {
            "joins": [
                {
                    "left": "traffic",
                    "right": "cust_segmentation",
                    "keys": [["msisdn", "customer_id"]],
                    "key_normalization": ["strip_prefix_plus351"],
                    "cardinality": "N:1",
                    "coverage": 0.97,
                }
            ],
            "grain": {"dataset": "traffic", "columns": ["call_id"]},
            "entity": {"column": "msisdn", "alternatives": ["imei"]},
            "time_column": "start_time",
            "columns": {"duration_sec": {"dataset": "traffic", "semantic_type": "measure"}},
            "cleaning_plan": {"status": "proposed"},
        },
    )
    assert patched.joins[0].keys == [("msisdn", "customer_id")]
    assert patched.cleaning_plan.artifact_id is None  # nested merge keeps other fields
    assert patched.compact_summary()["columns"] == {"measure": 1}  # counts per semantic type


@pytest.mark.parametrize(
    "patch",
    [
        {"joins": [{"left": "traffic", "right": "unknown", "keys": [["a", "b"]]}]},
        {"columns": {"x": {"dataset": "traffic", "semantic_type": "colour"}}},
        {"grain": {"dataset": "orders", "columns": ["id"]}},
        {"datasets": [{"name": "a", "path": "p", "role": "primary"}, {"name": "b", "path": "p", "role": "primary"}]},
        {
            "joins": [
                {
                    "left": "traffic",
                    "right": "cust_segmentation",
                    "keys": [["a", "b"]],
                    "key_normalization": ["none", "none"],
                }
            ]
        },
        {"unknown_field": 1},
        {"phase": "dreaming"},
    ],
)
def test_invalid_patches_are_rejected(patch: dict) -> None:
    with pytest.raises((ValidationError, ValueError)):
        apply_patch(_spec(), patch)


def test_protected_fields_and_decisions() -> None:
    with pytest.raises(ValueError, match="cannot be patched"):
        apply_patch(_spec(), {"seed": 1})
    spec = record_decision(_spec(), Decision(topic="join", choice="confirmed", by="auto", evidence={"coverage": 0.97}))
    assert len(spec.decisions) == 1 and spec.decisions[0].at


THRESHOLDS = JoinConfidenceConfig()


def test_assess_join_auto_confirm() -> None:
    result = assess_join(0.97, 0.9992, 1.0009, 0.64, 1, "N:1", None, THRESHOLDS)
    assert result.auto_confirm and result.confidence > 0.9


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"coverage": 0.90}, None),
        ({"right_uniqueness": 0.99}, None),
        ({"fanout": 1.2}, None),
        ({"score_margin": 0.05}, None),
        ({"score_margin": None}, None),
        ({"n_keys": 2}, "composite key"),
        ({"cardinality": "N:M"}, "N:M relation"),
        ({"temporal_validity": "valid_from/valid_to"}, "dimension with temporal validity"),
    ],
)
def test_assess_join_requires_confirmation(kwargs: dict, reason: str | None) -> None:
    args = {
        "coverage": 0.97,
        "right_uniqueness": 0.9992,
        "fanout": 1.0009,
        "score_margin": 0.64,
        "n_keys": 1,
        "cardinality": "N:1",
        "temporal_validity": None,
    } | kwargs
    result = assess_join(**args, thresholds=THRESHOLDS)
    assert not result.auto_confirm
    if reason:
        assert reason in result.requires_confirmation_reasons


def test_context_notes_reach_the_summary() -> None:
    spec = _spec().model_copy(update={"context_notes": "duration_sec: call duration in seconds. " * 60})
    summary = spec.compact_summary()
    assert summary["context_notes"].startswith("duration_sec: call duration in seconds.")
    assert len(summary["context_notes"]) <= 1201 and summary["context_notes"].endswith("…")


def test_summary_of_a_five_dataset_spec_fits_the_output_limit() -> None:
    """A real top-up spec (5 datasets, 4 joins, references, invariants) is shown whole by get_spec."""
    import json
    from pathlib import Path

    from jellybean.runtime.tool_registry import ToolResult, enforce_limits

    names = ["topup", "promotions", "dealers", "voucher_batches", "cust_segmentation"]
    spec = AnalysisSpec(
        run_id="r",
        instruction="analisa os dados em topup e enriquece-os com promotions, dealers, voucher_batches e "
        "cust_segmentation",
        seed=1,
        datasets=[
            {
                "name": n,
                "path": str(Path("/very/long/path/to/the/data/root") / n),
                "role": "primary" if i == 0 else "enrichment",
            }
            for i, n in enumerate(names)
        ],
        joins=[
            {
                "left": "topup",
                "right": n,
                "keys": [[f"{n}_id", f"{n}_id"]],
                "key_normalization": ["strip_prefix_plus351"],
                "status": "confirmed",
                "coverage": 0.97,
                "fanout": 1.0,
                "right_uniqueness": 0.999,
                "confidence": 0.96,
                "evidence_artifact": f"join_test_topup_{n}_v1",
            }
            for n in names[1:]
        ],
        context_notes="x" * 3000,
        columns={f"c{i}": {"dataset": "topup", "semantic_type": "measure"} for i in range(80)},
        structural_nulls={f"topup.col_{i}": "movement_type" for i in range(10)},
        cleaning_plan={
            "artifact_id": "cleaning_plan_v1",
            "status": "approved",
            "approved_ops": ["P01", "P02", "P04", "P05"],
        },
    )
    result = enforce_limits(ToolResult.ok({"updated": ["joins"], "spec": spec.compact_summary()}), 4000, 20)
    assert not result.truncated, len(json.dumps(spec.compact_summary()))
    shown = result.summary["spec"]
    assert len(shown["datasets"]) == 5 and len(shown["joins"]) == 4
    assert shown["cleaning_plan"]["approved_ops"] == ["P01", "P02", "P04", "P05"]


def test_update_spec_cannot_rewrite_a_tested_join(run_ctx) -> None:
    from jellybean.skills._shared import spec_tools as st
    from jellybean.spec import save_spec

    save_spec(run_ctx.spec_path, AnalysisSpec(run_id=run_ctx.run_id, instruction="x", seed=1))
    datasets = [
        {"name": "events", "path": "p", "role": "primary"},
        {"name": "customers", "path": "p", "role": "enrichment"},
    ]
    join = {
        "left": "events",
        "right": "customers",
        "keys": [["phone", "customer_id"]],
        "key_normalization": ["strip_prefix_plus351"],
        "evidence_artifact": "join_test_events_customers_v1",
    }
    st.patch_spec({"datasets": datasets, "joins": [join]})
    lost = st.update_spec(patch={"joins": [{**join, "key_normalization": []}]})
    assert lost.status == "error" and "propose_spec" in lost.summary["error"]
    dropped = st.update_spec(patch={"joins": []})
    assert dropped.status == "error" and "cannot be removed" in dropped.summary["error"]
    assert st.update_spec(patch={"joins": [{**join, "status": "proposed"}]}).status == "ok"
