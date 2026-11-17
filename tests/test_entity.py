# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""The analysis entity chosen by the user (`jellybean run --entity <dataset>.<column>`)."""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from jellybean.cli import app
from jellybean.entity import EntityError, parse_entity, validate_entity
from jellybean.evaluation import combine
from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext
from jellybean.runtime.llm_client import FakeLLM, call
from jellybean.settings import Settings, reset_settings
from jellybean.skills.orquestrador import tools as ot
from jellybean.skills.profiler import tools as pt
from jellybean.spec import AnalysisSpec, EntitySpec, apply_patch, load_spec


# ------------------------------------------------------------------ parse / validate
def test_parse_and_validate(settings: Settings, small_data) -> None:
    assert parse_entity("events.phone", settings) == ("events", "phone")
    stats = validate_entity(settings, "events", "phone")
    assert stats["entities"] == 50 and stats["median_events"] == 20 and stats["rows"] == 1000
    # Filters are applied before validating (as in the run).
    assert validate_entity(settings, "events", "phone", "\"date\" = DATE '2026-10-01'")["rows"] == 500


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("phone", "Use the form <dataset>.<column>"),
        ("nope.phone", "Unknown dataset"),
        ("events.nope", "Unknown column"),
        ("events.event_id", "Row-level analysis is not supported"),
    ],
)
def test_invalid_entities(settings: Settings, small_data, expression: str, message: str) -> None:
    with pytest.raises(EntityError, match=message):
        validate_entity(settings, *parse_entity(expression, settings))


# --------------------------------------------------------------- spec protection
def test_user_entity_is_protected() -> None:
    spec = AnalysisSpec(
        run_id="r", instruction="i", seed=1, entity=EntitySpec(column="phone", source="user", dataset="events")
    )
    with pytest.raises(ValueError, match="set by the user"):
        apply_patch(spec, {"entity": {"column": "amount"}})
    assert apply_patch(spec, {"entity": {"alternatives": ["x"]}}).entity.column == "phone"


# ------------------------------------------------- init_run, propose_spec, control 1
def _run_profiler_chain(ctx: RunContext, entity: dict | None, primary: str = "events") -> Runtime:
    options = {"instruction": "analyse events enriched with customers", "entity": entity}
    runtime = Runtime(ctx=ctx, llm=FakeLLM({}), options=options)
    with runtime.active():
        ot.init_run(instruction="analyse events enriched with customers")
        other = "customers" if primary == "events" else "events"
        ot.update_spec(
            patch={
                "datasets": [
                    {"name": primary, "path": "p", "role": "primary"},
                    {"name": other, "path": "p", "role": "enrichment"},
                ]
            }
        )
        types = [pt.infer_semantic_types(d).artifact_id for d in ("events", "customers")]
        grain, ents = pt.infer_grain(primary).artifact_id, pt.suggest_entities(primary).artifact_id
        cand = pt.discover_join_candidates("events", "customers")
        best = cand.summary["best"]
        join = pt.test_join(
            "events",
            "customers",
            [[best["left_column"], best["right_column"]]],
            [best["normalization"]],
            candidates_artifact=cand.artifact_id,
        ).artifact_id
        pt.propose_spec([join], grain, ents, types)
    return runtime


def test_user_entity_flows_through_the_run(run_ctx: RunContext) -> None:
    user = {"dataset": "events", "column": "amount", "stats": {"entities": 97}}  # not the profiler's top choice
    runtime = _run_profiler_chain(run_ctx, user)
    spec = load_spec(run_ctx.spec_path)
    assert (spec.entity.column, spec.entity.source, spec.entity.dataset) == ("amount", "user", "events")
    assert spec.decisions[0].topic == "entity" and spec.decisions[0].by == "human"
    with runtime.active():
        prep = ot.prepare_spec_confirmation().summary
        entity_item = next(i for i in prep["items"] if i["item"] == "entity")
        assert entity_item["auto"] is True and entity_item["evidence"]["source"].startswith("set by the user")
        assert ot.update_spec(patch={"entity": {"column": "phone"}}).status == "error"
        ot.confirm_spec()
    decision = next(d for d in load_spec(run_ctx.spec_path).decisions if d.topic == "control_1:entity")
    assert decision.by == "human"


def test_user_entity_must_be_on_the_primary(run_ctx: RunContext) -> None:
    user = {"dataset": "customers", "column": "segment", "stats": {}}
    runtime = _run_profiler_chain(run_ctx, user)
    with runtime.active():
        prep = ot.prepare_spec_confirmation().summary
    assert prep["ready"] is False and "make 'customers' the primary dataset" in " ".join(prep["problems"])


def test_without_entity_the_profiler_chooses(run_ctx: RunContext) -> None:
    _run_profiler_chain(run_ctx, None)
    spec = load_spec(run_ctx.spec_path)
    assert spec.entity.column == "phone" and spec.entity.source == "auto"


# ----------------------------------------------------------------------------- CLI
def test_cli_run_with_entity(settings: Settings, small_data, monkeypatch: pytest.MonkeyPatch) -> None:
    for var, value in (
        ("JELLYBEAN_DATA_ROOT", settings.data_root),
        ("JELLYBEAN_WORKSPACE_ROOT", settings.workspace_root),
        ("JELLYBEAN_GROUND_TRUTH_ROOT", settings.ground_truth_root),
    ):
        monkeypatch.setenv(var, str(value))
    reset_settings()
    llm = FakeLLM({"orquestrador": [call("init_run", instruction="x"), call("finish", status="done", summary="ok")]})
    monkeypatch.setattr("jellybean.runtime.llm_client.build_llm_client", lambda s: llm)
    monkeypatch.setattr("jellybean.runtime.llm_client.build_chat_model", lambda s: None)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "x", "--auto-approve", "--quiet", "--run-id", "e1", "--entity", "events.phone"])
    assert result.exit_code == 0, result.output
    assert "Entity events.phone: 50 entities, median 20 events each" in result.output
    entity = yaml.safe_load((settings.workspace_root / "e1" / "spec.yaml").read_text())["entity"]
    assert entity["column"] == "phone" and entity["source"] == "user"
    bad = runner.invoke(app, ["run", "x", "--entity", "events.event_id"])
    assert bad.exit_code == 2 and "Row-level analysis is not supported" in bad.output
    reset_settings()


# ---------------------------------------------------------------------- evaluation
def test_combined_evaluation_keeps_the_best_run_per_anomaly() -> None:
    acceptance = {"unit_error_is_artifact": True, "four_of_six_anomalies_recall_0_6": False}
    customer = {
        "run_id": "r1",
        "entity": "topup.msisdn",
        "recall": {"T2": 0.9, "T1": None},
        "where_lost": {"T2": "confirmed", "T1": "not applicable: anomaly entity is 'agent_user'"},
        "acceptance": acceptance,
    }
    agent = {
        "run_id": "r2",
        "entity": "topup.agent_user",
        "recall": {"T2": None, "T1": 0.7},
        "where_lost": {"T2": "not applicable", "T1": "confirmed"},
        "acceptance": acceptance,
    }
    combined = combine([customer, agent])
    assert combined["best_recall"]["T2"]["run"] == "r1" and combined["best_recall"]["T1"]["run"] == "r2"
    assert combined["acceptance"]["unit_error_is_artifact"] is True
