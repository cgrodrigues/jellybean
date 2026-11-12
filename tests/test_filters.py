"""Row filters (`jellybean run --filter`): parsing, validation and application to every dataset view."""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from jellybean.cli import app
from jellybean.filters import FilterError, parse_filters, validate_filters
from jellybean.runtime.context import RunContext, use_run
from jellybean.runtime.llm_client import FakeLLM, call
from jellybean.settings import Settings, reset_settings
from jellybean.skills._shared import data_tools as dt
from jellybean.spec import AnalysisSpec, apply_patch

DATASETS = {"events", "customers", "wide"}


def test_parse_filters() -> None:
    parsed = parse_filters(
        [
            "events.date >= '2026-10-01' and events.date < '2026-10-02'",
            "events.amount > 10",
            "customers.segment = 'Business'",
        ],
        DATASETS,
    )
    assert parsed == {
        "events": '("date" >= \'2026-10-01\' and "date" < \'2026-10-02\') AND ("amount" > 10)',
        "customers": "(\"segment\" = 'Business')",
    }
    # Prefix-like text inside string literals is left untouched.
    assert parse_filters(["customers.contact = 'events.x@example.com'"], DATASETS) == {
        "customers": "(\"contact\" = 'events.x@example.com')"
    }


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("amount > 10", "Prefix the columns"),
        ("events.amount > 1 and customers.segment = 'x'", "one dataset only"),
        ("events.amount > 1; DROP TABLE x", "no ';'"),
    ],
)
def test_parse_errors(expression: str, message: str) -> None:
    with pytest.raises(FilterError, match=message):
        parse_filters([expression], DATASETS)


def test_validate_filters(settings: Settings, small_data) -> None:
    assert validate_filters(settings, {"events": "\"date\" = DATE '2026-10-01'"}) == {"events": (500, 1000)}
    with pytest.raises(FilterError, match="keeps no rows"):
        validate_filters(settings, {"events": '"amount" < 0'})
    with pytest.raises(FilterError, match="not valid"):
        validate_filters(settings, {"events": '"nope" > 1'})
    with pytest.raises(FilterError, match="rejected"):
        validate_filters(settings, {"events": "\"amount\" IN (SELECT 1 FROM read_csv('/etc/passwd'))"})


def test_filters_apply_to_every_view_of_the_run(settings: Settings, small_data) -> None:
    ctx = RunContext.create(settings, seed=1, run_id="filtered", filters={"events": "\"date\" = DATE '2026-10-01'"})
    with use_run(ctx):
        listed = {d["name"]: d["rows"] for d in dt.list_datasets().summary["datasets"]}
        assert listed["events"] == 500 and listed["customers"] == 50  # other datasets untouched
        assert dt.run_sql("SELECT count(DISTINCT date) AS d FROM events").summary["rows"][0][0] == 1
        assert dt.sample_rows("events", n=20).summary["rows"]
        from jellybean.skills.profiler import tools as pt

        assert pt.profile_table("events").summary["rows"] == 500
        # The internal file-provenance relation used by profiler/preparer/critic is filtered too.
        from jellybean.skills._shared.catalog import connect, dataset_relation_with_files, discover_datasets

        con = connect()
        info = discover_datasets(settings.data_root)["events"]
        assert con.execute(f"SELECT count(*) FROM {dataset_relation_with_files(info)}").fetchall()[0][0] == 500
        con.close()


def test_filters_are_protected_in_the_spec() -> None:
    spec = AnalysisSpec(run_id="r", instruction="i", seed=1, filters={"events": "x > 1"})
    with pytest.raises(ValueError, match="cannot be patched"):
        apply_patch(spec, {"filters": {}})
    assert spec.compact_summary()["filters"] == {"events": "x > 1"}


def test_cli_run_with_filter(settings: Settings, small_data, monkeypatch: pytest.MonkeyPatch) -> None:
    for var, value in (
        ("JELLYBEAN_DATA_ROOT", settings.data_root),
        ("JELLYBEAN_WORKSPACE_ROOT", settings.workspace_root),
        ("JELLYBEAN_GROUND_TRUTH_ROOT", settings.ground_truth_root),
    ):
        monkeypatch.setenv(var, str(value))
    reset_settings()
    llm = FakeLLM(
        {
            "orquestrador": [
                call("init_run", instruction="analyse events"),
                call("list_datasets"),
                call("finish", status="done", summary="ok"),
            ]
        }
    )
    monkeypatch.setattr("jellybean.runtime.llm_client.build_llm_client", lambda s: llm)
    monkeypatch.setattr("jellybean.runtime.llm_client.build_chat_model", lambda s: None)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "analyse events",
            "--auto-approve",
            "--run-id",
            "f1",
            "--quiet",
            "--filter",
            "events.date >= '2026-10-02'",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Filter events: (\"date\" >= '2026-10-02') -> 500 of 1,000 rows" in result.output
    spec = yaml.safe_load((settings.workspace_root / "f1" / "spec.yaml").read_text())
    assert spec["filters"] == {"events": "(\"date\" >= '2026-10-02')"}
    import json

    content = llm.calls[-1].messages[-1].content
    listed = json.loads(content)["summary"]["datasets"]
    assert next(d for d in listed if d["name"] == "events")["rows"] == 500  # the agent sees only filtered rows
    bad = runner.invoke(app, ["run", "x", "--filter", "amount > 1"])
    assert bad.exit_code == 2 and "Filter error" in bad.output
    reset_settings()
