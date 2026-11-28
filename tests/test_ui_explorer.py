# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Results explorer: the read-only data layer, and a smoke test of every page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from jellybean.runtime.context import RunContext
from jellybean.settings import reset_settings
from jellybean.ui import data
from jellybean.ui.data import ExplorerError, Run


@pytest.fixture
def run(phase5: dict) -> Run:
    ctx: RunContext = phase5["ctx"]
    return Run(ctx.settings.workspace_root, ctx.run_id, data_root=ctx.settings.data_root)


def _snapshot(folder: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file()}


def test_runs_are_listed_with_their_outcome(run: Run, tmp_path: Path) -> None:
    runs = data.list_runs(run.workspace_root)
    row = runs[runs["run"] == run.run_id].iloc[0]
    assert row["primary"] == "traffic" and row["entity"] == "msisdn"
    assert row["findings"] == len(run.findings()) > 0 and row["confirmed"] >= 1
    assert data.list_runs(tmp_path / "nothing").empty  # no workspace yet: an empty list, not an error


def test_findings_and_their_members(run: Run) -> None:
    findings = run.findings()
    assert {"id", "type", "size", "verdict", "stability"} <= set(findings.columns)
    entity = findings[(findings["type"] != "temporal") & (findings["verdict"] == "confirmed")].iloc[0]
    members = run.members(entity["id"])
    assert len(members) == entity["size"] and "msisdn" in members.columns and "n_events" in members.columns
    features = [t["feature"] for t in run.finding(entity["id"])["top_features"] if t["kind"] == "numeric"]
    compare = run.compare_with_population(entity["id"], features[:3])
    assert list(compare["feature"]) == features[:3] and compare["members"].notna().all()
    events, table = run.entity_events("msisdn", members.iloc[0]["msisdn"])
    assert table == run.spec["enriched_artifact"] and len(events) > 0
    assert (events["msisdn"].astype(str) == str(members.iloc[0]["msisdn"])).all()
    temporal = findings[findings["type"] == "temporal"]
    if len(temporal):
        assert {"start", "end"} <= set(run.members(temporal.iloc[0]["id"]).columns)
    with pytest.raises(ExplorerError):
        run.finding("F999")


def test_tables_are_paged_filtered_and_sorted_in_the_database(run: Run) -> None:
    tables = run.tables()
    assert run.spec["features"]["table"] in tables and "traffic" in tables  # artifacts and datasets
    features = run.spec["features"]["table"]
    page, total = run.page(features, order_by="n_events", descending=True, limit=10)
    assert len(page) == 10 and total > 10 and page["n_events"].is_monotonic_decreasing
    second, _ = run.page(features, order_by="n_events", descending=True, limit=10, offset=10)
    assert second["n_events"].max() <= page["n_events"].min()
    busy, matching = run.page(features, where="n_events > 100")
    assert matching < total and (busy["n_events"] > 100).all()
    for bad in ("1=1; DROP TABLE x", "n_events > (SELECT count(*) FROM read_csv('/etc/passwd'))"):
        with pytest.raises(ExplorerError):
            run.page(features, where=bad)
    with pytest.raises(ExplorerError):
        run.page(features, order_by="no_such_column")


def test_sql_is_read_only_and_capped(run: Run) -> None:
    features = run.spec["features"]["table"]
    counted = run.sql(f'SELECT count(*) AS n FROM "{features}"')
    assert counted.iloc[0]["n"] == run.page(features)[1]
    assert len(run.sql("SELECT * FROM traffic", limit=50)) == 50
    for bad in (f'DELETE FROM "{features}"', "COPY (SELECT 1) TO '/tmp/x.csv'", "SELECT * FROM nowhere"):
        with pytest.raises(ExplorerError):
            run.sql(bad)


def test_masking_is_a_switch(run: Run) -> None:
    masker = data.make_masker(None, run.run_id)
    raw, _ = run.page("traffic", limit=20)
    masked, _ = run.page("traffic", limit=20, masker=masker)
    assert not raw["msisdn"].astype(str).str.startswith("masked:").any()
    assert masked["msisdn"].astype(str).str.startswith("masked:").all()
    assert masked["duration_sec"].equals(raw["duration_sec"])  # measures are left as they are


def test_steps_and_agent_summary_come_from_the_trace(run: Run) -> None:
    steps = run.steps()
    assert len(steps) > 20 and {"tool", "status", "duration_s", "error"} <= set(steps.columns)
    assert "run_outlier_detector" in set(steps["tool"])
    summary = run.agent_summary()
    assert summary["tool_calls"].sum() == len(steps)
    empty = Run(run.workspace_root, "no_such_run")
    assert empty.steps().empty and empty.agent_summary().empty and empty.findings().empty


def test_the_explorer_never_writes_to_a_run(run: Run) -> None:
    before = _snapshot(run.path)
    run.findings(), run.artifacts(), run.steps(), run.report()
    run.page(run.spec["features"]["table"], limit=5)
    run.sql("SELECT count(*) FROM traffic")
    run.members(run.findings().iloc[0]["id"])
    assert _snapshot(run.path) == before


@pytest.mark.parametrize("view", ["Runs", "Overview", "Findings", "Tables", "Steps"])
def test_every_page_renders(run: Run, view: str, monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    import jellybean.ui.app  # noqa: F401  (only to find the file)

    monkeypatch.setenv("JELLYBEAN_WORKSPACE_ROOT", str(run.workspace_root))
    monkeypatch.setenv("JELLYBEAN_DATA_ROOT", str(run.data_root))
    reset_settings()
    app = testing.AppTest.from_file(str(Path(data.__file__).with_name("app.py")), default_timeout=120)
    app.session_state["run"] = run.run_id
    app.session_state["view"] = view
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert app.header[0].value in (view, f"Run {run.run_id}")


@pytest.mark.parametrize("view", ["Findings", "Tables"])
def test_selecting_a_row_shows_its_detail(run: Run, view: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("JELLYBEAN_WORKSPACE_ROOT", str(run.workspace_root))
    monkeypatch.setenv("JELLYBEAN_DATA_ROOT", str(run.data_root))
    reset_settings()
    # The same app with the first row of every table picked (row selection is a browser event).
    script = tmp_path / "picked.py"
    script.write_text("import jellybean.ui.app as app\napp._selected_row = lambda event: 0\napp.main()\n")
    app = testing.AppTest.from_file(str(script), default_timeout=120)
    app.session_state["run"] = run.run_id
    app.session_state["view"] = view
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    if view == "Findings":
        assert any(h.value.startswith("F001") for h in app.subheader)
        assert any("Members" in m.label for m in app.metric)


def test_report_images_are_found(run: Run) -> None:
    text = run.report() or ""
    images = [line for line in text.splitlines() if line.startswith("![")]
    for line in images:
        assert (run.path / line.split("](")[1].rstrip(") ")).exists()
    assert isinstance(run.artifacts(), pd.DataFrame) and (run.artifacts()["type"] == "plot").any()
