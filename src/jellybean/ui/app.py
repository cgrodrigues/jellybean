# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Streamlit app of the results explorer. Start it with ``jellybean ui``.

Read-only: it shows what the runs wrote to the workspace (see ``ui/data.py``)
and never changes a run, so it can stay open while a run is in progress.
Values are shown as stored; the "Mask personal data" switch hides them the way
the agents' tools do (for sharing the screen).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from jellybean.settings import get_settings
from jellybean.ui import data
from jellybean.ui.data import ExplorerError, Run

LOGO = Path(__file__).resolve().parents[3] / "docs" / "assets" / "jellybean.svg"
VIEWS = ["Runs", "Overview", "Findings", "Tables", "Steps", "Snapshots"]
_IMAGE_LINE = re.compile(r"^!\[(.*?)\]\((.+?)\)\s*$")


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _logo_svg() -> str | None:
    """The logo as SVG markup (Streamlit draws a string that starts with ``<svg``)."""
    try:
        text = LOGO.read_text(encoding="utf-8")
    except OSError:
        return None
    start = text.find("<svg")
    return text[start:] if start >= 0 else None


def _selected_row(event: Any) -> int | None:
    """Index of the row picked in a ``st.dataframe(on_select="rerun")``, if any."""
    rows = getattr(getattr(event, "selection", None), "rows", None) or []
    return rows[0] if rows else None


def _download(frame: pd.DataFrame, name: str, key: str) -> None:
    st.download_button(
        "Download these rows (CSV)", frame.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv", key=key
    )


def _render_report(run: Run, text: str) -> None:
    """Markdown with its images: a line that is only an image is drawn from the run folder."""
    block: list[str] = []

    def flush() -> None:
        if block:
            st.markdown("\n".join(block))
            block.clear()

    for line in text.splitlines():
        match = _IMAGE_LINE.match(line.strip())
        if match:
            flush()
            path = run.path / match.group(2)
            if path.exists():
                st.image(str(path), caption=match.group(1) or None)
            else:
                st.caption(f"(image not found: {match.group(2)})")
        else:
            block.append(line)
    flush()


# --------------------------------------------------------------------------- #
# Views                                                                       #
# --------------------------------------------------------------------------- #
def view_runs(runs: pd.DataFrame) -> None:
    st.header("Runs")
    if runs.empty:
        st.info('No run in the workspace yet. Start one with `jellybean run "..."`.')
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Runs", len(runs))
    c2.metric("Findings", int(runs["findings"].sum()))
    c3.metric("Confirmed findings", int(runs["confirmed"].sum()))
    st.caption("Pick a row to open that run.")
    event = st.dataframe(
        runs,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="runs_table",
        column_config={"instruction": st.column_config.TextColumn(width="large")},
    )
    row = _selected_row(event)
    if row is not None:
        # Widget state cannot be changed once the widgets exist: applied at the next rerun.
        st.session_state["_goto"] = (runs.iloc[row]["run"], "Overview")
        st.rerun()


def view_overview(run: Run) -> None:
    spec, result = run.spec, run.result
    st.header(f"Run {run.run_id}")
    st.write(spec.get("instruction", ""))
    findings = run.findings()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Phase", spec.get("phase") or "-")
    c2.metric("Status", result.get("status") or "not finished")
    c3.metric("Findings", len(findings))
    c4.metric("Confirmed", int((findings["verdict"] == "confirmed").sum()) if len(findings) else 0)

    report_tab, spec_tab, decisions_tab = st.tabs(["Report", "Spec", "Decisions"])
    with report_tab:
        text = run.report()
        if text is None:
            st.info("This run has no report yet (it did not reach the report phase).")
            if result.get("summary"):
                st.write(result["summary"])
        else:
            st.download_button("Download report.md", text.encode("utf-8"), file_name=f"{run.run_id}_report.md")
            _render_report(run, text)
    with spec_tab:
        grain = spec.get("grain") or {}
        entity = spec.get("entity") or {}
        left, right = st.columns(2)
        left.markdown("**Datasets**")
        left.dataframe(pd.DataFrame(spec.get("datasets") or []), hide_index=True, width="stretch")
        right.markdown("**Entity, grain and time**")
        grain_text = (
            ", ".join(grain.get("columns") or [])
            if grain.get("unique", True)
            else "no unique key" + (f" (closest: {', '.join(grain['columns'])})" if grain.get("columns") else "")
        )
        right.dataframe(
            pd.DataFrame(
                [
                    ("entity", entity.get("column")),
                    ("entity alternatives", ", ".join(entity.get("alternatives") or [])),
                    ("grain", grain_text or None),
                    ("time column", spec.get("time_column")),
                ],
                columns=["item", "value"],
            ),
            hide_index=True,
            width="stretch",
        )
        if spec.get("joins"):
            st.markdown("**Joins**")
            joins = pd.DataFrame(spec["joins"])
            joins["keys"] = joins["keys"].map(lambda ks: ", ".join(f"{a} = {b}" for a, b in ks))
            keep = ["left", "right", "keys", "key_normalization", "cardinality", "coverage", "fanout", "status"]
            st.dataframe(joins[[c for c in keep if c in joins.columns]], hide_index=True, width="stretch")
        kinds = pd.Series([c.get("semantic_type") for c in (spec.get("columns") or {}).values()]).value_counts()
        if len(kinds):
            st.markdown("**Columns by semantic type**")
            st.bar_chart(kinds, horizontal=True, height=220)
        with st.expander("spec.yaml (everything)"):
            st.json(spec, expanded=False)
    with decisions_tab:
        decisions = pd.DataFrame(spec.get("decisions") or [])
        if decisions.empty:
            st.info("No decision recorded.")
        else:
            decisions["evidence"] = decisions["evidence"].map(lambda e: json.dumps(e, ensure_ascii=False, default=str))
            keep = ["at", "topic", "choice", "by", "evidence"]
            st.dataframe(decisions[[c for c in keep if c in decisions.columns]], hide_index=True, width="stretch")


def view_findings(run: Run, masker: Any) -> None:
    st.header("Findings")
    findings = run.findings()
    if findings.empty:
        st.info("This run has no findings (yet).")
        return
    c1, c2 = st.columns(2)
    verdicts = c1.multiselect("Verdict", sorted(findings["verdict"].unique()), key="f_verdict")
    types = c2.multiselect("Type", sorted(findings["type"].dropna().unique()), key="f_type")
    shown = findings
    if verdicts:
        shown = shown[shown["verdict"].isin(verdicts)]
    if types:
        shown = shown[shown["type"].isin(types)]
    left, right = st.columns([3, 2])
    with right:
        st.bar_chart(findings["verdict"].value_counts(), horizontal=True, height=200)
    with left:
        st.caption("Pick a row to see the finding.")
        event = st.dataframe(
            shown, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="f_table"
        )
    row = _selected_row(event)
    if row is None or row >= len(shown):
        return
    _finding_detail(run, shown.iloc[row]["id"], masker)


def _finding_detail(run: Run, finding_id: str, masker: Any) -> None:
    f = run.finding(finding_id)
    checks = f.get("checks") or {}
    st.subheader(f"{f['id']} · {f.get('type')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Members", f.get("size"))
    c2.metric("Verdict", f.get("critic_verdict") or "not judged")
    stability = (checks.get("stability") or {}).get("stability")
    c3.metric("Stability", "-" if stability is None else f"{stability:.2f}")
    c4.metric("Novelty", f.get("novelty") or "-")
    if f.get("description"):
        st.write(f["description"])
    st.caption("Methods: " + ", ".join(f.get("methods") or []))

    why, members_tab, checks_tab, plots_tab = st.tabs(["Why it stands out", "Members", "Critic checks", "Plots"])
    with why:
        top = pd.DataFrame(f.get("top_features") or [])
        if top.empty:
            st.info("No feature comparison recorded for this finding.")
        else:
            st.markdown("**Features that differ most from the population** (robust effect size)")
            if "effect" in top.columns:
                st.bar_chart(top.set_index("feature")["effect"], horizontal=True, height=min(60 + 28 * len(top), 420))
            st.dataframe(top, hide_index=True, width="stretch")
        if f.get("rules"):
            st.markdown("**Rules that describe the group**")
            st.dataframe(pd.DataFrame(f["rules"]), hide_index=True, width="stretch")
        if f.get("context"):
            with st.expander("Context (windows, families, attribute)"):
                st.json(f["context"], expanded=False)
    with members_tab:
        _finding_members(run, f, masker)
    with checks_tab:
        if not checks:
            st.info("The critic has not checked this finding.")
        for name, result in checks.items():
            with st.expander(name, expanded=name in ("stability", "artifact")):
                associations = result.get("associations") if isinstance(result, dict) else None
                st.json({k: v for k, v in result.items() if k != "associations"}, expanded=True)
                if associations:
                    st.dataframe(pd.DataFrame(associations), hide_index=True, width="stretch")
        if f.get("critic_evidence"):
            with st.expander("Evidence given with the verdict"):
                st.json(f["critic_evidence"].get("evidence", f["critic_evidence"]), expanded=True)
    with plots_tab:
        plots = [run.path / p for p in f.get("plots") or []]
        plots = [p for p in plots if p.exists()]
        if not plots:
            st.info("No plot for this finding.")
        for path in plots:
            st.image(str(path), caption=path.name)


def _finding_members(run: Run, f: dict[str, Any], masker: Any) -> None:
    try:
        raw = run.members(f["id"])
    except ExplorerError as exc:
        st.warning(str(exc))
        return
    shown = data.mask_frame(raw, masker)
    st.caption(f"{len(raw)} rows" + (" (first 5,000)" if len(raw) >= data.MAX_SQL_ROWS else ""))
    features = [t["feature"] for t in f.get("top_features") or [] if t.get("kind") == "numeric"][:8]
    compare = run.compare_with_population(f["id"], features)
    if not compare.empty:
        st.markdown("**Median of the members against all entities**")
        st.dataframe(compare, hide_index=True, width="stretch")
    st.dataframe(shown, hide_index=True, width="stretch", height=320)
    _download(shown, f"{run.run_id}_{f['id']}_members.csv", key=f"dl_members_{f['id']}")

    entity = f.get("entity_column")
    if f.get("type") == "temporal" or entity not in raw.columns or run.events_table() is None:
        return
    st.markdown("**Events of one member**")
    position = st.selectbox(
        "Member",
        range(min(len(raw), 500)),
        format_func=lambda i: str(shown.iloc[i][entity]),
        key=f"member_{f['id']}",
        index=None,
        placeholder="Choose a member to see its rows in the event table",
    )
    if position is None:
        return
    events, table = run.entity_events(entity, raw.iloc[position][entity], masker)
    st.caption(f"{len(events)} rows from `{table}`" + (" (first 1,000)" if len(events) >= 1000 else ""))
    time_column = run.spec.get("time_column")
    if time_column in events.columns and len(events):
        per_day = pd.to_datetime(events[time_column], errors="coerce").dt.floor("D").value_counts().sort_index()
        st.bar_chart(per_day, height=180)
    st.dataframe(events, hide_index=True, width="stretch", height=320)


def view_tables(run: Run, masker: Any) -> None:
    st.header("Tables")
    tables = run.tables()
    if not tables:
        st.info("This run has no table yet.")
        return
    artifacts = run.artifacts()
    browse, sql_tab, all_tab = st.tabs(["Browse a table", "SQL", "All artifacts"])
    with browse:
        name = st.selectbox("Table", list(tables), key="t_name")
        try:
            schema = run.schema(name)
        except ExplorerError as exc:
            st.error(str(exc))
            return
        params = run.artifact_params(name)
        with st.expander(f"{len(schema)} columns" + (" · how it was made" if params else "")):
            st.dataframe(schema, hide_index=True, width="stretch")
            if params:
                st.json(params, expanded=False)
        c1, c2, c3, c4 = st.columns([4, 2, 1, 1])
        where = c1.text_input("Filter (a SQL condition)", placeholder="amount > 100 AND channel = 'app'", key="t_where")
        order = c2.selectbox("Order by", [None, *schema["column"]], key="t_order")
        descending = c3.checkbox("Descending", key="t_desc")
        page = c4.number_input("Page", min_value=1, value=1, step=1, key="t_page")
        try:
            frame, total = run.page(
                name, where, order, descending, offset=(int(page) - 1) * data.PAGE_ROWS, masker=masker
            )
        except ExplorerError as exc:
            st.error(str(exc))
            return
        pages = max(1, -(-total // data.PAGE_ROWS))
        st.caption(f"{total:,} rows match · page {int(page)} of {pages:,} ({data.PAGE_ROWS} rows per page)")
        st.dataframe(frame, hide_index=True, width="stretch", height=480)
        _download(frame, f"{run.run_id}_{name}_page{int(page)}.csv", key="dl_table")
    with sql_tab:
        st.caption(
            "One read-only SELECT over the tables of this run and the datasets. "
            f"At most {data.MAX_SQL_ROWS:,} rows come back. Tables: " + ", ".join(f"`{t}`" for t in list(tables)[:40])
        )
        query = st.text_area(
            "Query", height=140, key="t_sql", placeholder=f'SELECT * FROM "{next(iter(tables))}" LIMIT 100'
        )
        if st.button("Run query", key="t_run") and query.strip():
            try:
                result = run.sql(query, masker)
            except ExplorerError as exc:
                st.error(str(exc))
            else:
                st.caption(f"{len(result):,} rows")
                st.dataframe(result, hide_index=True, width="stretch", height=420)
                _download(result, f"{run.run_id}_query.csv", key="dl_sql")
    with all_tab:
        kinds = st.multiselect("Type", sorted(artifacts["type"].unique()), key="a_type")
        shown = artifacts[artifacts["type"].isin(kinds)] if kinds else artifacts
        event = st.dataframe(
            shown.drop(columns=["path"]),
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="a_table",
        )
        row = _selected_row(event)
        if row is not None and row < len(shown):
            record = shown.iloc[row]
            path = run.path / record["path"]
            st.markdown(f"**{record['id']}** (`{record['path']}`)")
            if record["type"] == "plot" and path.exists():
                st.image(str(path))
            elif record["type"] == "json" and path.exists():
                st.json(run.read_json(record["id"]), expanded=False)
            else:
                st.caption('A table: open it in "Browse a table".')


def view_steps(run: Run) -> None:
    st.header("Steps")
    steps = run.steps()
    if steps.empty:
        st.info("This run has no trace.")
        return
    summary = run.agent_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tool calls", len(steps))
    c2.metric("Errors", int((steps["status"] == "error").sum()))
    c3.metric("Time in tools", f"{steps['duration_s'].fillna(0).sum() / 60:.1f} min")
    c4.metric("Tokens (in + out)", f"{int(summary['input_tokens'].sum() + summary['output_tokens'].sum()):,}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Per agent**")
        st.dataframe(summary, hide_index=True, width="stretch")
    with right:
        st.markdown("**Where the time went** (seconds per tool, top 15)")
        per_tool = steps.groupby("tool")["duration_s"].sum().sort_values(ascending=False).head(15)
        st.bar_chart(per_tool, horizontal=True, height=320)
    c1, c2, c3 = st.columns(3)
    agents = c1.multiselect("Agent", sorted(steps["agent"].dropna().unique()), key="s_agent")
    only_errors = c2.checkbox("Errors only", key="s_errors")
    slow = c3.number_input("Slower than (seconds)", min_value=0.0, value=0.0, step=1.0, key="s_slow")
    shown = steps
    if agents:
        shown = shown[shown["agent"].isin(agents)]
    if only_errors:
        shown = shown[shown["status"] == "error"]
    if slow:
        shown = shown[shown["duration_s"].fillna(0) >= slow]
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        height=480,
        column_config={"arguments": st.column_config.TextColumn(width="large")},
    )


def view_snapshots(snapshots: pd.DataFrame) -> None:
    st.header("Snapshots")
    st.caption(
        "Local copies of external sources, taken with `jellybean snapshot` or `jellybean run --source`. "
        "A snapshot no run uses can be deleted from disk; the runs that used one keep their findings and report."
    )
    if snapshots.empty:
        st.info("No snapshot yet. Take one with `jellybean snapshot <source> --from ... --to ...`.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Snapshots", len(snapshots))
    c2.metric("Rows", f"{int(snapshots['rows'].sum()):,}")
    c3.metric("Size on disk", f"{snapshots['size_mb'].sum():,.1f} MB")
    st.dataframe(
        snapshots,
        hide_index=True,
        width="stretch",
        column_config={
            "tables": st.column_config.TextColumn(width="large"),
            "size_mb": st.column_config.NumberColumn("size (MB)", format="%.1f"),
        },
    )


def _source_notice(run: Run) -> None:
    """Say, on every page of a run made from a source, that its data is a copy with a window and a sample."""
    source = run.source
    if not source:
        return
    limits = []
    if source.get("start") and source.get("end"):
        limits.append(f"rows from {source['start']} to {source['end']} (end excluded)")
    if source.get("sample") is not None:
        tables = ", ".join(source.get("sampled_tables") or [])
        limits.append(f"{float(source['sample']) * 100:g}% of the entities of {tables}")
    text = f"Data from source `{source.get('name')}` (snapshot `{source.get('snapshot')}`)"
    st.caption(text + (": " + "; ".join(limits) + ". Rows outside it are not here." if limits else "."))
    if run.raw_missing:
        st.warning(
            "The snapshot this run read is no longer on disk: the report, findings and artifacts are shown, "
            "but the source tables and the events of a member are not available."
        )


# --------------------------------------------------------------------------- #
# Page                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Jellybean explorer", layout="wide")
    settings = get_settings()
    runs = data.list_runs(settings.workspace_root)
    ids = runs["run"].tolist()

    if "_goto" in st.session_state:
        st.session_state["run"], st.session_state["view"] = st.session_state.pop("_goto")
    with st.sidebar:
        logo = _logo_svg()
        if logo:
            st.image(logo, width=72)
        st.title("Jellybean")
        st.caption("Results explorer (read-only)")
        if ids and st.session_state.get("run") not in ids:
            st.session_state["run"] = ids[0]
        st.selectbox("Run", ids, key="run", disabled=not ids)
        st.radio("View", VIEWS, key="view")
        mask = st.toggle("Mask personal data", value=False, help="Hide identifiers the way the agents see them.")
        st.caption(f"Workspace: `{settings.workspace_root}`")
        if st.button("Reload"):
            st.rerun()

    view = st.session_state.get("view", "Runs")
    if view == "Snapshots":
        view_snapshots(data.list_snapshots(settings.snapshots_root, settings.workspace_root))
        return
    if view == "Runs" or not ids:
        view_runs(runs)
        return
    run = Run(
        settings.workspace_root,
        st.session_state["run"],
        data_root=settings.data_root,
        snapshots_root=settings.snapshots_root,
    )
    masker = data.make_masker(settings, run.run_id) if mask else None
    _source_notice(run)
    try:
        if view == "Overview":
            view_overview(run)
        elif view == "Findings":
            view_findings(run, masker)
        elif view == "Tables":
            view_tables(run, masker)
        else:
            view_steps(run)
    except ExplorerError as exc:
        st.error(str(exc))


if __name__ == "__main__":  # Streamlit runs the file as a script
    main()
