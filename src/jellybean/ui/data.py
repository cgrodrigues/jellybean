# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Read-only access to the results of finished (or running) runs, for the explorer.

Everything here reads the files of ``workspace/<run_id>/`` and never writes:
``spec.yaml``, ``manifest.json``, ``findings/*.json``, ``logs/trace.jsonl``,
``report.md`` and the Parquet artifacts. Tables are queried through an
in-memory DuckDB connection with one view per table artifact and per dataset,
so a page of a table of millions of rows costs one small query: no table is
ever loaded whole.

This module does not import Streamlit, so it can be tested on its own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

from jellybean.settings import MaskingConfig, Settings
from jellybean.skills._shared.masking import Masker
from jellybean.skills._shared.sql_guard import SQLRejectedError, validate_select

PAGE_ROWS = 200
MAX_SQL_ROWS = 5000
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ExplorerError(ValueError):
    """A problem to show to the user (unknown run, rejected query, missing file)."""


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


# --------------------------------------------------------------------------- #
# Runs                                                                        #
# --------------------------------------------------------------------------- #
def list_runs(workspace_root: Path) -> pd.DataFrame:
    """One row per run folder with a spec: id, instruction, phase, status, findings by verdict."""
    rows = []
    if workspace_root.is_dir():
        for folder in sorted(workspace_root.iterdir(), reverse=True):
            spec_path = folder / "spec.yaml"
            if not spec_path.exists():
                continue
            spec = _read_yaml(spec_path)
            result = _read_json(folder / "run_result.json") or {}
            verdicts = [f.get("critic_verdict") or "not judged" for f in _finding_docs(folder)]
            primary = next((d["name"] for d in spec.get("datasets") or [] if d.get("role") == "primary"), None)
            rows.append(
                {
                    "run": folder.name,
                    "instruction": spec.get("instruction", ""),
                    "primary": primary,
                    "entity": (spec.get("entity") or {}).get("column"),
                    "source": source_label(spec.get("source")),
                    "phase": spec.get("phase"),
                    "status": result.get("status") or "not finished",
                    "findings": len(verdicts),
                    "confirmed": verdicts.count("confirmed"),
                    "report": (folder / "report.md").exists(),
                    "modified": pd.Timestamp(spec_path.stat().st_mtime, unit="s").floor("s"),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "run",
            "instruction",
            "primary",
            "entity",
            "source",
            "phase",
            "status",
            "findings",
            "confirmed",
            "report",
            "modified",
        ],
    )


def source_label(source: dict[str, Any] | None) -> str:
    """``local``, or the source of a run with its window and sample: ``ch_prod · 2026-10-01 to 2026-11-01 · 10%``."""
    if not source:
        return "local"
    parts = [str(source.get("name"))]
    if source.get("start") and source.get("end"):
        parts.append(f"{source['start']} to {source['end']}")
    if source.get("sample") is not None:
        parts.append(f"{float(source['sample']) * 100:g}%")
    return " · ".join(parts)


def list_snapshots(snapshots_root: Path, workspace_root: Path) -> pd.DataFrame:
    """One row per local snapshot of an external source, with the runs that read it."""
    from jellybean.sources.snapshot import list_snapshots as find

    used: dict[tuple[str, str], list[str]] = {}
    if workspace_root.is_dir():
        for folder in sorted(workspace_root.iterdir()):
            source = _read_yaml(folder / "spec.yaml").get("source") if (folder / "spec.yaml").exists() else None
            if source:
                used.setdefault((str(source.get("name")), str(source.get("snapshot"))), []).append(folder.name)
    rows = []
    for item in find(snapshots_root):
        m = item.manifest
        rows.append(
            {
                "source": m.source,
                "snapshot": m.id,
                "from": m.start,
                "to": m.end,
                "sample": None if m.sample is None else f"{m.sample * 100:g}%",
                "tables": ", ".join(f"{name} ({t.rows:,})" for name, t in m.tables.items()),
                "rows": sum(t.rows for t in m.tables.values()),
                "size_mb": round(sum(t.bytes for t in m.tables.values()) / 1e6, 1),
                "created": m.created_at,
                "runs": ", ".join(used.get((m.source, m.id), [])),
                "path": str(item.path),
            }
        )
    columns = ["source", "snapshot", "from", "to", "sample", "tables", "rows", "size_mb", "created", "runs", "path"]
    return pd.DataFrame(rows, columns=columns)


def _finding_docs(run_dir: Path) -> list[dict[str, Any]]:
    folder = run_dir / "findings"
    if not folder.is_dir():
        return []
    docs = (_read_json(p) for p in sorted(folder.glob("F*.json")))
    return [d for d in docs if isinstance(d, dict)]


@dataclass
class Run:
    """The files of one run, read lazily."""

    workspace_root: Path
    run_id: str
    data_root: Path | None = None
    # Where the snapshots of external sources are kept (runs made with `--source` read from there).
    snapshots_root: Path | None = None
    _spec: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def path(self) -> Path:
        return self.workspace_root / self.run_id

    def exists(self) -> bool:
        return (self.path / "spec.yaml").exists()

    @property
    def source(self) -> dict[str, Any] | None:
        """The external source of the run's datasets (``spec.source``), or ``None`` for local data."""
        return self.spec.get("source") or None

    @property
    def raw_root(self) -> Path | None:
        """Folder of the datasets the run read: its snapshot, or the data root."""
        source = self.source
        if not source:
            return self.data_root
        if self.snapshots_root is None:
            return None
        return self.snapshots_root / str(source.get("name")) / str(source.get("snapshot"))

    @property
    def raw_missing(self) -> bool:
        """The run read a snapshot that is no longer on disk (its raw data cannot be shown)."""
        root = self.raw_root
        return self.source is not None and (root is None or not root.is_dir())

    @property
    def spec(self) -> dict[str, Any]:
        if self._spec is None:
            self._spec = _read_yaml(self.path / "spec.yaml")
        return self._spec

    @property
    def result(self) -> dict[str, Any]:
        return _read_json(self.path / "run_result.json") or {}

    def report(self) -> str | None:
        path = self.path / "report.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    # ------------------------------------------------------------ artifacts
    def artifacts(self) -> pd.DataFrame:
        """The manifest as a table: id, type, rows, tool, agent, parents, created, path."""
        manifest = _read_json(self.path / "manifest.json") or {}
        records = manifest.get("artifacts", {})
        records = list(records.values()) if isinstance(records, dict) else records
        rows = [
            {
                "id": r["id"],
                "type": r["type"],
                "rows": r.get("row_count"),
                "tool": (r.get("created_by") or {}).get("tool"),
                "agent": (r.get("created_by") or {}).get("agent"),
                "parents": ", ".join(r.get("parents") or []),
                "created": r.get("created_at"),
                "path": r["path"],
            }
            for r in records
        ]
        return pd.DataFrame(rows, columns=["id", "type", "rows", "tool", "agent", "parents", "created", "path"])

    def artifact_path(self, artifact_id: str) -> Path:
        table = self.artifacts()
        match = table[table["id"] == artifact_id]
        if match.empty:
            raise ExplorerError(f"Unknown artifact '{artifact_id}'.")
        return self.path / match.iloc[0]["path"]

    def artifact_params(self, artifact_id: str) -> dict[str, Any]:
        manifest = _read_json(self.path / "manifest.json") or {}
        records = manifest.get("artifacts", {})
        records = list(records.values()) if isinstance(records, dict) else records
        return next((r.get("params") or {} for r in records if r["id"] == artifact_id), {})

    def read_json(self, artifact_id: str) -> Any:
        return _read_json(self.artifact_path(artifact_id))

    # ------------------------------------------------------------- findings
    def findings(self) -> pd.DataFrame:
        rows = []
        for f in _finding_docs(self.path):
            checks = f.get("checks") or {}
            rows.append(
                {
                    "id": f["id"],
                    "type": f.get("type"),
                    "size": f.get("size"),
                    "verdict": f.get("critic_verdict") or "not judged",
                    "novelty": f.get("novelty"),
                    "methods": ", ".join(f.get("methods") or []),
                    "stability": (checks.get("stability") or {}).get("stability"),
                    "feedback": f.get("human_feedback"),
                    "top feature": ((f.get("top_features") or [{}])[0]).get("feature"),
                    "described": bool(f.get("description")),
                }
            )
        columns = [
            "id",
            "type",
            "size",
            "verdict",
            "novelty",
            "methods",
            "stability",
            "feedback",
            "top feature",
            "described",
        ]
        return pd.DataFrame(rows, columns=columns)

    def finding(self, finding_id: str) -> dict[str, Any]:
        doc = _read_json(self.path / "findings" / f"{finding_id}.json")
        if not isinstance(doc, dict):
            raise ExplorerError(f"Unknown finding '{finding_id}'.")
        return doc

    # ---------------------------------------------------------------- trace
    def trace(self) -> pd.DataFrame:
        """Every event of ``logs/trace.jsonl`` (one row each); empty when there is no trace."""
        path = self.path / "logs" / "trace.jsonl"
        rows = []
        if path.exists():
            with path.open(encoding="utf-8") as lines:
                for line in lines:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue  # a line being written by a running run
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
        return frame

    def steps(self) -> pd.DataFrame:
        """Tool calls in order: time, agent, tool, status, duration, artifact, error, arguments."""
        trace = self.trace()
        columns = ["ts", "agent", "tool", "status", "duration_s", "artifact_id", "error", "arguments"]
        if trace.empty or "tool" not in trace.columns:
            return pd.DataFrame(columns=columns)
        calls = trace[trace["kind"] == "tool_call"].copy()
        for column in columns:
            if column not in calls.columns:
                calls[column] = None
        calls["arguments"] = calls["arguments"].map(lambda a: json.dumps(a, ensure_ascii=False, default=str)[:300])
        return calls[columns].reset_index(drop=True)

    def agent_summary(self) -> pd.DataFrame:
        """Per agent: runs, LLM calls, tokens, tool calls, errors and time spent in tools."""
        trace = self.trace()
        columns = [
            "agent",
            "runs",
            "llm_calls",
            "input_tokens",
            "output_tokens",
            "tool_calls",
            "errors",
            "tool_seconds",
        ]
        if trace.empty:
            return pd.DataFrame(columns=columns)

        def col(frame: pd.DataFrame, name: str) -> pd.Series:
            return frame[name] if name in frame.columns else pd.Series(0, index=frame.index)

        rows = []
        for agent, events in trace.groupby("agent", sort=False):
            llm, tools = events[events["kind"] == "llm_call"], events[events["kind"] == "tool_call"]
            rows.append(
                {
                    "agent": agent,
                    "runs": int((events["kind"] == "agent_start").sum()),
                    "llm_calls": len(llm),
                    "input_tokens": int(col(llm, "input_tokens").fillna(0).sum()),
                    "output_tokens": int(col(llm, "output_tokens").fillna(0).sum()),
                    "tool_calls": len(tools),
                    "errors": int((col(tools, "status") == "error").sum()),
                    "tool_seconds": round(float(col(tools, "duration_s").fillna(0).sum()), 1),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    # ---------------------------------------------------------------- tables
    def tables(self) -> dict[str, Path]:
        """Queryable tables: table artifacts of the run, then the datasets it read (data root or snapshot)."""
        found: dict[str, Path] = {}
        artifacts = self.artifacts()
        for _, row in artifacts[artifacts["type"] == "table"].iterrows():
            path = self.path / row["path"]
            if path.exists():
                found[row["id"]] = path
        root = self.raw_root
        if root and root.is_dir():
            for folder in sorted(root.iterdir()):
                usable = folder.is_dir() and _IDENT_RE.match(folder.name) and folder.name not in found
                if usable and any(folder.rglob("*.parquet")):
                    found[folder.name] = folder
        return found

    def connect(self) -> duckdb.DuckDBPyConnection:
        """In-memory connection with one view per table (see :meth:`tables`). Close it after use."""
        con = duckdb.connect(":memory:")
        con.execute("SET memory_limit = '2GB'")
        for name, path in self.tables().items():
            source = (
                f"read_parquet('{_escape(str(path / '**' / '*.parquet'))}', hive_partitioning=true, union_by_name=true)"
                if path.is_dir()
                else f"read_parquet('{_escape(str(path))}')"
            )
            con.execute(f"CREATE VIEW {quote(name)} AS SELECT * FROM {source}")
        return con

    def schema(self, table: str) -> pd.DataFrame:
        con = self.connect()
        try:
            rows = con.execute(f"DESCRIBE SELECT * FROM {quote(self._known(table))}").fetchall()
        finally:
            con.close()
        return pd.DataFrame([(r[0], r[1]) for r in rows], columns=["column", "type"])

    def _known(self, table: str) -> str:
        if table not in self.tables():
            raise ExplorerError(f"Unknown table '{table}'.")
        return table

    def page(
        self,
        table: str,
        where: str | None = None,
        order_by: str | None = None,
        descending: bool = False,
        offset: int = 0,
        limit: int = PAGE_ROWS,
        masker: Masker | None = None,
    ) -> tuple[pd.DataFrame, int]:
        """One page of ``table`` and the number of rows matching ``where`` (a SQL condition)."""
        name = quote(self._known(table))
        condition = ""
        if where and where.strip():
            _check_condition(where)
            condition = f" WHERE {where}"
        con = self.connect()
        try:
            columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {name}").fetchall()]
            if order_by and order_by not in columns:
                raise ExplorerError(f"Unknown column '{order_by}'.")
            order = f" ORDER BY {quote(order_by)} {'DESC' if descending else 'ASC'} NULLS LAST" if order_by else ""
            total = con.execute(f"SELECT count(*) FROM {name}{condition}").fetchall()[0][0]
            frame = con.execute(
                f"SELECT * FROM {name}{condition}{order} LIMIT {int(limit)} OFFSET {max(int(offset), 0)}"
            ).df()
        except duckdb.Error as exc:
            raise ExplorerError(str(exc).splitlines()[0]) from exc
        finally:
            con.close()
        return mask_frame(frame, masker), int(total)

    def sql(self, query: str, masker: Masker | None = None, limit: int = MAX_SQL_ROWS) -> pd.DataFrame:
        """Run one read-only query over the tables of the run (at most ``limit`` rows come back)."""
        try:
            clean = validate_select(query, forbidden_paths=[])
        except SQLRejectedError as exc:
            raise ExplorerError(f"Query rejected: {exc}") from exc
        con = self.connect()
        try:
            frame = con.execute(f"SELECT * FROM ({clean}) LIMIT {int(limit)}").df()
        except duckdb.Error as exc:
            raise ExplorerError(str(exc).splitlines()[0]) from exc
        finally:
            con.close()
        return mask_frame(frame, masker)

    # ----------------------------------------------------- finding members
    def feature_table(self) -> str | None:
        return (self.spec.get("features") or {}).get("table")

    def events_table(self) -> str | None:
        """The table with one row per event: enriched, else cleaned primary, else the primary dataset."""
        spec = self.spec
        primary = next((d["name"] for d in spec.get("datasets") or [] if d.get("role") == "primary"), None)
        candidates = [spec.get("enriched_artifact"), (spec.get("prepared") or {}).get(primary or ""), primary]
        tables = self.tables()
        return next((c for c in candidates if c and c in tables), None)

    def members(self, finding_id: str, masker: Masker | None = None, limit: int = MAX_SQL_ROWS) -> pd.DataFrame:
        """The entities (or windows) of a finding, with their features when there is a feature table."""
        finding = self.finding(finding_id)
        ids, entity = finding.get("entity_ids_artifact"), finding.get("entity_column")
        tables = self.tables()
        if ids not in tables:
            raise ExplorerError(f"The members of {finding_id} are not available ({ids}).")
        features = self.feature_table()
        con = self.connect()
        try:
            id_columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {quote(ids)}").fetchall()]
            joinable = features in tables and entity in id_columns
            if joinable:
                feature_columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {quote(features)}").fetchall()]
                joinable = entity in feature_columns
            if joinable:
                e = quote(entity)
                query = (
                    f"SELECT f.* FROM {quote(features)} f WHERE CAST(f.{e} AS VARCHAR) IN "
                    f"(SELECT CAST({e} AS VARCHAR) FROM {quote(ids)}) ORDER BY f.{e} LIMIT {int(limit)}"
                )
            else:
                query = f"SELECT * FROM {quote(ids)} LIMIT {int(limit)}"
            frame = con.execute(query).df()
        finally:
            con.close()
        return mask_frame(frame, masker)

    def compare_with_population(self, finding_id: str, features: list[str]) -> pd.DataFrame:
        """Median of the finding's members and of all entities, for the given numeric features."""
        finding = self.finding(finding_id)
        ids, entity, table = finding.get("entity_ids_artifact"), finding.get("entity_column"), self.feature_table()
        tables = self.tables()
        if not features or table not in tables or ids not in tables:
            return pd.DataFrame(columns=["feature", "members", "population"])
        con = self.connect()
        try:
            described = con.execute(f"DESCRIBE SELECT * FROM {quote(table)}").fetchall()
            types = {r[0]: r[1].upper() for r in described}
            numeric = [
                f
                for f in features
                if f in types and any(k in types[f] for k in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL"))
            ]
            if not numeric or entity not in types:
                return pd.DataFrame(columns=["feature", "members", "population"])
            e = quote(entity)
            inside = f"CAST({e} AS VARCHAR) IN (SELECT CAST({e} AS VARCHAR) FROM {quote(ids)})"
            exprs = ", ".join(
                f"median({quote(f)}) FILTER (WHERE {inside}), approx_quantile({quote(f)}, 0.5)" for f in numeric
            )
            values = con.execute(f"SELECT {exprs} FROM {quote(table)}").fetchall()[0]
        finally:
            con.close()
        rows = [
            {"feature": f, "members": values[2 * i], "population": values[2 * i + 1]} for i, f in enumerate(numeric)
        ]
        return pd.DataFrame(rows, columns=["feature", "members", "population"])

    def entity_events(
        self, entity_column: str, value: Any, masker: Masker | None = None, limit: int = 1000
    ) -> tuple[pd.DataFrame, str | None]:
        """Raw events of one entity (from :meth:`events_table`), most recent first when there is a time column."""
        table = self.events_table()
        if table is None:
            return pd.DataFrame(), None
        con = self.connect()
        try:
            columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {quote(table)}").fetchall()]
            if entity_column not in columns:
                return pd.DataFrame(), table
            time_column = self.spec.get("time_column")
            order = f" ORDER BY {quote(time_column)} DESC" if time_column in columns else ""
            frame = con.execute(
                f"SELECT * FROM {quote(table)} WHERE CAST({quote(entity_column)} AS VARCHAR) = ?{order} "
                f"LIMIT {int(limit)}",
                [str(value)],
            ).df()
        finally:
            con.close()
        return mask_frame(frame, masker), table


def _escape(text: str) -> str:
    return text.replace("'", "''")


def _check_condition(condition: str) -> None:
    """A filter is one SQL condition: validated like a query, so it cannot write or read files."""
    try:
        validate_select(f"SELECT 1 FROM t WHERE {condition}", forbidden_paths=[])
    except SQLRejectedError as exc:
        raise ExplorerError(f"Filter rejected: {exc}") from exc


# --------------------------------------------------------------------------- #
# Masking                                                                     #
# --------------------------------------------------------------------------- #
def make_masker(settings: Settings | None, run_id: str) -> Masker:
    """The same masker the agents' tools use, so masked values match the ones in the report."""
    config = settings.config.masking if settings else MaskingConfig()
    return Masker(config, salt=run_id)


def mask_frame(frame: pd.DataFrame, masker: Masker | None) -> pd.DataFrame:
    """Mask the personal-data columns of ``frame`` (by name or by value pattern); None = raw values."""
    if masker is None or frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        values = out[column]
        sample = [v for v in values.head(50).tolist() if isinstance(v, str)]
        if masker.is_sensitive_name(str(column)) or masker.is_sensitive_values(sample):
            out[column] = values.map(lambda v: None if pd.isna(v) else masker.mask_value(v))
    return out
