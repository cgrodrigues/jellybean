# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Dataset discovery and DuckDB connections.

A *dataset* is any sub-folder of the data root that contains Parquet files, at
any depth (Hive partitions such as ``date=2026-10-01/`` are detected). Datasets
and table artifacts are exposed to SQL as **views named after them**, so agents
query ``SELECT ... FROM traffic`` and never handle file paths.

:func:`connect` opens an in-memory DuckDB connection with those views. In
sandboxed mode (the default) the connection can only touch files under the
data root and the run workspace, extension auto-loading is disabled and the
configuration is locked, so a query cannot widen its own permissions.

Every connection has a memory limit, a thread limit and a spill directory inside
the run folder (``limits`` in ``config.yaml``), so a heavy query slows down and
uses disk instead of exhausting the machine. Connections are registered with the
tool call that opened them, which interrupts them on timeout or Ctrl+C.
"""

from __future__ import annotations

import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from jellybean.runtime.cancellation import track_connection
from jellybean.runtime.context import RunContext, maybe_current_run
from jellybean.settings import Settings, get_settings

# Dataset names become view names, so only plain identifiers are accepted.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PARTITION_RE = re.compile(r"^([^=/]+)=([^/]*)$")


class SourceNotFoundError(LookupError):
    """Raised when a name is neither a dataset nor a table artifact."""


@dataclass(frozen=True)
class DatasetInfo:
    """A Parquet dataset folder."""

    name: str
    path: Path
    files: tuple[Path, ...]
    partition_columns: tuple[str, ...] = field(default_factory=tuple)

    @property
    def glob(self) -> str:
        return str(self.path / "**" / "*.parquet")

    @property
    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.files)

    def relation_sql(self, where: str | None = None) -> str:
        """``read_parquet`` expression covering every file of the dataset, optionally filtered."""
        hive = "true" if self.partition_columns else "false"
        relation = f"read_parquet('{sql_escape(self.glob)}', hive_partitioning={hive}, union_by_name=true)"
        return f"(SELECT * FROM {relation} WHERE {where})" if where else relation


def dataset_filter(name: str, ctx: RunContext | None = None) -> str | None:
    """Row filter of a dataset in the active run (``jellybean run --filter``), if any."""
    ctx = ctx or maybe_current_run()
    return ctx.filters.get(name) if ctx is not None else None


def sql_escape(text: str) -> str:
    """Escape text for a single-quoted SQL literal."""
    return text.replace("'", "''")


def quote_ident(name: str) -> str:
    """Quote an identifier for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def discover_datasets(data_root: Path) -> dict[str, DatasetInfo]:
    """Find every dataset folder directly under ``data_root``."""
    datasets: dict[str, DatasetInfo] = {}
    if not data_root.is_dir():
        return datasets
    for folder in sorted(p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        if not _IDENT_RE.match(folder.name):
            continue  # cannot be exposed as a view; skipped silently
        files = tuple(sorted(folder.rglob("*.parquet")))
        if not files:
            continue
        partitions: list[str] = []
        for part in files[0].relative_to(folder).parts[:-1]:
            m = _PARTITION_RE.match(part)
            if m:
                partitions.append(m.group(1))
        datasets[folder.name] = DatasetInfo(folder.name, folder, files, tuple(partitions))
    return datasets


def _table_artifacts(ctx: RunContext) -> dict[str, Path]:
    """Table artifacts of the run, as ``{artifact_id: absolute_path}``."""
    from jellybean.skills._shared.artifacts import ArtifactStore

    if not ctx.manifest_path.exists():
        return {}
    return {r.id: ctx.run_dir / r.path for r in ArtifactStore(ctx).list("table")}


def connect(
    settings: Settings | None = None,
    ctx: RunContext | None = None,
    sandboxed: bool = True,
) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with dataset and artifact views.

    Args:
        settings: Settings to use (defaults to the active run or global ones).
        ctx: Run whose artifacts are exposed (defaults to the active run).
        sandboxed: Restrict file access to the data root and the run folder.
    """
    ctx = ctx or maybe_current_run()
    settings = settings or (ctx.settings if ctx else get_settings())
    con = duckdb.connect(":memory:")
    try:
        # A connection opened by a tool call is interrupted and closed if the call is cancelled.
        track_connection(con)
        allowed = [str(settings.data_root) + "/"]
        # Spill directory: inside the run folder, or a private temporary folder outside a run.
        tmp = ctx.tmp_dir if ctx is not None else Path(tempfile.gettempdir()) / "jellybean-duckdb"
        tmp.mkdir(parents=True, exist_ok=True)
        allowed.append((str(ctx.run_dir) if ctx is not None else str(tmp)) + "/")
        if sandboxed:
            dirs = ", ".join(f"'{sql_escape(d)}'" for d in allowed)
            con.execute(f"SET allowed_directories = [{dirs}]")
        # Resource limits (config.yaml, `limits`): bounded memory and threads, spilling to disk.
        limits = settings.config.limits
        con.execute(f"SET temp_directory = '{sql_escape(str(tmp))}'")
        if limits.duckdb_memory_limit:
            con.execute(f"SET memory_limit = '{sql_escape(limits.duckdb_memory_limit)}'")
        if limits.duckdb_threads:
            con.execute(f"SET threads = {int(limits.duckdb_threads)}")
        if limits.duckdb_max_temp_size:
            con.execute(f"SET max_temp_directory_size = '{sql_escape(limits.duckdb_max_temp_size)}'")

        for info in discover_datasets(settings.data_root).values():
            where = dataset_filter(info.name, ctx)
            con.execute(f"CREATE VIEW {quote_ident(info.name)} AS SELECT * FROM {info.relation_sql(where)}")
        if ctx is not None:
            datasets = set(discover_datasets(settings.data_root))
            for artifact_id, path in _table_artifacts(ctx).items():
                if artifact_id in datasets:
                    continue  # dataset names take precedence
                con.execute(
                    f"CREATE VIEW {quote_ident(artifact_id)} AS SELECT * FROM read_parquet('{sql_escape(str(path))}')"
                )

        if sandboxed:
            # Order matters: directories first, then disable everything else and lock.
            con.execute("SET enable_external_access = false")
            con.execute("SET autoinstall_known_extensions = false")
            con.execute("SET autoload_known_extensions = false")
            con.execute("SET lock_configuration = true")
    except Exception:
        con.close()
        raise
    return con


FILE_COLUMN = "__file"


def dataset_relation_with_files(info: DatasetInfo) -> str:
    """Relation of a dataset with an internal ``__file`` column (path relative to the dataset).

    Provenance stays inside trusted tool code: the LLM only ever sees
    aggregated, relative file names.
    """
    hive = "true" if info.partition_columns else "false"
    prefix = sql_escape(str(info.path) + "/")
    where = dataset_filter(info.name)
    return (
        f"(SELECT * EXCLUDE (filename), replace(filename, '{prefix}', '') AS {FILE_COLUMN} FROM "
        f"read_parquet('{sql_escape(info.glob)}', hive_partitioning={hive}, union_by_name=true, filename=true)"
        f"{f' WHERE {where}' if where else ''})"
    )


@dataclass(frozen=True)
class Source:
    """A queryable name: either a dataset or a table artifact."""

    name: str
    kind: str  # "dataset" | "artifact"
    dataset: DatasetInfo | None = None

    @property
    def view(self) -> str:
        return quote_ident(self.name)


def resolve_source(name: str, ctx: RunContext | None = None, settings: Settings | None = None) -> Source:
    """Resolve a dataset name or table artifact id into a :class:`Source`."""
    ctx = ctx or maybe_current_run()
    settings = settings or (ctx.settings if ctx else get_settings())
    datasets = discover_datasets(settings.data_root)
    if name in datasets:
        return Source(name, "dataset", datasets[name])
    if ctx is not None and name in _table_artifacts(ctx):
        return Source(name, "artifact")
    if ctx is not None:
        from jellybean.skills._shared.artifacts import ArtifactStore

        store = ArtifactStore(ctx)
        if store.exists(name):
            kind = store.get(name).type
            how = "read it with read_json_artifact" if kind == "json" else "it cannot be queried"
            raise SourceNotFoundError(
                f"'{name}' is a {kind} artifact, not a table: {how} (get_artifact_info shows what it is)."
            )
    available = sorted(datasets)
    raise SourceNotFoundError(f"Unknown dataset or table artifact '{name}'. Datasets: {available}")


# Writes to run.duckdb are serialised; the file is only a convenience catalog.
_RUN_DB_LOCK = threading.Lock()


def register_run_db_view(ctx: RunContext, view_name: str, parquet_path: Path) -> None:
    """Add a view for a table artifact to ``run.duckdb`` (best effort)."""
    with _RUN_DB_LOCK:
        try:
            con = duckdb.connect(str(ctx.db_path))
        except duckdb.Error:
            return  # e.g. file opened by a human in another process
        try:
            con.execute(
                f"CREATE OR REPLACE VIEW {quote_ident(view_name)} AS "
                f"SELECT * FROM read_parquet('{sql_escape(str(parquet_path))}')"
            )
        finally:
            con.close()


def init_run_database(ctx: RunContext) -> None:
    """Create ``run.duckdb`` with one view per input dataset."""
    with _RUN_DB_LOCK:
        con = duckdb.connect(str(ctx.db_path))
        try:
            for info in discover_datasets(ctx.settings.data_root).values():
                where = dataset_filter(info.name, ctx)
                con.execute(
                    f"CREATE OR REPLACE VIEW {quote_ident(info.name)} AS SELECT * FROM {info.relation_sql(where)}"
                )
        finally:
            con.close()
