"""Shared data tools available to every agent that declares them (spec section 5).

All tools return small summaries; raw rows (at most ``max_raw_rows``) are
always placed under ``summary["rows"]`` so the decorator can enforce the cap.
"""

from __future__ import annotations

import difflib
import re
from typing import Annotated, Any

from pydantic import Field

from jellybean.runtime.context import maybe_current_run
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.settings import get_settings
from jellybean.skills._shared.artifacts import ArtifactNotFoundError, ArtifactStore
from jellybean.skills._shared.catalog import (
    SourceNotFoundError,
    connect,
    discover_datasets,
    quote_ident,
    resolve_source,
    sql_escape,
)
from jellybean.skills._shared.masking import Masker
from jellybean.skills._shared.sql_guard import SQLRejectedError, validate_select

# Longest README excerpt returned by read_dataset_notes.
_MAX_NOTES_CHARS = 3000
_SAVE_AS_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def _settings():
    """Settings of the active run, or the global ones outside a run."""
    run = maybe_current_run()
    return run.settings if run else get_settings()


def _masker() -> Masker:
    run = maybe_current_run()
    return Masker(_settings().config.masking, salt=run.run_id if run else "")


def _normalise_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _fetch_preview(con: Any, query: str, limit: int) -> tuple[list[str], list[list[Any]]]:
    """Return column names and at most ``limit`` rows of ``query``."""
    rel = con.sql(query)
    columns = list(rel.columns)
    rows = [list(r) for r in rel.limit(limit).fetchall()]
    return columns, rows


@tool
def list_datasets() -> ToolResult:
    """List the datasets (folders of Parquet files) available in the data root.

    Returns per dataset the number of files, number of rows, size on disk,
    Hive partition columns and whether a README.md with notes exists.
    """
    datasets = discover_datasets(_settings().data_root)
    if not datasets:
        return ToolResult.ok({"datasets": []}, warnings=["No datasets found in the data root."])
    con = connect()
    try:
        items = []
        for info in datasets.values():
            rows = con.execute(f"SELECT count(*) FROM {quote_ident(info.name)}").fetchall()[0][0]
            items.append(
                {
                    "name": info.name,
                    "files": len(info.files),
                    "rows": int(rows),
                    "size_mb": round(info.size_bytes / 1e6, 2),
                    "partition_columns": list(info.partition_columns),
                    "has_readme": (info.path / "README.md").exists(),
                }
            )
    finally:
        con.close()
    return ToolResult.ok({"datasets": items})


@tool
def resolve_dataset(name: str) -> ToolResult:
    """Map a dataset name mentioned in the instruction to an existing dataset.

    Matching is case-insensitive and ignores spaces/hyphens. When there is no
    exact match, `match` is null and close candidates are returned instead.

    Args:
        name: Dataset name as written by the user.
    """
    datasets = discover_datasets(_settings().data_root)
    names = sorted(datasets)
    wanted = _normalise_name(name)
    by_norm = {_normalise_name(n): n for n in names}
    if wanted in by_norm:
        match = by_norm[wanted]
        return ToolResult.ok({"query": name, "match": match, "path": str(datasets[match].path), "candidates": []})
    close = difflib.get_close_matches(wanted, list(by_norm), n=5, cutoff=0.5)
    contains = [n for n in by_norm if wanted in n or n in wanted]
    candidates = [by_norm[n] for n in dict.fromkeys(close + contains)]
    return ToolResult.ok(
        {"query": name, "match": None, "candidates": candidates, "available": names[:50]},
        warnings=[f"No exact match for '{name}'. Ask the user to choose among the candidates or available datasets."],
    )


@tool
def get_schema(dataset_or_artifact: str) -> ToolResult:
    """Return columns and types of a dataset or table artifact, plus partitions.

    Args:
        dataset_or_artifact: Dataset name or table artifact id.
    """
    try:
        source = resolve_source(dataset_or_artifact)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        described = con.execute(f"DESCRIBE SELECT * FROM {source.view}").fetchall()
    finally:
        con.close()
    summary: dict[str, Any] = {
        "source": source.name,
        "kind": source.kind,
        "columns": [{"name": r[0], "type": r[1]} for r in described],
    }
    if source.dataset is not None:
        summary["files"] = len(source.dataset.files)
        summary["partition_columns"] = list(source.dataset.partition_columns)
    return ToolResult.ok(summary)


@tool
def run_sql(query: str, save_as: str | None = None) -> ToolResult:
    """Run a read-only SQL query (DuckDB dialect) over datasets and artifacts.

    Datasets and table artifacts are available as views with their own names,
    e.g. `SELECT count(*) FROM traffic`. Only a single SELECT/WITH statement is
    accepted. At most 20 rows are returned; use `save_as` to keep the full
    result as a new table artifact.

    Args:
        query: A single SELECT or WITH query.
        save_as: Optional artifact prefix (lowercase, digits, '_') to store the result.
    """
    settings = _settings()
    try:
        clean = validate_select(query, forbidden_paths=[settings.ground_truth_root])
    except SQLRejectedError as exc:
        return ToolResult.error(f"Query rejected: {exc}")
    if save_as is not None and not _SAVE_AS_RE.match(save_as):
        return ToolResult.error("save_as must match ^[a-z][a-z0-9_]{0,40}$")

    max_rows = settings.config.tool_output.max_raw_rows
    con = connect()
    try:
        artifact_id = None
        warnings: list[str] = []
        if save_as is not None:
            run = maybe_current_run()
            if run is None:
                return ToolResult.error("save_as requires an active run.")
            store = ArtifactStore(run)
            # Parents: dataset/artifact views referenced in the query text.
            known = set(discover_datasets(settings.data_root)) | {r.id for r in store.list()}
            parents = sorted(n for n in known if re.search(rf"\b{re.escape(n)}\b", clean))
            record = store.save_table_from_sql(con, clean, save_as, parents=parents, params={"query": clean})
            artifact_id, row_count = record.id, record.row_count
            saved = sql_escape(str(store.path_of(record.id)))
            columns, rows = _fetch_preview(con, f"SELECT * FROM read_parquet('{saved}')", max_rows)
        else:
            columns, rows = _fetch_preview(con, clean, max_rows)
            row_count = int(con.execute(f"SELECT count(*) FROM ({clean}) AS _q").fetchall()[0][0])
    finally:
        con.close()

    masked: list[str] = []
    if settings.config.masking.apply_to_run_sql:
        rows, masked = _masker().mask_rows(columns, rows)
    if row_count > len(rows):
        warnings.append(f"Showing {len(rows)} of {row_count} rows.")
    return ToolResult.ok(
        {"columns": columns, "row_count": row_count, "rows": rows, "masked_columns": masked},
        artifact_id=artifact_id,
        warnings=warnings,
    )


@tool
def sample_rows(
    dataset_or_artifact: str,
    n: Annotated[int, Field(ge=1, le=20)] = 10,
    where: str | None = None,
    seed: int | None = None,
) -> ToolResult:
    """Return a deterministic random sample of rows, with personal data masked.

    Args:
        dataset_or_artifact: Dataset name or table artifact id.
        n: Number of rows (1 to 20).
        where: Optional SQL boolean filter, e.g. "duration_sec < 0".
        seed: Sampling seed (defaults to the run seed).
    """
    try:
        source = resolve_source(dataset_or_artifact)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    filter_sql = f"WHERE ({where})" if where else ""
    # Ordering by a seeded hash of the whole row is deterministic regardless of threads.
    query = f"SELECT * FROM {source.view} AS _t {filter_sql} ORDER BY hash(_t, {int(seed)}) LIMIT {n}"
    try:
        query = validate_select(query, forbidden_paths=[_settings().ground_truth_root])
    except SQLRejectedError as exc:
        return ToolResult.error(f"Filter rejected: {exc}")
    con = connect()
    try:
        columns, rows = _fetch_preview(con, query, n)
    finally:
        con.close()
    rows, masked = _masker().mask_rows(columns, rows)
    return ToolResult.ok(
        {"source": source.name, "columns": columns, "rows": rows, "masked_columns": masked, "seed": seed}
    )


@tool
def read_dataset_notes(dataset: str) -> ToolResult:
    """Read the optional README.md stored in a dataset folder (column meanings, context).

    Args:
        dataset: Dataset name.
    """
    datasets = discover_datasets(_settings().data_root)
    if dataset not in datasets:
        return ToolResult.error(f"Unknown dataset '{dataset}'. Available: {sorted(datasets)}")
    readme = next((p for p in (datasets[dataset].path / n for n in ("README.md", "readme.md")) if p.exists()), None)
    if readme is None:
        return ToolResult.ok({"dataset": dataset, "notes": None}, warnings=["No README.md for this dataset."])
    text = readme.read_text(encoding="utf-8", errors="replace")
    warnings = []
    if len(text) > _MAX_NOTES_CHARS:
        text = text[:_MAX_NOTES_CHARS]
        warnings.append(f"Notes cut to {_MAX_NOTES_CHARS} characters.")
    return ToolResult.ok({"dataset": dataset, "notes": text}, warnings=warnings)


@tool
def get_artifact_info(artifact_id: str) -> ToolResult:
    """Return metadata and lineage (ancestors) of an artifact.

    Args:
        artifact_id: Artifact id, e.g. "profile_traffic_v1".
    """
    run = maybe_current_run()
    if run is None:
        return ToolResult.error("No active run.")
    store = ArtifactStore(run)
    try:
        record = store.get(artifact_id)
        lineage = store.lineage(artifact_id)
    except ArtifactNotFoundError:
        ids = [r.id for r in store.list()][-30:]
        return ToolResult.error(f"Unknown artifact '{artifact_id}'.", recent_artifacts=ids)
    info = record.model_dump()
    info["lineage"] = [{"id": r.id, "type": r.type, "created_by": r.created_by.model_dump()} for r in lineage]
    return ToolResult.ok(info, artifact_id=artifact_id)


# JSON artifacts an agent may read in full: plans, issues, dictionaries and the
# profiler's evidence (summaries with masked examples). Join tests (they list raw
# duplicated key values) and profiles (sample values) stay summarised by the tools
# that made them; table artifacts are queried with run_sql.
_READABLE_JSON_PREFIXES = (
    "cleaning_plan",
    "quality_issues",
    "feature_dictionary",
    "join_candidates",
    "grain",
    "entities",
    "temporal",
    "nulls",
    "invariants",
    "reference",
    "reciprocity",
    "duplicate_keys",
    "sign",
    "self_references",
)


@tool
def read_json_artifact(artifact_id: str) -> ToolResult:
    """Read the content of a JSON artifact: a plan, quality issues, a feature dictionary or the profiler's evidence.

    Readable: cleaning plans, quality issues, feature dictionaries, join candidates,
    grain, entities, temporal coverage and the integrity checks (nulls,
    invariants, references, reciprocity, duplicate keys, signs, self-references).

    Large content is shortened to fit the output limit.

    Args:
        artifact_id: e.g. "cleaning_plan_v1" or "quality_issues_v1".
    """
    run = maybe_current_run()
    if run is None:
        return ToolResult.error("No active run.")
    store = ArtifactStore(run)
    try:
        record = store.get(artifact_id)
    except ArtifactNotFoundError:
        return ToolResult.error(f"Unknown artifact '{artifact_id}'.")
    if record.type != "json" or not artifact_id.startswith(_READABLE_JSON_PREFIXES):
        return ToolResult.error(
            f"'{artifact_id}' cannot be read directly. Readable kinds: {list(_READABLE_JSON_PREFIXES)}; "
            "table artifacts are queried with run_sql (JSON artifacts are not tables)."
        )
    return ToolResult.ok({"artifact_id": artifact_id, "content": store.load_json(artifact_id)}, artifact_id=artifact_id)
