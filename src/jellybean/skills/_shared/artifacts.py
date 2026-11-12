"""Immutable, versioned artifacts with lineage (spec section 6).

Every intermediate result produced by a tool (tables, JSON documents, plots) is
stored under ``workspace/<run_id>/artifacts/`` and registered in
``manifest.json`` with its lineage. Artifacts are never modified: a change
creates a new version (``<prefix>_v1``, ``<prefix>_v2``...). Files are made
read-only once written.
"""

from __future__ import annotations

import json
import os
import re
import stat
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from jellybean.runtime.context import RunContext, current_run
from jellybean.runtime.tool_registry import current_tool, to_jsonable

ArtifactType = Literal["table", "json", "plot", "text"]

# Artifact ids double as DuckDB view names, so they must be plain identifiers.
_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_]{0,60}$")
_ID_RE = re.compile(r"^([a-z][a-z0-9_]{0,60})_v(\d+)$")

_EXTENSIONS: dict[str, str] = {"table": "parquet", "json": "json", "plot": "png", "text": "md"}


class ArtifactNotFoundError(KeyError):
    """Raised when an artifact id is not in the manifest."""


class CreatedBy(BaseModel):
    agent: str
    tool: str | None = None


class ArtifactRecord(BaseModel):
    """One entry of ``manifest.json``."""

    id: str
    type: ArtifactType
    path: str  # relative to the run directory
    created_by: CreatedBy
    parents: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    row_count: int | None = None
    created_at: str


def slugify(text: str) -> str:
    """Turn free text into a valid artifact prefix."""
    slug = re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"a_{slug}"
    return slug[:60]


# One lock and one set of reserved ids per manifest, shared by all stores of the
# same run inside this process (tools may run in parallel threads).
_LOCKS: dict[Path, threading.Lock] = {}
_RESERVED: dict[Path, set[str]] = {}
_GLOBAL_LOCK = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _GLOBAL_LOCK:
        return _LOCKS.setdefault(path, threading.Lock())


class ArtifactStore:
    """Read and write artifacts of one run."""

    def __init__(self, ctx: RunContext | None = None) -> None:
        self.ctx = ctx or current_run()
        self.manifest_path = self.ctx.manifest_path
        self._lock = _lock_for(self.manifest_path)

    # ------------------------------------------------------------------ io --
    def _load(self) -> dict[str, ArtifactRecord]:
        if not self.manifest_path.exists():
            return {}
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {item["id"]: ArtifactRecord.model_validate(item) for item in raw.get("artifacts", [])}

    def _save(self, records: dict[str, ArtifactRecord]) -> None:
        payload = {"run_id": self.ctx.run_id, "artifacts": [r.model_dump() for r in records.values()]}
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.manifest_path)  # atomic on POSIX

    # --------------------------------------------------------------- query --
    def exists(self, artifact_id: str) -> bool:
        return artifact_id in self._load()

    def get(self, artifact_id: str) -> ArtifactRecord:
        records = self._load()
        if artifact_id not in records:
            raise ArtifactNotFoundError(artifact_id)
        return records[artifact_id]

    def list(self, type_: ArtifactType | None = None) -> list[ArtifactRecord]:
        return [r for r in self._load().values() if type_ is None or r.type == type_]

    def path_of(self, artifact_id: str) -> Path:
        """Absolute path of an artifact file."""
        return self.ctx.run_dir / self.get(artifact_id).path

    def lineage(self, artifact_id: str) -> list[ArtifactRecord]:
        """All ancestors of an artifact, nearest first (breadth-first)."""
        records = self._load()
        if artifact_id not in records:
            raise ArtifactNotFoundError(artifact_id)
        seen: list[str] = []
        queue = list(records[artifact_id].parents)
        while queue:
            parent = queue.pop(0)
            if parent in seen:
                continue
            seen.append(parent)
            if parent in records:
                queue.extend(records[parent].parents)
        return [records[p] for p in seen if p in records]

    # --------------------------------------------------------------- write --
    def _reserve_id(self, prefix: str) -> str:
        """Allocate the next free version id for ``prefix`` (thread-safe)."""
        if not _PREFIX_RE.match(prefix):
            raise ValueError(f"Invalid artifact prefix {prefix!r}: use lowercase letters, digits and '_'")
        with self._lock:
            reserved = _RESERVED.setdefault(self.manifest_path, set())
            taken = set(self._load()) | reserved
            versions = [int(m.group(2)) for i in taken if (m := _ID_RE.match(i)) and m.group(1) == prefix]
            artifact_id = f"{prefix}_v{max(versions, default=0) + 1}"
            reserved.add(artifact_id)
            return artifact_id

    def create(
        self,
        prefix: str,
        type_: ArtifactType,
        writer: Callable[[Path], int | None],
        parents: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Create a new artifact.

        Args:
            prefix: Base name; the version suffix is added automatically.
            type_: Artifact type, which also decides the file extension.
            writer: Callback that writes the file at the given path and returns
                its row count (or ``None`` for non-tabular artifacts).
            parents: Ids of artifacts or dataset names this one derives from.
            params: Parameters that produced it (kept for reproducibility).
        """
        artifact_id = self._reserve_id(prefix)
        rel_path = Path("artifacts") / f"{artifact_id}.{_EXTENSIONS[type_]}"
        abs_path = self.ctx.run_dir / rel_path
        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            row_count = writer(abs_path)
            # Make the file read-only: artifacts are immutable.
            abs_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            record = ArtifactRecord(
                id=artifact_id,
                type=type_,
                path=str(rel_path),
                created_by=CreatedBy(agent=self.ctx.agent, tool=current_tool()),
                parents=list(parents or []),
                params=to_jsonable(params or {}),
                row_count=row_count,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            with self._lock:
                records = self._load()
                if artifact_id in records:  # defensive: ids are reserved above
                    raise FileExistsError(f"Artifact {artifact_id} already exists")
                records[artifact_id] = record
                self._save(records)
        finally:
            with self._lock:
                _RESERVED.get(self.manifest_path, set()).discard(artifact_id)
        # Announce the new artifact (the CLI progress view lists it as an output).
        from jellybean.runtime.tracing import trace_event

        trace_event(
            "artifact", artifact_id=artifact_id, type=type_, path=str(abs_path), rows=row_count, tool=current_tool()
        )
        if type_ == "table":
            # Keep run.duckdb views in sync for humans inspecting the run.
            from jellybean.skills._shared.catalog import register_run_db_view

            register_run_db_view(self.ctx, artifact_id, abs_path)
        return record

    # ------------------------------------------------------- conveniences --
    def save_json(
        self, prefix: str, obj: Any, parents: list[str] | None = None, params: dict[str, Any] | None = None
    ) -> ArtifactRecord:
        """Store a JSON document as an artifact."""

        def write(path: Path) -> int | None:
            path.write_text(json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")
            return len(obj) if isinstance(obj, list) else None

        return self.create(prefix, "json", write, parents, params)

    def load_json(self, artifact_id: str) -> Any:
        return json.loads(self.path_of(artifact_id).read_text(encoding="utf-8"))

    def save_table_from_sql(
        self,
        con: Any,
        query: str,
        prefix: str,
        parents: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Materialise a DuckDB query as a Parquet artifact (no pandas involved)."""

        def write(path: Path) -> int:
            con.execute(f"COPY ({query}) TO '{_escape(str(path))}' (FORMAT parquet, COMPRESSION zstd)")
            return int(con.execute(f"SELECT count(*) FROM read_parquet('{_escape(str(path))}')").fetchall()[0][0])

        return self.create(prefix, "table", write, parents, params)

    def save_dataframe(
        self, prefix: str, df: Any, parents: list[str] | None = None, params: dict[str, Any] | None = None
    ) -> ArtifactRecord:
        """Store a (small, per-entity) pandas DataFrame as a Parquet artifact."""

        def write(path: Path) -> int:
            df.to_parquet(path, index=False)
            return len(df)

        return self.create(prefix, "table", write, parents, params)


def _escape(text: str) -> str:
    """Escape a value for use inside a single-quoted SQL literal."""
    return text.replace("'", "''")
