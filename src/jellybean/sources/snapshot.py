# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Copy the tables of an external source to a local folder of Parquet datasets.

A snapshot is ``<snapshots_root>/<source>/<id>/`` with one folder per table and
a ``snapshot.json`` describing what was copied (window, sample, queries, rows).
The source does the filtering: only the rows of the time window, of the sampled
entities and of the configured columns leave it.

* **Window** (``start <= time_column < end``): required when a table has a
  ``time_column``; such a table is copied one day per file, so an interrupted
  copy continues where it stopped.
* **Sample** (share of entities): entities are chosen by a hash of their id, so
  every row of a sampled entity is kept and the same entities are chosen in every
  table and on every run.
* **Reuse**: the id is derived from the source's table configuration, the window
  and the sample; asking for the same thing again returns the existing snapshot.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from jellybean.settings import Settings, SourceConfig, SourceTableConfig

MANIFEST = "snapshot.json"
_PARTIAL = ".partial"
_SAMPLE_BUCKETS = 1_000_000


class SnapshotError(RuntimeError):
    """The snapshot cannot be taken or found; the message says what to change."""


class SourceClient(Protocol):
    """What a source must offer; the queries are written in the source's SQL."""

    def columns(self, database: str, table: str) -> list[tuple[str, str]]:
        """Column names and types of a table, in order."""
        ...

    def stream_parquet(self, sql: str) -> Iterable[bytes]:
        """Result of a query as the bytes of one Parquet file."""
        ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SnapshotRequest:
    """What to copy: a time window (end exclusive) and an optional share of entities."""

    start: datetime | None = None
    end: datetime | None = None
    sample: float | None = None


class TableManifest(BaseModel):
    table: str
    rows: int
    files: int
    bytes: int
    time_column: str | None = None
    entity_column: str | None = None
    windowed: bool = False
    sampled: bool = False
    # The query of one file, with the window of the first one.
    query: str = ""


class SnapshotManifest(BaseModel):
    source: str
    type: str
    database: str
    id: str
    created_at: str
    start: str | None = None
    end: str | None = None
    sample: float | None = None
    tables: dict[str, TableManifest] = Field(default_factory=dict)

    def describe(self) -> str:
        """One line for people: ``ch_prod, 2026-10-01 to 2026-11-01, 10% of entities``."""
        parts = [self.source]
        if self.start and self.end:
            parts.append(f"{self.start} to {self.end}")
        if self.sample is not None:
            parts.append(f"{self.sample * 100:g}% of entities")
        return ", ".join(parts)


@dataclass(frozen=True)
class Snapshot:
    path: Path
    manifest: SnapshotManifest
    reused: bool = False


# --------------------------------------------------------------------------- #
# Request                                                                     #
# --------------------------------------------------------------------------- #
def parse_moment(text: str) -> datetime:
    """Parse ``2026-10-01`` or ``2026-10-01T06:00`` (no time zone: the source's own)."""
    try:
        moment = datetime.fromisoformat(text.strip())
    except ValueError as exc:
        raise SnapshotError(f"'{text}' is not a date (use 2026-10-01 or 2026-10-01T06:00).") from exc
    if moment.tzinfo is not None:
        raise SnapshotError(f"'{text}': give the moment without a time zone; the source's own is used.")
    return moment.replace(microsecond=0)


def make_request(start: str | None, end: str | None, sample: float | None) -> SnapshotRequest:
    """Validate the options of the command line."""
    if (start is None) != (end is None):
        raise SnapshotError("Give both --from and --to (the end is exclusive), or neither.")
    first, last = (parse_moment(start), parse_moment(end)) if start and end else (None, None)
    if first is not None and last is not None and first >= last:
        raise SnapshotError("--from must be before --to.")
    if sample is not None and not 0 < sample <= 1:
        raise SnapshotError("--sample is a share of the entities: above 0 and at most 1 (0.1 = 10%).")
    return SnapshotRequest(first, last, None if sample == 1 else sample)


def _text(moment: datetime) -> str:
    return moment.date().isoformat() if moment.time() == time(0) else moment.isoformat(timespec="seconds")


def _check(name: str, source: SourceConfig, request: SnapshotRequest) -> None:
    timed = sorted(t for t, cfg in source.tables.items() if cfg.time_column)
    if timed and request.start is None:
        raise SnapshotError(
            f"Source '{name}': {', '.join(timed)} have a time_column, so a window is required (--from and --to)."
        )
    if request.sample is not None and not any(cfg.entity_column for cfg in source.tables.values()):
        raise SnapshotError(
            f"Source '{name}': --sample needs an entity_column on at least one table (config.yaml, sources)."
        )


def snapshot_id(source: SourceConfig, request: SnapshotRequest) -> str:
    """Stable id of a snapshot: readable window and sample, plus a digest of what defines its content."""
    window = "all"
    if request.start and request.end:
        fmt = "%Y%m%d" if request.start.time() == request.end.time() == time(0) else "%Y%m%dT%H%M%S"
        window = f"{request.start:{fmt}}-{request.end:{fmt}}"
    sample = f"_s{request.sample * 100:g}".replace(".", "p") if request.sample is not None else ""
    content = {
        "type": source.type,
        "database": source.database,
        "tables": {name: cfg.model_dump() for name, cfg in sorted(source.tables.items())},
        "start": request.start.isoformat() if request.start else None,
        "end": request.end.isoformat() if request.end else None,
        "sample": request.sample,
    }
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:8]
    return f"{window}{sample}_{digest}"


# --------------------------------------------------------------------------- #
# Queries (ClickHouse SQL)                                                    #
# --------------------------------------------------------------------------- #
def _quote(name: str) -> str:
    return "`" + name.replace("`", "") + "`"


_WRAPPER_RE = re.compile(r"^(?:Nullable|LowCardinality)\((.*)\)$")


def _inner_type(kind: str) -> str:
    while (m := _WRAPPER_RE.match(kind.strip())) is not None:
        kind = m.group(1)
    return kind.strip()


def select_expression(name: str, kind: str) -> str:
    """Column expression whose Parquet type does not depend on the server or on this machine.

    * Timestamps are shifted to the wall-clock time of the column's time zone (the zone
      is then dropped from the file, see :func:`_drop_time_zones`), so the hour of an
      event is the one people of the source see, on any machine.
    * Older servers write ``DateTime`` and ``Date`` as plain integers, and UUIDs, IP
      addresses and enums as bytes or codes; they are converted in the query.
    """
    column, inner = _quote(name), _inner_type(kind)
    if inner.startswith("DateTime64"):
        return f"addSeconds({column}, timezoneOffset({column})) AS {column}"
    if inner.startswith("DateTime"):
        return f"addSeconds(toDateTime64({column}, 0), timezoneOffset({column})) AS {column}"
    if inner == "Date":
        return f"toDate32({column}) AS {column}"
    if inner in ("UUID", "IPv4", "IPv6") or inner.startswith("Enum"):
        return f"toString({column}) AS {column}"
    return column


_ZONE_RE = re.compile(r"'([A-Za-z0-9_+\-/]+)'\s*\)$")


def _moment_sql(moment: datetime, kind: str) -> str:
    """A moment of the window, in the time zone of the column it is compared with."""
    zone = _ZONE_RE.search(_inner_type(kind))
    return f"toDateTime('{moment:%Y-%m-%d %H:%M:%S}'" + (f", '{zone.group(1)}')" if zone else ")")


@dataclass(frozen=True)
class _Part:
    file: str
    sql: str


@dataclass(frozen=True)
class _TablePlan:
    dataset: str
    table: str
    config: SourceTableConfig
    parts: list[_Part]
    windowed: bool
    sampled: bool


def _days(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split a window at midnights: one piece per day."""
    pieces, cursor = [], start
    while cursor < end:
        following = min(datetime.combine(cursor.date() + timedelta(days=1), time(0)), end)
        pieces.append((cursor, following))
        cursor = following
    return pieces


def plan_table(
    dataset: str,
    config: SourceTableConfig,
    database: str,
    columns: list[tuple[str, str]],
    request: SnapshotRequest,
) -> _TablePlan:
    """The queries that copy one table, one per output file."""
    table = config.table or dataset
    kinds = dict(columns)
    if not kinds:
        raise SnapshotError(f"Table {database}.{table} was not found or has no columns.")
    wanted = config.columns or [name for name, _ in columns]
    needed = [*wanted, *(c for c in (config.time_column, config.entity_column) if c)]
    missing = sorted({c for c in needed if c not in kinds})
    if missing:
        raise SnapshotError(
            f"Table {database}.{table} has no column {', '.join(missing)} (it has: {', '.join(sorted(kinds))})."
        )
    # Rows are filtered on the table's own columns in an inner query, and converted outside:
    # ClickHouse resolves an alias in WHERE, so a converted `ts AS ts` would shift the window.
    converted = ", ".join(select_expression(name, kinds[name]) for name in wanted)
    plain = ", ".join(_quote(name) for name in wanted)
    conditions = [f"({config.where})"] if config.where else []
    sampled = request.sample is not None and config.entity_column is not None
    if sampled:
        buckets = round(request.sample * _SAMPLE_BUCKETS)  # type: ignore[operator]
        conditions.append(f"cityHash64(toString({_quote(config.entity_column)})) % {_SAMPLE_BUCKETS} < {buckets}")  # type: ignore[arg-type]
    source = f"{_quote(database)}.{_quote(table)}"

    def query(extra: list[str]) -> str:
        where = conditions + extra
        inner = f"SELECT {plain} FROM {source}" + (" WHERE " + " AND ".join(where) if where else "")
        return inner if converted == plain else f"SELECT {converted} FROM ({inner})"

    windowed = config.time_column is not None and request.start is not None and request.end is not None
    if not windowed:
        return _TablePlan(dataset, table, config, [_Part("part-all.parquet", query([]))], False, sampled)
    time_kind = kinds[config.time_column]  # type: ignore[index]
    if not _inner_type(time_kind).startswith("Date"):
        raise SnapshotError(
            f"{database}.{table}.{config.time_column} is {time_kind}: the time_column must be a Date or DateTime."
        )
    moment = _quote(config.time_column)  # type: ignore[arg-type]
    parts = []
    for first, last in _days(request.start, request.end):  # type: ignore[arg-type]
        name = f"part-{first:%Y-%m-%d}" + ("" if first.time() == time(0) else f"T{first:%H%M%S}")
        window = [f"{moment} >= {_moment_sql(first, time_kind)}", f"{moment} < {_moment_sql(last, time_kind)}"]
        parts.append(_Part(f"{name}.parquet", query(window)))
    return _TablePlan(dataset, table, config, parts, True, sampled)


# --------------------------------------------------------------------------- #
# Copy                                                                        #
# --------------------------------------------------------------------------- #
def snapshot_dir(settings: Settings, source: str, snapshot: str) -> Path:
    return settings.snapshots_root / source / snapshot


def load_manifest(folder: Path) -> SnapshotManifest | None:
    """Manifest of a finished snapshot, or ``None`` when the folder is not one."""
    path = folder / MANIFEST
    if not path.is_file():
        return None
    try:
        return SnapshotManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def list_snapshots(snapshots_root: Path) -> list[Snapshot]:
    """Every finished snapshot, newest first."""
    found = []
    if snapshots_root.is_dir():
        for folder in snapshots_root.glob("*/*"):
            manifest = load_manifest(folder) if folder.is_dir() and not folder.name.endswith(_PARTIAL) else None
            if manifest is not None:
                found.append(Snapshot(folder, manifest))
    return sorted(found, key=lambda s: s.manifest.created_at, reverse=True)


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _drop_time_zones(path: Path) -> None:
    """Rewrite a file so its timestamps carry no time zone (they already hold wall-clock time).

    A timestamp with a zone would be shown, and split into hours and days, in the
    time zone of whatever machine reads it. The file is rewritten batch by batch.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    schema = source.schema_arrow
    zoned = [f for f in schema if pa.types.is_timestamp(f.type) and f.type.tz is not None]
    if not zoned:
        return
    plain = pa.schema([pa.field(f.name, pa.timestamp(f.type.unit), f.nullable) if f in zoned else f for f in schema])
    rewritten = path.with_name(path.name + ".plain")
    with pq.ParquetWriter(rewritten, plain, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=200_000):
            writer.write_table(pa.Table.from_batches([batch]).cast(plain))
    source.close()
    rewritten.replace(path)


def _download(client: SourceClient, part: _Part, folder: Path) -> int:
    """Write one file (through a temporary name) and return its rows; an empty result leaves no file."""
    target, scratch = folder / part.file, folder / (part.file + ".tmp")
    with open(scratch, "wb") as out:
        for chunk in client.stream_parquet(part.sql):
            out.write(chunk)
    rows = _parquet_rows(scratch) if scratch.stat().st_size else 0
    if rows:
        _drop_time_zones(scratch)
        scratch.replace(target)
    else:
        scratch.unlink()
    return rows


def take_snapshot(
    settings: Settings,
    source_name: str,
    request: SnapshotRequest,
    *,
    refresh: bool = False,
    client_factory: Callable[[str, SourceConfig], SourceClient] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Snapshot:
    """Copy the tables of ``source_name`` for ``request``, or return the snapshot that already has them.

    Raises:
        SnapshotError: Unknown source, invalid request, a missing table or column,
            a table without rows, or a failure of the source.
    """
    say = progress or (lambda _message: None)
    sources = settings.config.sources
    if source_name not in sources:
        known = ", ".join(sorted(sources)) or "none; add a `sources` section to config.yaml"
        raise SnapshotError(f"Unknown source '{source_name}' (configured: {known}).")
    source = sources[source_name]
    _check(source_name, source, request)
    identifier = snapshot_id(source, request)
    final = snapshot_dir(settings, source_name, identifier)
    existing = load_manifest(final)
    if existing is not None and not refresh:
        return Snapshot(final, existing, reused=True)

    if client_factory is None:
        from jellybean.sources.clickhouse import connect as client_factory
    client = client_factory(source_name, source)
    work = final.with_name(final.name + _PARTIAL)
    if refresh:
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    for leftover in [*work.rglob("*.tmp"), *work.rglob("*.plain")]:
        leftover.unlink()
    tables: dict[str, TableManifest] = {}
    try:
        for dataset, config in source.tables.items():
            plan = plan_table(
                dataset, config, source.database, client.columns(source.database, config.table or dataset), request
            )
            folder = work / dataset
            folder.mkdir(exist_ok=True)
            say(f"{dataset}: copying {len(plan.parts)} file(s) from {source.database}.{plan.table}")
            rows = 0
            for number, part in enumerate(plan.parts, start=1):
                if (folder / part.file).exists():  # kept from an interrupted copy
                    rows += _parquet_rows(folder / part.file)
                    continue
                copied = _download(client, part, folder)
                rows += copied
                if len(plan.parts) > 1:
                    say(f"{dataset}: {part.file} {copied:,} rows ({number}/{len(plan.parts)})")
            files = sorted(folder.glob("*.parquet"))
            if rows == 0:
                raise SnapshotError(
                    f"Table {source.database}.{plan.table} has no rows for this window, sample and filter."
                )
            tables[dataset] = TableManifest(
                table=plan.table,
                rows=rows,
                files=len(files),
                bytes=sum(f.stat().st_size for f in files),
                time_column=config.time_column,
                entity_column=config.entity_column,
                windowed=plan.windowed,
                sampled=plan.sampled,
                query=plan.parts[0].sql,
            )
            say(f"{dataset}: {rows:,} rows in {len(files)} file(s)")
    finally:
        client.close()

    manifest = SnapshotManifest(
        source=source_name,
        type=source.type,
        database=source.database,
        id=identifier,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        start=_text(request.start) if request.start else None,
        end=_text(request.end) if request.end else None,
        sample=request.sample,
        tables=tables,
    )
    (work / MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    work.rename(final)
    return Snapshot(final, manifest)


def manifest_summary(manifest: SnapshotManifest) -> dict[str, Any]:
    """Compact form stored with a run (``spec.yaml``)."""
    return {
        "name": manifest.source,
        "type": manifest.type,
        "snapshot": manifest.id,
        "start": manifest.start,
        "end": manifest.end,
        "sample": manifest.sample,
        "created_at": manifest.created_at,
        "tables": {name: table.rows for name, table in manifest.tables.items()},
        "sampled_tables": sorted(name for name, table in manifest.tables.items() if table.sampled),
    }
