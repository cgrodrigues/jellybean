# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Column statistics and rule-based semantic typing, computed inside DuckDB.

These helpers are pure functions over an open connection and a view name. They
power several profiler tools (and later the feature engineer), so the rules live
in one place.

Semantic typing uses only the column *name*, its *statistics* and up to five
*sample values*. It never assumes a domain: name tokens are generic
(``id``, ``code``, ``number``...), and the rules are documented in
:func:`classify_column`.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import duckdb

from jellybean.skills._shared.catalog import quote_ident

_INT_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
}
_FLOAT_TYPES = {"FLOAT", "DOUBLE", "REAL"}
_ID_TOKENS = {"id", "key", "number", "num", "no", "uuid", "guid", "ref", "msisdn", "imei", "imsi", "iccid", "code"}
_GEO_TOKENS = {"lat", "lon", "lng", "latitude", "longitude", "geom", "geometry", "coord", "coords", "coordinates"}
_BOOL_VALUES = {"0", "1", "true", "false", "t", "f", "y", "n", "yes", "no", "sim", "nao", "não"}
QUANTILES = (0.01, 0.25, 0.5, 0.75, 0.99)


def type_family(duckdb_type: str) -> str:
    """Map a DuckDB type name to a coarse family."""
    t = duckdb_type.upper()
    if t in ("VARCHAR", "TEXT", "STRING") or t.startswith("VARCHAR"):
        return "string"
    if t == "BOOLEAN":
        return "boolean"
    if t in _INT_TYPES:
        return "integer"
    if t in _FLOAT_TYPES or t.startswith("DECIMAL"):
        return "float"
    if t.startswith(("TIMESTAMP", "DATE", "TIME")):
        return "temporal"
    return "other"


def name_tokens(column: str) -> set[str]:
    """Split ``customerId`` / ``customer_id`` / ``Customer ID`` into lowercase tokens."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", column)
    return {t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t}


@dataclass
class ColumnStats:
    """Statistics of one column."""

    name: str
    type: str
    family: str
    rows: int
    non_null: int
    distinct: int
    min: Any = None
    max: Any = None
    quantiles: dict[str, float] | None = None
    mean: float | None = None
    std: float | None = None
    avg_length: float | None = None
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    sample_values: list[Any] = field(default_factory=list)

    @property
    def null_share(self) -> float:
        return 0.0 if self.rows == 0 else 1 - self.non_null / self.rows

    @property
    def distinct_ratio(self) -> float:
        """Distinct values divided by non-null rows (1.0 = all different)."""
        return 0.0 if self.non_null == 0 else self.distinct / self.non_null

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["null_share"] = round(self.null_share, 5)
        data["distinct_ratio"] = round(self.distinct_ratio, 5)
        return data


def describe(con: duckdb.DuckDBPyConnection, view: str) -> list[tuple[str, str]]:
    """Return ``[(column, duckdb_type), ...]`` of a view (``view`` is already quoted)."""
    return [(r[0], r[1]) for r in con.execute(f"DESCRIBE SELECT * FROM {view}").fetchall()]


# Statistics of datasets and table artifacts, kept for the run that computed them: both
# are immutable during a run and the profiler asks for the same ones from many tools.
# Keys start with the run folder; the caches are emptied when they grow past their cap.
_BASE_CACHE: dict[tuple[str, str], tuple[int, list[ColumnStats]]] = {}
_EXTRA_CACHE: dict[tuple[str, str, str, str, int], list[Any]] = {}
_MAX_CACHED_SOURCES = 64
_MAX_CACHED_EXTRAS = 4096
# Large tables: distinct counts up to this estimate are exact in one statement (small
# hash sets); above it each column is counted by its own spillable GROUP BY.
_SMALL_DISTINCT = 100_000


def _cache_key(view: str) -> tuple[str, str] | None:
    """Cache key of ``view`` in the active run; ``None`` for temporary tables and outside a run."""
    from jellybean.runtime.context import maybe_current_run

    ctx = maybe_current_run()
    if ctx is None or not view.startswith('"'):
        return None
    return str(ctx.run_dir), view


def clear_stats_cache() -> None:
    """Forget cached statistics (tests, or a dataset rewritten during a run)."""
    _BASE_CACHE.clear()
    _EXTRA_CACHE.clear()


def _exact_limit() -> int:
    from jellybean.runtime.context import maybe_current_run
    from jellybean.settings import get_settings

    ctx = maybe_current_run()
    return (ctx.settings if ctx else get_settings()).config.profiling.exact_stats_max_rows


def _base_stats(con: duckdb.DuckDBPyConnection, view: str) -> tuple[int, list[ColumnStats]]:
    """Row count and per-column statistics (no top values or samples) of every column."""
    schema = describe(con, view)
    rows = int(con.execute(f"SELECT count(*) FROM {view}").fetchall()[0][0])
    # Exact distinct counts and quantiles keep every value in memory: fine for a few million
    # rows, not for hundreds of millions. Large tables get sketches first.
    large = rows > _exact_limit()
    aggregates = []
    for name, dtype in schema:
        q = quote_ident(name)
        family = type_family(dtype)
        aggregates += [f"count({q})", f"approx_count_distinct({q})" if large else f"count(DISTINCT {q})"]
        if family in ("integer", "float", "temporal", "string"):
            aggregates += [f"min({q})", f"max({q})"]
        else:
            aggregates += ["NULL", "NULL"]
        if family in ("integer", "float"):
            qs = ", ".join(str(x) for x in QUANTILES)
            quantile = "approx_quantile" if large else "quantile_cont"
            aggregates += [f"{quantile}({q}, [{qs}])", f"avg({q})", f"stddev_samp({q})"]
        else:
            aggregates += ["NULL", "NULL", "NULL"]
        aggregates.append(f"avg(length({q}))" if family == "string" else "NULL")
    row = con.execute(f"SELECT {', '.join(aggregates)} FROM {view}").fetchall()[0] if schema else []

    stats: list[ColumnStats] = []
    for i, (name, dtype) in enumerate(schema):
        base = i * 8
        non_null, distinct, mn, mx, qs, mean, std, avg_len = row[base : base + 8]
        stats.append(
            ColumnStats(
                name=name,
                type=dtype,
                family=type_family(dtype),
                rows=rows,
                non_null=int(non_null),
                distinct=int(distinct),
                min=mn,
                max=mx,
                quantiles=None if qs is None else {f"p{int(p * 100)}": v for p, v in zip(QUANTILES, qs, strict=True)},
                mean=None if mean is None else float(mean),
                std=None if std is None else float(std),
                avg_length=None if avg_len is None else float(avg_len),
            )
        )
    if large:
        _exact_distinct(con, view, stats)
    return rows, stats


def _exact_distinct(con: duckdb.DuckDBPyConnection, view: str, stats: list[ColumnStats]) -> None:
    """Replace the estimated distinct counts by exact ones without holding every value in memory.

    Uniqueness decides keys, grain and join cardinality, so the counts must be
    exact. Low-cardinality columns are counted together (their hash sets are
    small); each high-cardinality column gets its own ``GROUP BY``, which reads
    one column and spills to disk when it does not fit.
    """
    small = [c for c in stats if c.distinct <= _SMALL_DISTINCT]
    for start in range(0, len(small), 50):
        chunk = small[start : start + 50]
        exprs = ", ".join(f"count(DISTINCT {quote_ident(c.name)})" for c in chunk)
        for col, n in zip(chunk, con.execute(f"SELECT {exprs} FROM {view}").fetchall()[0], strict=True):
            col.distinct = int(n)
    for col in stats:
        if col.distinct > _SMALL_DISTINCT:
            q = quote_ident(col.name)
            col.distinct = int(
                con.execute(
                    f"SELECT count(*) FROM (SELECT {q} FROM {view} WHERE {q} IS NOT NULL GROUP BY 1)"
                ).fetchall()[0][0]
            )


def count_duplicate_rows(con: duckdb.DuckDBPyConnection, view: str) -> int:
    """Exact duplicate rows of ``view`` (rows minus distinct rows), cached for the run.

    A full-width ``SELECT DISTINCT *`` keeps every row in a hash table. Here the rows
    are grouped by a 64-bit hash of the row first (8 bytes per group); only rows whose
    hash repeats, usually none or few, are compared in full. The result is exact: equal
    rows always share a hash, and rows that merely collide are told apart in the second step.
    """
    key = _cache_key(view)
    extra_key = (*key, "", "duplicate_rows", 0) if key else None
    if extra_key is not None and extra_key in _EXTRA_CACHE:
        return int(_EXTRA_CACHE[extra_key][0])
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE _jb_dup_hashes AS SELECT hash(_t) AS h FROM {view} AS _t "
        "GROUP BY 1 HAVING count(*) > 1"
    )
    try:
        duplicates = 0
        if con.execute("SELECT count(*) FROM _jb_dup_hashes").fetchall()[0][0]:
            repeated = f"(SELECT * FROM {view} AS _t WHERE hash(_t) IN (SELECT h FROM _jb_dup_hashes))"
            duplicates = con.execute(
                f"SELECT (SELECT count(*) FROM {repeated}) - (SELECT count(*) FROM (SELECT DISTINCT * FROM {repeated}))"
            ).fetchall()[0][0]
    finally:
        con.execute("DROP TABLE IF EXISTS _jb_dup_hashes")
    if extra_key is not None:
        _EXTRA_CACHE[extra_key] = [int(duplicates)]
    return int(duplicates)


def compute_column_stats(
    con: duckdb.DuckDBPyConnection,
    view: str,
    seed: int = 0,
    with_top: bool = True,
    with_samples: bool | Literal["classify"] = True,
    columns: list[str] | None = None,
) -> tuple[int, list[ColumnStats]]:
    """Compute statistics for every column of ``view`` in as few scans as possible.

    ``with_samples="classify"`` samples only the columns whose classification reads
    sample values (strings and two-valued columns): one scan fewer per other column.

    Statistics of datasets and table artifacts are cached for the run (they do not
    change), so the tools of one profiling session share them. Above
    ``profiling.exact_stats_max_rows`` rows the quantiles are approximate
    (``approx_quantile``) and the distinct counts, still exact, are computed
    column by column so they can spill to disk.

    Returns:
        The row count and one :class:`ColumnStats` per column.
    """
    key = _cache_key(view)
    cached = _BASE_CACHE.get(key) if key else None
    if cached is None:
        cached = _base_stats(con, view)
        if key:
            if len(_BASE_CACHE) >= _MAX_CACHED_SOURCES:
                _BASE_CACHE.clear()
            _BASE_CACHE[key] = cached
    rows = cached[0]
    # Callers may edit what they get: hand out copies, never the cached objects.
    stats = [copy.copy(c) for c in cached[1] if columns is None or c.name in columns]
    for col in stats:
        col.top_values, col.sample_values = [], []

    def extra(col: ColumnStats, kind: str, query: str) -> list[Any]:
        extra_key = (*key, col.name, kind, int(seed)) if key else None
        if extra_key is not None and extra_key in _EXTRA_CACHE:
            return list(_EXTRA_CACHE[extra_key])
        values = con.execute(query).fetchall()
        if extra_key is not None:
            if len(_EXTRA_CACHE) >= _MAX_CACHED_EXTRAS:
                _EXTRA_CACHE.clear()
            _EXTRA_CACHE[extra_key] = values
        return list(values)

    for col in stats:
        q = quote_ident(col.name)
        if with_top and col.family in ("string", "integer", "boolean", "temporal") and col.non_null:
            col.top_values = [
                (r[0], int(r[1]))
                for r in extra(
                    col,
                    "top",
                    f"SELECT {q}, count(*) AS n FROM {view} WHERE {q} IS NOT NULL "
                    "GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 5",
                )
            ]
        wanted = with_samples is True or (with_samples == "classify" and (col.family == "string" or col.distinct <= 2))
        if wanted and col.non_null:
            col.sample_values = [
                r[0]
                for r in extra(
                    col,
                    "sample",
                    f"SELECT v FROM (SELECT DISTINCT {q} AS v FROM {view} WHERE {q} IS NOT NULL) "
                    f"ORDER BY hash(v, {int(seed)}) LIMIT 5",
                )
            ]
    return rows, stats


@dataclass(frozen=True)
class SemanticGuess:
    semantic_type: str
    confidence: float
    reason: str


_DATE_PATTERNS = (
    r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?",  # ISO date / datetime
    r"^\d{2}/\d{2}/\d{4}( \d{2}:\d{2}(:\d{2})?)?$",  # dd/mm/yyyy [hh:mm[:ss]]
    r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(T?\d{6})?$",  # yyyymmdd[hhmmss]
)


_CODE_RE = re.compile(r"^\+?[0-9][0-9 \-]{5,}$|^[A-Za-z]{0,4}[0-9]{5,}$")


def _looks_like_code(values: list[Any]) -> bool:
    """True when every sample is a long number or a short prefix plus digits (C0001234, +351...)."""
    strings = [str(v).strip() for v in values if v is not None]
    return bool(strings) and all(_CODE_RE.match(s) for s in strings)


def _looks_temporal(values: list[Any]) -> bool:
    """True when every sample value looks like a date or datetime."""
    strings = [str(v).strip() for v in values if v is not None]
    return bool(strings) and all(any(re.match(p, s) for p in _DATE_PATTERNS) for s in strings)


def classify_column(col: ColumnStats) -> SemanticGuess:
    """Classify a column into one of the allowed semantic types.

    Rules, in order:
      1. temporal storage type -> ``timestamp``; boolean storage -> ``boolean``.
      2. at most two distinct values from a yes/no vocabulary -> ``boolean``.
      3. latitude/longitude/geometry name tokens -> ``geo``.
      4. ``code`` in the name with at most 1000 distinct values -> ``category``.
      5. identifier name tokens (id, key, number, uuid...) -> ``identifier``.
      6. strings whose samples all look like dates -> ``timestamp``.
      7. long, mostly distinct strings -> ``free_text``.
      8. strings whose samples are long numbers / codes (>= 20 distinct) -> ``identifier``.
      9. strings with low cardinality -> ``category``; other strings -> ``identifier``.
      10. integers with at most 20 distinct values -> ``category``; other numbers -> ``measure``.
    """
    tokens = name_tokens(col.name)
    if col.non_null == 0:
        return SemanticGuess("unknown", 0.3, "column is entirely null")
    if col.family == "temporal":
        return SemanticGuess("timestamp", 0.95, f"storage type {col.type}")
    if col.family == "boolean":
        return SemanticGuess("boolean", 0.95, "boolean storage type")
    sample_text = {str(v).strip().lower() for v, _ in col.top_values} | {
        str(v).strip().lower() for v in col.sample_values
    }
    if col.distinct <= 2 and sample_text and sample_text <= _BOOL_VALUES:
        return SemanticGuess("boolean", 0.85, "two values from a yes/no vocabulary")
    if tokens & _GEO_TOKENS:
        return SemanticGuess("geo", 0.8, "geographic name token")
    if "code" in tokens and col.distinct <= 1000:
        return SemanticGuess("category", 0.7, "'code' name token with low cardinality")
    if tokens & _ID_TOKENS:
        return SemanticGuess("identifier", 0.85, f"identifier name token {sorted(tokens & _ID_TOKENS)}")
    if col.family == "string" and _looks_temporal(col.sample_values):
        return SemanticGuess("timestamp", 0.7, "string samples look like dates")
    if col.family == "string":
        if col.distinct >= 20 and _looks_like_code(col.sample_values):
            return SemanticGuess("identifier", 0.75, "values look like numbers or codes")
        if (col.avg_length or 0) > 40 and col.distinct_ratio > 0.3:
            return SemanticGuess("free_text", 0.75, "long and mostly distinct text")
        if col.distinct <= 1000 or col.distinct_ratio <= 0.05:
            return SemanticGuess("category", 0.8, f"{col.distinct} distinct values")
        return SemanticGuess("identifier", 0.6, f"high-cardinality string (ratio {col.distinct_ratio:.2f})")
    if col.family == "integer" and col.distinct <= 20:
        return SemanticGuess("category", 0.6, f"integer with {col.distinct} distinct values")
    if col.family in ("integer", "float"):
        return SemanticGuess("measure", 0.8, f"numeric ({col.family})")
    return SemanticGuess("unknown", 0.3, f"unsupported type {col.type}")
