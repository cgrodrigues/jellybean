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

    Returns:
        The row count and one :class:`ColumnStats` per column.
    """
    schema = [(n, t) for n, t in describe(con, view) if columns is None or n in columns]
    aggregates = ["count(*)"]
    for name, dtype in schema:
        q = quote_ident(name)
        family = type_family(dtype)
        aggregates += [f"count({q})", f"count(DISTINCT {q})"]
        if family in ("integer", "float", "temporal", "string"):
            aggregates += [f"min({q})", f"max({q})"]
        else:
            aggregates += ["NULL", "NULL"]
        if family in ("integer", "float"):
            qs = ", ".join(str(x) for x in QUANTILES)
            aggregates += [f"quantile_cont({q}, [{qs}])", f"avg({q})", f"stddev_samp({q})"]
        else:
            aggregates += ["NULL", "NULL", "NULL"]
        aggregates.append(f"avg(length({q}))" if family == "string" else "NULL")
    row = con.execute(f"SELECT {', '.join(aggregates)} FROM {view}").fetchall()[0]
    rows = int(row[0])

    stats: list[ColumnStats] = []
    for i, (name, dtype) in enumerate(schema):
        base = 1 + i * 8
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

    for col in stats:
        q = quote_ident(col.name)
        if with_top and col.family in ("string", "integer", "boolean", "temporal") and col.non_null:
            col.top_values = [
                (r[0], int(r[1]))
                for r in con.execute(
                    f"SELECT {q}, count(*) AS n FROM {view} WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 5"
                ).fetchall()
            ]
        wanted = with_samples is True or (with_samples == "classify" and (col.family == "string" or col.distinct <= 2))
        if wanted and col.non_null:
            col.sample_values = [
                r[0]
                for r in con.execute(
                    f"SELECT v FROM (SELECT DISTINCT {q} AS v FROM {view} WHERE {q} IS NOT NULL) "
                    f"ORDER BY hash(v, {int(seed)}) LIMIT 5"
                ).fetchall()
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
