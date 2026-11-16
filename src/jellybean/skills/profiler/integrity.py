"""Deterministic integrity checks of the profiler (``requirements_2.md`` section 10.2).

Rules that can be verified without any model, computed inside DuckDB:

* :func:`discover_arithmetic_invariants`: relations between measures that hold
  on (almost) every row (``a + b = c``, ``a = b``, ``a >= 0``), with the
  violations and where they concentrate. A violation that covers a whole
  segment (one category value on a few dates) is reported apart from scattered
  violations: the first is a systematic error, the second isolated records.
* :func:`check_referential_integrity`: orphan references, self-references included.
* :func:`find_duplicate_keys`: duplicated values of a key that should be unique,
  in rows that are otherwise different (exact duplicates are a separate issue).
* :func:`classify_nulls`: whether the nulls of a column are *structural*
  (determined by another column, e.g. a type) or errors.
* :func:`infer_sign_convention`: sign of a measure per category value, and
  sign inversions concentrated in time.
* :func:`discover_self_references` and :func:`check_reciprocity`: columns that
  refer to other columns of the same dataset, and rows expected to come in
  pairs (``A -> B`` matched by ``B -> A``).

Nothing here knows a domain: columns are chosen from their statistics and
semantic types. Each issue carries a ``predicate``: a SQL condition selecting
its rows when the dataset is read with the alias ``x`` (``SELECT ... FROM
<dataset> x WHERE <predicate>``), so later steps and the evaluation can find
them. Every check runs on the distinct rows of the dataset, so exact duplicate
rows (reported by ``profile_table``) do not inflate the counts.
Issues use the profiler's format, so ``compile_quality_issues`` merges them.

Resource use. The distinct rows are materialised **once** per call in a temporary
table (:func:`_working_rows`) and every later statement reads that table; it is
never an inline ``SELECT DISTINCT *`` recomputed per statement. Datasets above
``profiling.distinct_max_rows`` are read as they are (no copy, columns pruned by
the scan), and the result says that exact duplicates were not removed. Where a
problem concentrates is computed in one scan for all category columns
(:func:`_cells`), not one scan per column.
"""

from __future__ import annotations

import contextlib
import itertools
from typing import Annotated, Any

import duckdb
from pydantic import Field

from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.skills._shared.catalog import SourceNotFoundError, connect, quote_ident, resolve_source
from jellybean.skills._shared.column_stats import ColumnStats, classify_column, compute_column_stats
from jellybean.skills._shared.keys import is_valid_strategy, strategy_sql
from jellybean.skills.profiler._common import _issue, _masker, _settings, _store

# Columns used to describe where problems concentrate: low-cardinality categories.
_MAX_SEGMENT_VALUES = 50
# Measures considered by discover_arithmetic_invariants (by number of non-null values).
_MAX_MEASURES = 12
# A segment (category value x date) is systematic when most of its rows violate.
_SEGMENT_RATE = 0.5
_MIN_SEGMENT_ROWS = 10
# Temporary tables of one call (dataset views never start with "_").
_ROWS = "_jb_rows"
_REF_ROWS = "_jb_ref_rows"
_CELLS = "_jb_cells"
_SAMPLE = "_jb_sample"
# A candidate relation screened on a sample is kept when it holds this close to `min_share`
# (two standard errors of a share near 0.95 on a few hundred rows).
_SCREEN_MARGIN = 0.03
# infer_sign_convention: a "category" with more values than this is not one.
_MAX_SIGN_CATEGORIES = 200
# discover_self_references: identifier columns compared (the ones with most distinct values).
_MAX_KEY_COLUMNS = 16


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _working_rows(con: duckdb.DuckDBPyConnection, view: str, table: str = _ROWS) -> tuple[str, list[str]]:
    """Relation every statement of a check reads, and the warnings to return with the result.

    Up to ``profiling.distinct_max_rows`` rows: a temporary table with the distinct
    rows, built once (dropped by :func:`_close`). Above it: the dataset itself, so
    nothing is copied and each statement reads only its columns; exact duplicate
    rows are then counted, which the warning says.
    """
    limit = _settings().config.profiling.distinct_max_rows
    total = con.execute(f"SELECT count(*) FROM {view}").fetchall()[0][0]
    if total > limit:
        return view, [
            f"Large dataset ({total:,} rows, above profiling.distinct_max_rows={limit:,}): exact duplicate rows "
            "were not removed before this check, so they count in its figures."
        ]
    con.execute(f"CREATE OR REPLACE TEMP TABLE {table} AS SELECT DISTINCT * FROM {view}")
    return table, []


def _close(con: duckdb.DuckDBPyConnection) -> None:
    """Drop the temporary tables of a check and close its connection."""
    for table in (_ROWS, _REF_ROWS, _CELLS, _SAMPLE):
        with contextlib.suppress(duckdb.Error):
            con.execute(f"DROP TABLE IF EXISTS {table}")
    con.close()


def _norm(column: str) -> str:
    """Category value with spelling variants merged (case, surrounding spaces)."""
    return f"lower(trim(CAST({quote_ident(column)} AS VARCHAR)))"


def _segment_columns(stats: list[ColumnStats], exclude: set[str] = frozenset()) -> list[str]:
    """Low-cardinality, non-sensitive category columns used to locate problems."""
    masker = _masker()
    out = []
    for c in stats:
        if c.name in exclude or c.family not in ("string", "integer", "boolean") or not c.distinct >= 2:
            continue
        if c.distinct > _MAX_SEGMENT_VALUES or masker.is_sensitive_name(c.name):
            continue
        if c.family == "integer" and classify_column(c).semantic_type != "category":
            continue
        out.append(c.name)
    return out


def _date_column(stats: list[ColumnStats]) -> str | None:
    """Column giving the day of each row: a ``date`` column, else the first timestamp column."""
    by_name = {c.name: c for c in stats}
    if "date" in by_name and by_name["date"].family == "temporal":
        return "date"
    temporal = [c for c in stats if c.family == "temporal"]
    return temporal[0].name if temporal else None


def _date_expr(stats: list[ColumnStats]) -> str | None:
    """Day of each row, as a SQL expression."""
    column = _date_column(stats)
    return f"{quote_ident(column)}::DATE" if column else None


def _cells(
    con: duckdb.DuckDBPyConnection, rel: str, flag: str, scope: str, columns: list[str], date: str | None
) -> None:
    """Count flagged and total rows per (category column, value, day), in one scan of ``rel``.

    Fills the temporary table ``_jb_cells(col, v, d, f, n)``: ``col`` is the index of
    the category column in ``columns`` (``-1``: no column, i.e. the totals per day),
    ``v`` its normalised value, ``d`` the day (null without ``date``), ``f`` the rows
    matching ``flag`` and ``n`` all rows, both within ``scope``. Each row is counted
    once per category column while it streams by, so the cost is one scan whatever
    the number of columns.
    """
    values = ", ".join(["CAST(NULL AS VARCHAR)", *(_norm(c) for c in columns)])
    index = ", ".join(str(i) for i in range(-1, len(columns)))
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE {_CELLS} AS "
        "SELECT _jb_col AS col, _jb_v AS v, _jb_d AS d, count(*) FILTER (WHERE _jb_f) AS f, count(*) AS n FROM ("
        f"SELECT unnest([{values}]) AS _jb_v, unnest([{index}]) AS _jb_col, {date or 'NULL'} AS _jb_d, "
        f"coalesce(({flag}), false) AS _jb_f FROM {rel} WHERE {scope}) GROUP BY ALL"
    )


def _concentration_of(con: duckdb.DuckDBPyConnection, columns: list[str], date: str | None) -> dict[str, Any]:
    """Where the flagged rows of the last :func:`_cells` concentrate.

    Returns, for the best category column, the value holding the largest share of
    flagged rows and its lift over the base share, plus the top dates.
    """
    total, base_rows = con.execute(
        f"SELECT coalesce(sum(f), 0), coalesce(sum(n), 0) FROM {_CELLS} WHERE col = -1"
    ).fetchall()[0]
    if not total:
        return {}
    tops = con.execute(
        "SELECT col, v, f, n FROM (SELECT col, v, sum(f) AS f, sum(n) AS n, "
        "row_number() OVER (PARTITION BY col ORDER BY sum(f) DESC, v) AS r "
        f"FROM {_CELLS} WHERE col >= 0 GROUP BY col, v) WHERE r = 1 ORDER BY col"
    ).fetchall()
    best: dict[str, Any] = {}
    for col, value, flagged, n in tops:
        if not flagged:
            continue
        share, base = int(flagged) / int(total), int(n) / int(base_rows)
        lift = share / base if base else 0.0
        if share >= 0.5 and lift > best.get("lift", 1.5):
            best = {
                "column": columns[col],
                "value": value,
                "share": round(share, 4),
                "base_share": round(base, 4),
                "lift": round(lift, 1),
            }
    out: dict[str, Any] = {"rows": int(total)}
    if best:
        out["segment"] = best
    if date:
        dates = con.execute(f"SELECT d, f FROM {_CELLS} WHERE col = -1 AND f > 0 ORDER BY f DESC, d LIMIT 5").fetchall()
        n_dates = con.execute(f"SELECT count(d) FROM {_CELLS} WHERE col = -1 AND f > 0").fetchall()[0][0]
        out["dates"] = {"distinct": int(n_dates), "top": [[str(d), int(n)] for d, n in dates]}
    return out


def _concentration(
    con: duckdb.DuckDBPyConnection, rel: str, flag: str, scope: str, columns: list[str], date: str | None
) -> dict[str, Any]:
    """Where the rows matching ``flag`` concentrate (within the rows of ``scope``): one scan."""
    _cells(con, rel, flag, scope, columns, date)
    return _concentration_of(con, columns, date)


def _systematic_of(con: duckdb.DuckDBPyConnection, columns: list[str], date: str | None) -> dict[str, Any] | None:
    """The (category value x dates) block where most rows are flagged, if it explains many flagged rows.

    Reads the last :func:`_cells`. A cell (value, day) is systematic when at least
    half of its rows are flagged and it has at least ``_MIN_SEGMENT_ROWS`` flagged
    rows. The column whose systematic cells cover the most flagged rows wins.
    """
    if not date:
        return None
    total = int(con.execute(f"SELECT coalesce(sum(f), 0) FROM {_CELLS} WHERE col = -1").fetchall()[0][0])
    rows = con.execute(
        f"SELECT col, v, d, f, n FROM {_CELLS} WHERE col >= 0 AND d IS NOT NULL "
        f"AND f >= {_MIN_SEGMENT_ROWS} AND f >= {_SEGMENT_RATE} * n ORDER BY col, v, d"
    ).fetchall()
    by_column: dict[int, list[tuple[Any, Any, int, int]]] = {}
    for col, v, d, f, n in rows:
        by_column.setdefault(col, []).append((v, d, int(f), int(n)))
    best = None
    for col in sorted(by_column):
        cells = by_column[col]
        covered = sum(c[2] for c in cells)
        if best is None or covered > best["rows"]:
            values = sorted({c[0] for c in cells}, key=str)
            dates = sorted({str(c[1]) for c in cells})
            best = {"column": columns[col], "values": values, "dates": dates, "rows": int(covered), "cells": cells}
    if best is None or best["rows"] < max(_MIN_SEGMENT_ROWS * 3, 0.2 * total):
        return None
    cells = best.pop("cells")
    # SQL predicate of the systematic rows, so other tools (and the evaluation) can use it.
    pairs = " OR ".join(f"({_norm(best['column'])} = {_lit(v)} AND {date} = DATE '{d}')" for v, d, _, _ in cells)
    best["predicate"] = f"({pairs})"
    best["share_of_violations"] = round(best["rows"] / total, 4) if total else 0.0
    return best


def _resolve(dataset: str) -> tuple[Any, duckdb.DuckDBPyConnection] | ToolResult:
    try:
        source = resolve_source(dataset)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    return source, connect()


def _null_groups(
    con: duckdb.DuckDBPyConnection, rel: str, columns: list[str], determinant: str
) -> dict[str, list[tuple[Any, int, int]]]:
    """Rows and nulls of every column in ``columns`` per value of ``determinant``.

    One scan for all the columns (in batches of 100), instead of one per column.
    Returns ``{column: [(value, rows, nulls), ...]}``.
    """
    out: dict[str, list[tuple[Any, int, int]]] = {c: [] for c in columns}
    for start in range(0, len(columns), 100):
        chunk = columns[start : start + 100]
        counts = ", ".join(f"count(*) - count({quote_ident(c)})" for c in chunk)
        for row in con.execute(f"SELECT {_norm(determinant)} v, count(*) n, {counts} FROM {rel} GROUP BY 1").fetchall():
            for i, c in enumerate(chunk):
                out[c].append((row[0], int(row[1]), int(row[2 + i])))
    return out


def null_structure(
    con: duckdb.DuckDBPyConnection,
    rel: str,
    column: str,
    determinants: list[str],
    groups: dict[str, dict[str, list[tuple[Any, int, int]]]] | None = None,
) -> dict[str, Any] | None:
    """Explain the nulls of ``column`` by a category column, if one determines them.

    Groups are the (normalised) values of the determinant: a group is "null"
    when at least half of its rows are null, "filled" otherwise. Nulls inside
    filled groups are unexpected (errors). The determinant explaining the most
    nulls with the fewest exceptions wins.

    ``groups`` are precomputed counts (``{determinant: _null_groups(...)}``) when
    many columns are classified at once; without them each determinant is one query.

    Returns ``None`` when no determinant puts at least 95% of the nulls in null groups.
    """
    best = None
    for det in determinants:
        if det == column:
            continue
        counted = groups[det][column] if groups is not None else _null_groups(con, rel, [column], det)[column]
        nulls = sum(g[2] for g in counted)
        if not nulls or any(g[0] is None for g in counted):
            continue
        null_groups = [g for g in counted if g[2] >= 0.5 * g[1]]
        filled_groups = [g for g in counted if g[2] < 0.5 * g[1]]
        if not null_groups or not filled_groups:
            continue
        explained = sum(g[2] for g in null_groups)
        unexpected_nulls = nulls - explained
        unexpected_values = sum(g[1] - g[2] for g in null_groups)
        if explained < 0.95 * nulls:
            continue
        score = (unexpected_nulls + unexpected_values, len(counted))
        if best is None or score < best["_score"]:
            best = {
                "_score": score,
                "determined_by": det,
                "null_when": sorted(str(g[0]) for g in null_groups),
                "filled_when": sorted(str(g[0]) for g in filled_groups),
                "nulls": int(nulls),
                "unexpected_nulls": int(unexpected_nulls),
                "unexpected_nulls_in": sorted(
                    ([str(g[0]), int(g[2])] for g in filled_groups if g[2]), key=lambda x: -x[1]
                ),
                "unexpected_values": int(unexpected_values),
                "_applicable": int(sum(g[1] for g in filled_groups)),
            }
    if best is None:
        return None
    best.pop("_score")
    best["applicable_rows"] = best.pop("_applicable")
    best["classification"] = "structural" if best["unexpected_nulls"] == 0 else "structural_with_errors"
    return best


def _lit(value: Any) -> str:
    """SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _short(values: list[Any], n: int = 6) -> list[Any] | dict[str, Any]:
    return values if len(values) <= n else {"first": values[:n], "count": len(values)}


# --------------------------------------------------------------------------- #
# discover_arithmetic_invariants                                              #
# --------------------------------------------------------------------------- #
def _count_candidates(
    con: duckdb.DuckDBPyConnection, rel: str, candidates: list[dict[str, Any]], batch: int
) -> list[tuple[int, int]]:
    """Rows where each candidate relation applies and where it holds, ``batch`` candidates per statement."""
    out: list[tuple[int, int]] = []
    for start in range(0, len(candidates), batch):
        chunk = candidates[start : start + batch]
        aggregates = []
        for cand in chunk:
            aggregates += [
                f"count(*) FILTER (WHERE {cand['present']})",
                f"count(*) FILTER (WHERE {cand['present']} AND {cand['holds']})",
            ]
        row = con.execute(f"SELECT {', '.join(aggregates)} FROM {rel}").fetchall()[0]
        out += [(int(row[2 * i]), int(row[2 * i + 1])) for i in range(len(chunk))]
    return out


@tool
def discover_arithmetic_invariants(
    dataset: str,
    columns: list[str] | None = None,
    min_share: Annotated[float, Field(gt=0.5, le=1)] = 0.95,
    seed: int | None = None,
) -> ToolResult:
    """Find arithmetic relations between measures that hold on almost every row, and their violations.

    Tests `a + b = c`, `a = b` and `a >= 0` on the measure columns and keeps those
    that hold on at least `min_share` of the rows where the columns are present.
    For each kept relation with violations it reports where they concentrate
    (category value, dates) and separates a systematic block (one category value
    on some dates, most of its rows violating; with the sign flip that explains
    it, if any) from scattered violations. Violations become quality issues; they
    are never findings for the detector. On large datasets the candidate relations
    are screened on a sample first; the reported counts are always exact.

    Args:
        dataset: Dataset name or table artifact id.
        columns: Measures to test (default: every numeric measure, at most 12).
        min_share: Minimum share of rows where a relation must hold.
        seed: Seed of the screening sample (defaults to the run seed).
    """
    opened = _resolve(dataset)
    if isinstance(opened, ToolResult):
        return opened
    source, con = opened
    try:
        rel, notes = _working_rows(con, source.view)
        _, stats = compute_column_stats(con, rel, with_top=False, with_samples="classify")
        by_name = {c.name: c for c in stats}
        if columns:
            unknown = [c for c in columns if c not in by_name or by_name[c].family not in ("integer", "float")]
            if unknown:
                return ToolResult.error(f"Not numeric columns of '{source.name}': {unknown}")
            measures = columns
        else:
            measures = [
                c.name
                for c in sorted(stats, key=lambda c: -c.non_null)
                if c.family in ("integer", "float") and classify_column(c).semantic_type == "measure"
            ][:_MAX_MEASURES]
        rows = con.execute(f"SELECT count(*) FROM {rel}").fetchall()[0][0]
        exact = all(by_name[m].family == "integer" for m in measures)
        tol = "0" if exact else "0.0051"

        def present(*cols: str) -> str:
            return " AND ".join(f"{quote_ident(c)} IS NOT NULL" for c in cols)

        candidates: list[dict[str, Any]] = []
        for a, b, c in itertools.combinations(measures, 3):
            for x, y, z in ((a, b, c), (a, c, b), (b, c, a)):
                expr = f"abs({quote_ident(x)} + {quote_ident(y)} - {quote_ident(z)}) <= {tol}"
                candidates.append(
                    {
                        "expression": f"{x} + {y} = {z}",
                        "terms": [x, y, z],
                        "kind": "sum",
                        "holds": expr,
                        "present": present(x, y, z),
                    }
                )
        for a, b in itertools.combinations(measures, 2):
            expr = f"abs({quote_ident(a)} - {quote_ident(b)}) <= {tol}"
            candidates.append(
                {"expression": f"{a} = {b}", "terms": [a, b], "kind": "equal", "holds": expr, "present": present(a, b)}
            )
        for a in measures:
            candidates.append(
                {
                    "expression": f"{a} >= 0",
                    "terms": [a],
                    "kind": "non_negative",
                    "holds": f"{quote_ident(a)} >= 0",
                    "present": present(a),
                }
            )
        # Screen the candidates on a sample when the table is large: most of them fail
        # clearly, and only the survivors are counted exactly on every row.
        profiling = _settings().config.profiling
        if rows > profiling.sample_rows and candidates:
            keep = max(1, round(1_000_000 * profiling.sample_rows / rows))
            cols = ", ".join(quote_ident(m) for m in measures)
            con.execute(
                f"CREATE OR REPLACE TEMP TABLE {_SAMPLE} AS SELECT {cols} FROM {rel} AS _t "
                f"WHERE hash(_t, {int(seed or 0)}) % 1000000 < {keep}"
            )
            sampled = _count_candidates(con, _SAMPLE, candidates, profiling.candidate_batch)
            candidates = [
                c
                for c, (n, ok) in zip(candidates, sampled, strict=True)
                if n and ok >= (min_share - _SCREEN_MARGIN) * n
            ]
        counts = _count_candidates(con, rel, candidates, profiling.candidate_batch)
        held = []
        min_rows = max(20, int(0.001 * rows))
        for cand, (n, ok) in zip(candidates, counts, strict=True):
            if n < min_rows or ok < min_share * n:
                continue
            if cand["kind"] == "equal" and ok == n and by_name[cand["terms"][0]].distinct == 1:
                continue  # two constant columns: not informative
            held.append({**cand, "rows_checked": int(n), "holds_share": round(ok / n, 6), "violations": int(n - ok)})
        # A relation implied by others is redundant: with x + y = z and y = w, x + w = z
        # only repeats them on fewer rows (and duplicates their violations).
        equal = {frozenset(i["terms"]) for i in held if i["kind"] == "equal"}
        sums = {(frozenset(i["terms"][:2]), i["terms"][2]): i for i in held if i["kind"] == "sum"}

        def implied(inv: dict[str, Any]) -> bool:
            x, y, z = inv["terms"]
            for keep, swap in ((x, y), (y, x)):
                for pair in equal:
                    if swap in pair:
                        (other,) = pair - {swap}
                        twin = sums.get((frozenset((keep, other)), z))
                        if twin is not None and twin is not inv and twin["rows_checked"] > inv["rows_checked"]:
                            return True
            return False

        held = [i for i in held if not (i["kind"] == "sum" and implied(i))]
        segments = _segment_columns(stats)
        date = _date_expr(stats)
        issues: list[dict[str, Any]] = []
        for inv in held:
            if not inv["violations"] or (inv["kind"] == "non_negative" and inv["holds_share"] == 1):
                continue
            flag = f"NOT ({inv['holds']})"
            scope = inv["present"]
            # One scan gives both where the violations concentrate and the systematic block.
            _cells(con, rel, flag, scope, segments, date)
            inv["concentration"] = _concentration_of(con, segments, date)
            systematic = _systematic_of(con, segments, date) if inv["kind"] != "non_negative" else None
            scattered = inv["violations"]
            if systematic:
                if inv["kind"] == "sum":
                    x, y, z = inv["terms"]
                    flip_y = f"abs({quote_ident(x)} - {quote_ident(y)} - {quote_ident(z)}) <= {tol}"
                    flip_x = f"abs({quote_ident(y)} - {quote_ident(x)} - {quote_ident(z)}) <= {tol}"
                    n_y, n_x = con.execute(
                        f"SELECT count(*) FILTER (WHERE {flip_y}), count(*) FILTER (WHERE {flip_x}) FROM {rel} "
                        f"WHERE {scope} AND {flag} AND {systematic['predicate']}"
                    ).fetchall()[0]
                    flips = {y: n_y, x: n_x}
                    column, n_flip = max(flips.items(), key=lambda kv: kv[1])
                    if n_flip >= 0.9 * systematic["rows"]:
                        systematic["sign_flipped"] = {"column": column, "share": round(n_flip / systematic["rows"], 4)}
                scattered -= systematic["rows"]
                inv["systematic"] = {k: v for k, v in systematic.items() if k != "predicate"}
                issues.append(
                    _issue(
                        source.name,
                        (systematic.get("sign_flipped") or {}).get("column") or inv["terms"][-1],
                        "invariant_violation_systematic",
                        "high",
                        invariant=inv["expression"],
                        rows=systematic["rows"],
                        segment={"column": systematic["column"], "values": systematic["values"]},
                        dates=systematic["dates"],
                        sign_flipped=systematic.get("sign_flipped"),
                        date_column=date.split('"')[1] if date and '"' in date else None,
                        violation=flag,
                        predicate=f"{flag} AND {systematic['predicate']}",
                    )
                )
            inv["scattered_violations"] = int(scattered)
            if scattered > 0:
                not_systematic = f" AND NOT {systematic['predicate']}" if systematic else ""
                issues.append(
                    _issue(
                        source.name,
                        inv["terms"][-1],
                        "invariant_violation",
                        "medium",
                        invariant=inv["expression"],
                        rows=int(scattered),
                        share=round(scattered / inv["rows_checked"], 6),
                        concentration=inv["concentration"].get("segment"),
                        predicate=f"{scope} AND {flag}{not_systematic}",
                    )
                )
    finally:
        _close(con)
    for inv in held:
        inv.pop("holds"), inv.pop("present")
    record = _store().save_json(
        f"invariants_{source.name}",
        {"dataset": source.name, "rows": rows, "measures": measures, "invariants": held, "issues": issues},
        parents=[source.name],
        params={"min_share": min_share},
    )
    relations = [i for i in held if i["kind"] != "non_negative"]
    return ToolResult.ok(
        {
            "dataset": source.name,
            "measures": measures,
            "invariants": [
                {
                    "expression": i["expression"],
                    "holds_share": i["holds_share"],
                    "rows_checked": i["rows_checked"],
                    "violations": i["violations"],
                    **({"systematic": i["systematic"]} if i.get("systematic") else {}),
                    **({"scattered_violations": i["scattered_violations"]} if "scattered_violations" in i else {}),
                    **(
                        {"concentration": i["concentration"].get("segment")}
                        if i.get("concentration", {}).get("segment")
                        else {}
                    ),
                }
                for i in relations[:8]
            ],
            "non_negative": [i["terms"][0] for i in held if i["kind"] == "non_negative"],
            "issues": [
                {
                    "kind": i["kind"],
                    "column": i["column"],
                    "rows": i["detail"]["rows"],
                    "invariant": i["detail"]["invariant"],
                }
                for i in issues
            ],
        },
        artifact_id=record.id,
        warnings=[*notes, *([] if relations else ["No relation between measures holds on most rows."])],
    )


# --------------------------------------------------------------------------- #
# check_referential_integrity                                                 #
# --------------------------------------------------------------------------- #
@tool
def check_referential_integrity(
    dataset: str, column: str, ref_dataset: str, ref_column: str, normalization: str = "none"
) -> ToolResult:
    """Check that every non-null value of `column` exists in `ref_dataset.ref_column` (self-references allowed).

    Reports orphan references (values with no target), where the orphans
    concentrate (category value), the share of null references and whether those
    nulls are structural, and the uniqueness of the target column.

    Args:
        dataset: Dataset holding the reference.
        column: Referencing column.
        ref_dataset: Dataset holding the target (may be the same dataset).
        ref_column: Target column.
        normalization: Key normalisation applied to both sides (as in test_join).
    """
    if not is_valid_strategy(normalization):
        return ToolResult.error(f"Unknown normalization '{normalization}'")
    try:
        src, ref = resolve_source(dataset), resolve_source(ref_dataset)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        rel, notes = _working_rows(con, src.view)
        ref_rel = rel if ref.name == src.name else _working_rows(con, ref.view, _REF_ROWS)[0]
        _, stats = compute_column_stats(con, rel, with_top=False, with_samples=False)
        names = {c.name for c in stats}
        ref_names = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {ref.view}").fetchall()}
        if column not in names or ref_column not in ref_names:
            return ToolResult.error(f"Unknown column: {column if column not in names else ref_column}")
        k, rk = strategy_sql(normalization, quote_ident(column)), strategy_sql(normalization, quote_ident(ref_column))
        con.execute(
            f"CREATE TEMP TABLE _targets AS SELECT DISTINCT {rk} AS k FROM {ref_rel} "
            f"WHERE {quote_ident(ref_column)} IS NOT NULL"
        )
        rows, keyed, orphans, orphan_keys = con.execute(
            f"SELECT count(*), count({quote_ident(column)}), "
            f"count(*) FILTER (WHERE {quote_ident(column)} IS NOT NULL AND {k} NOT IN (SELECT k FROM _targets)), "
            f"count(DISTINCT {k}) FILTER (WHERE {quote_ident(column)} IS NOT NULL "
            f"AND {k} NOT IN (SELECT k FROM _targets)) "
            f"FROM {rel}"
        ).fetchall()[0]
        target_rows, target_distinct = con.execute(
            f"SELECT count({quote_ident(ref_column)}), count(DISTINCT {quote_ident(ref_column)}) FROM {ref_rel}"
        ).fetchall()[0]
        flag = f"{quote_ident(column)} IS NOT NULL AND {k} NOT IN (SELECT k FROM _targets)"
        segments = _segment_columns(stats, exclude={column})
        concentration = (
            _concentration(con, rel, flag, f"{quote_ident(column)} IS NOT NULL", segments, _date_expr(stats))
            if orphans
            else {}
        )
        examples = [
            r[0]
            for r in con.execute(
                f"SELECT DISTINCT {quote_ident(column)} FROM {rel} WHERE {flag} ORDER BY 1 LIMIT 3"
            ).fetchall()
        ]
        nulls = null_structure(con, rel, column, _segment_columns(stats)) if keyed < rows else None
    finally:
        _close(con)
    masker = _masker()
    sensitive = masker.is_sensitive_name(column) or masker.is_sensitive_values(examples)
    share = orphans / keyed if keyed else 0.0
    null_share = 1 - keyed / rows if rows else 0.0
    issues = []
    if orphans:
        issues.append(
            _issue(
                src.name,
                column,
                "orphan_references",
                "medium" if share < 0.2 else "high",
                rows=int(orphans),
                distinct_values=int(orphan_keys),
                share=round(share, 6),
                target=f"{ref.name}.{ref_column}",
                concentration=concentration.get("segment"),
                predicate=flag.replace("SELECT k FROM _targets", f"SELECT {rk} FROM {ref.view}"),
            )
        )
    result = {
        "dataset": src.name,
        "column": column,
        "target": f"{ref.name}.{ref_column}",
        "normalization": normalization,
        "self_reference": src.name == ref.name,
        "rows": int(rows),
        "non_null_references": int(keyed),
        "null_share": round(null_share, 4),
        "nulls": None
        if nulls is None
        else {k: nulls[k] for k in ("classification", "determined_by", "unexpected_nulls")},
        "orphan_rows": int(orphans),
        "orphan_distinct": int(orphan_keys),
        "orphan_share": round(share, 6),
        "orphans_concentrate_in": concentration.get("segment"),
        "target_uniqueness": round(target_distinct / target_rows, 6) if target_rows else 0.0,
        "orphan_examples": [masker.mask_value(v) if sensitive else v for v in examples],
    }
    record = _store().save_json(
        f"reference_{src.name}_{column}",
        {
            **result,
            "reference": {"dataset": src.name, "column": column, "ref_dataset": ref.name, "ref_column": ref_column},
            "concentration": concentration,
            "issues": issues,
        },
        parents=sorted({src.name, ref.name}),
        params={"normalization": normalization},
    )
    return ToolResult.ok(result, artifact_id=record.id, warnings=notes)


# --------------------------------------------------------------------------- #
# find_duplicate_keys                                                         #
# --------------------------------------------------------------------------- #
@tool
def find_duplicate_keys(dataset: str, columns: list[str] | None = None, min_uniqueness: float = 0.99) -> ToolResult:
    """Find values of a key that should be unique but repeat in rows that are otherwise different.

    With `columns`, checks that combination. Without it, scans every identifier
    column that is almost unique (at least `min_uniqueness` distinct values per
    non-null row) but not fully. Exact duplicate rows are not counted (see
    profile_table). For each duplicated key it reports how many keys repeat, the
    extra rows, which columns differ between the repeated rows, and where they
    concentrate.

    Args:
        dataset: Dataset name or table artifact id.
        columns: Key columns that should be unique together (optional).
        min_uniqueness: Scan threshold when `columns` is not given.
    """
    opened = _resolve(dataset)
    if isinstance(opened, ToolResult):
        return opened
    source, con = opened
    try:
        rel, notes = _working_rows(con, source.view)
        _, stats = compute_column_stats(con, rel, with_top=False, with_samples="classify")
        by_name = {c.name: c for c in stats}
        if columns:
            missing = [c for c in columns if c not in by_name]
            if missing:
                return ToolResult.error(f"Unknown columns: {missing}")
            keysets = [list(columns)]
        else:
            keysets = [
                [c.name]
                for c in stats
                if classify_column(c).semantic_type == "identifier"
                and c.non_null > 0
                and min_uniqueness <= c.distinct_ratio < 1
            ]
        segments = _segment_columns(stats)
        date = _date_expr(stats)
        checks, issues = [], []
        for keys in keysets:
            q = ", ".join(quote_ident(k) for k in keys)
            present = " AND ".join(f"{quote_ident(k)} IS NOT NULL" for k in keys)
            con.execute("DROP TABLE IF EXISTS _dup")
            con.execute(
                f"CREATE TEMP TABLE _dup AS SELECT {q}, count(*) AS n FROM {rel} WHERE {present} "
                "GROUP BY ALL HAVING count(*) > 1"
            )
            n_keys, extra = con.execute("SELECT count(*), coalesce(sum(n - 1), 0) FROM _dup").fetchall()[0]
            entry: dict[str, Any] = {"columns": keys, "duplicated_keys": int(n_keys), "extra_rows": int(extra)}
            if n_keys:
                join = " AND ".join(f"r.{quote_ident(k)} = d.{quote_ident(k)}" for k in keys)
                others = [c.name for c in stats if c.name not in keys]
                differ = con.execute(
                    "SELECT "
                    + ", ".join(f"count(*) FILTER (WHERE c{i} > 1)" for i in range(len(others)))
                    + " FROM (SELECT "
                    + ", ".join(f"count(DISTINCT r.{quote_ident(c)}) AS c{i}" for i, c in enumerate(others))
                    + f" FROM {rel} r JOIN _dup d ON {join} GROUP BY "
                    + ", ".join(f"r.{quote_ident(k)}" for k in keys)
                    + ")"
                ).fetchall()[0]
                entry["columns_that_differ"] = [c for c, n in zip(others, differ, strict=True) if n]
                # Rows of different kinds sharing a value (e.g. two movements referring to the same
                # origin) repeat it by design: a segment column differs in most repeated keys.
                by_design = [c for c, n in zip(others, differ, strict=True) if c in segments and n >= 0.5 * n_keys]
                entry["repeats_across"] = by_design or None
                flag = f"({q}) IN (SELECT ({q}) FROM _dup)" if len(keys) > 1 else f"{q} IN (SELECT {q} FROM _dup)"
                conc = _concentration(con, rel, flag, present, segments, date)
                entry["concentration"] = conc.get("segment")
                if by_design:
                    entry["note"] = f"repeated in rows of different {by_design[0]}: shared by design, not an issue"
                    checks.append(entry)
                    continue
                issues.append(
                    _issue(
                        source.name,
                        ",".join(keys),
                        "duplicate_business_keys",
                        "medium",
                        keys=int(n_keys),
                        extra_rows=int(extra),
                        columns_that_differ=entry["columns_that_differ"][:10],
                        concentration=conc.get("segment"),
                        predicate=f"({q}) IN (SELECT ({q}) FROM (SELECT DISTINCT * FROM {source.view}) "
                        f"WHERE {present} GROUP BY ALL HAVING count(*) > 1)",
                    )
                )
            checks.append(entry)
    finally:
        _close(con)
    record = _store().save_json(
        f"duplicate_keys_{source.name}",
        {"dataset": source.name, "checks": checks, "issues": issues},
        parents=[source.name],
    )
    return ToolResult.ok(
        {
            "dataset": source.name,
            "checks": [{**c, "columns_that_differ": _short(c.get("columns_that_differ", []))} for c in checks],
        },
        artifact_id=record.id,
        warnings=[*notes, *([] if keysets else ["No almost-unique identifier column to check; pass `columns`."])],
    )


# --------------------------------------------------------------------------- #
# classify_nulls                                                              #
# --------------------------------------------------------------------------- #
@tool
def classify_nulls(dataset: str) -> ToolResult:
    """Classify the nulls of every column as structural (determined by another column) or not.

    A null is structural when another low-cardinality column decides it: e.g. a
    column filled only for some values of a type column and empty for all the
    others. Structural nulls are not a quality problem and must never be imputed.
    Nulls found where the column is normally filled are reported as errors.

    Args:
        dataset: Dataset name or table artifact id.
    """
    opened = _resolve(dataset)
    if isinstance(opened, ToolResult):
        return opened
    source, con = opened
    try:
        rel, notes = _working_rows(con, source.view)
        _, stats = compute_column_stats(con, rel, with_top=False, with_samples="classify")
        determinants = [
            c
            for c in _segment_columns(stats)
            if next(s for s in stats if s.name == c).non_null == next(s for s in stats if s.name == c).rows
        ]
        results, issues = {}, []
        # One scan per determinant counts the nulls of every column at once.
        with_nulls = [c.name for c in stats if c.non_null < c.rows]
        groups = {det: _null_groups(con, rel, with_nulls, det) for det in determinants} if with_nulls else {}
        for col in stats:
            if col.non_null == col.rows:
                continue
            found = null_structure(con, rel, col.name, determinants, groups)
            if found is None:
                results[col.name] = {"classification": "not_structural", "nulls": col.rows - col.non_null}
                continue
            results[col.name] = found
            issues.append(
                _issue(
                    source.name,
                    col.name,
                    "structural_nulls",
                    "low",
                    determined_by=found["determined_by"],
                    null_when=found["null_when"],
                    rows=found["nulls"] - found["unexpected_nulls"],
                    not_a_problem=True,
                )
            )
            if found["unexpected_nulls"]:
                applicable = found["applicable_rows"]
                issues.append(
                    _issue(
                        source.name,
                        col.name,
                        "nulls",
                        "medium",
                        rows=found["unexpected_nulls"],
                        share=round(found["unexpected_nulls"] / max(applicable, 1), 5),
                        where={found["determined_by"]: [v for v, _ in found["unexpected_nulls_in"]]},
                        structural=True,
                    )
                )
    finally:
        _close(con)
    record = _store().save_json(
        f"nulls_{source.name}", {"dataset": source.name, "nulls": results, "issues": issues}, parents=[source.name]
    )
    compact = {
        name: {
            "classification": r["classification"],
            **(
                {
                    "determined_by": r["determined_by"],
                    "filled_when": _short(r["filled_when"], 4),
                    "unexpected_nulls": r["unexpected_nulls"],
                }
                if r["classification"] != "not_structural"
                else {"nulls": r["nulls"]}
            ),
        }
        for name, r in results.items()
    }
    return ToolResult.ok({"dataset": source.name, "columns": compact}, artifact_id=record.id, warnings=notes)


# --------------------------------------------------------------------------- #
# infer_sign_convention                                                       #
# --------------------------------------------------------------------------- #
@tool
def infer_sign_convention(dataset: str, measure: str, category: str) -> ToolResult:
    """Sign of a measure for each value of a category, and sign inversions concentrated in time.

    For every category value: share of positive, negative and zero values, and
    the dominant sign. When a value has a clear dominant sign and its minority
    rows fall on a few dates, they are reported as a probable sign inversion, with
    the dates and the other category value they share (if any).

    Args:
        dataset: Dataset name or table artifact id.
        measure: Numeric column (e.g. a signed amount).
        category: Low-cardinality column (e.g. a type).
    """
    opened = _resolve(dataset)
    if isinstance(opened, ToolResult):
        return opened
    source, con = opened
    try:
        rel, notes = _working_rows(con, source.view)
        _, stats = compute_column_stats(con, rel, with_top=False, with_samples="classify")
        by_name = {c.name: c for c in stats}
        if measure not in by_name or by_name[measure].family not in ("integer", "float"):
            return ToolResult.error(f"'{measure}' is not a numeric column of '{source.name}'")
        if category not in by_name:
            return ToolResult.error(f"Unknown column '{category}'")
        if by_name[category].distinct > _MAX_SIGN_CATEGORIES:
            return ToolResult.error(
                f"'{category}' has {by_name[category].distinct:,} distinct values: not a category. "
                f"Pass a low-cardinality column (at most {_MAX_SIGN_CATEGORIES} values)."
            )
        m = quote_ident(measure)
        rows = con.execute(
            f"SELECT {_norm(category)} v, count({m}) n, count(*) FILTER (WHERE {m} > 0) pos, "
            f"count(*) FILTER (WHERE {m} < 0) neg FROM {rel} GROUP BY 1 ORDER BY n DESC"
        ).fetchall()
        date = _date_expr(stats)
        # Signs per (value, day) for every value at once, instead of one query per value.
        per_day: dict[Any, list[tuple[Any, int, int]]] = {}
        n_dates = 0
        if date:
            signs = con.execute(
                f"SELECT {_norm(category)} v, {date} d, count(*) FILTER (WHERE {m} > 0), "
                f"count(*) FILTER (WHERE {m} < 0) FROM {rel} GROUP BY 1, 2"
            ).fetchall()
            n_dates = len({d for _, d, _, _ in signs if d is not None})
            for v, d, pos_d, neg_d in signs:
                per_day.setdefault(v, []).append((d, int(pos_d), int(neg_d)))
        others = _segment_columns(stats, exclude={category, measure})
        values, inversions = [], []
        for v, n, pos, neg in rows:
            if not n:
                continue
            dominant = (
                "positive"
                if pos >= 0.9 * n
                else "negative"
                if neg >= 0.9 * n
                else "zero"
                if pos + neg == 0
                else "mixed"
            )
            entry = {
                "value": v,
                "rows": int(n),
                "positive": round(pos / n, 4),
                "negative": round(neg / n, 4),
                "dominant": dominant,
            }
            values.append(entry)
            minority = (neg if dominant == "positive" else pos) if dominant in ("positive", "negative") else 0
            if not minority or not date:
                continue
            op = "<" if dominant == "positive" else ">"
            flag = f"{_norm(category)} = {_lit(v)} AND {m} {op} 0"
            minority_by_day = [
                (d, neg_d if dominant == "positive" else pos_d) for d, pos_d, neg_d in per_day.get(v, [])
            ]
            by_date = sorted(((d, k) for d, k in minority_by_day if k), key=lambda x: (-x[1], x[0] is None, str(x[0])))
            covered, chosen = 0, []
            for d, k in by_date:
                chosen.append(str(d))
                covered += k
                if covered >= 0.95 * minority:
                    break
            if len(chosen) <= max(3, 0.1 * n_dates) and minority >= _MIN_SEGMENT_ROWS:
                shared = _concentration(con, rel, flag, f"{_norm(category)} = {_lit(v)}", others, None).get("segment")
                inversions.append(
                    {
                        "value": v,
                        "rows": int(minority),
                        "dates": sorted(chosen),
                        "share_of_value": round(minority / n, 4),
                        "segment": shared,
                    }
                )
    finally:
        _close(con)
    # One issue per group of values inverted on the same dates (a single systematic error).
    issues = []
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for inv in inversions:
        groups.setdefault(tuple(inv["dates"]), []).append(inv)
    for dates, members in groups.items():
        issues.append(
            _issue(
                source.name,
                measure,
                "sign_inversion",
                "high",
                category=category,
                values=[x["value"] for x in members],
                dates=list(dates),
                rows=sum(x["rows"] for x in members),
                segment=members[0]["segment"],
            )
        )
    record = _store().save_json(
        f"sign_{source.name}_{measure}",
        {
            "dataset": source.name,
            "measure": measure,
            "category": category,
            "values": values,
            "inversions": inversions,
            "issues": issues,
        },
        parents=[source.name],
    )
    return ToolResult.ok(
        {
            "dataset": source.name,
            "measure": measure,
            "category": category,
            "values": values[:16],
            "inversions": inversions[:6],
        },
        artifact_id=record.id,
        warnings=notes,
    )


# --------------------------------------------------------------------------- #
# discover_self_references / check_reciprocity                                #
# --------------------------------------------------------------------------- #
@tool
def discover_self_references(dataset: str, min_coverage: float = 0.9) -> ToolResult:
    """Find columns whose values are values of another column of the same dataset.

    Two kinds: a **reference** points to a unique column (a row id: e.g. a
    "parent" or "origin" id), to be checked with check_referential_integrity; a
    **counterpart** points to a non-unique identifier of the same kind of entity
    (e.g. the other party of an exchange), to be checked with check_reciprocity.

    Args:
        dataset: Dataset name or table artifact id.
        min_coverage: Minimum share of the column's distinct values found in the target.
    """
    opened = _resolve(dataset)
    if isinstance(opened, ToolResult):
        return opened
    source, con = opened
    try:
        rel, notes = _working_rows(con, source.view)
        _, stats = compute_column_stats(con, rel, with_top=False, with_samples="classify")
        keys = [
            c
            for c in stats
            if c.family in ("string", "integer")
            and classify_column(c).semantic_type == "identifier"
            and c.distinct >= 20
        ]
        if len(keys) > _MAX_KEY_COLUMNS:
            kept = {c.name for c in sorted(keys, key=lambda c: -c.distinct)[:_MAX_KEY_COLUMNS]}
            notes.append(
                f"{len(keys)} identifier columns: only the {_MAX_KEY_COLUMNS} with most distinct values were compared."
            )
            keys = [c for c in keys if c.name in kept]
        # A column can only be covered by a target with enough distinct values: pairs that
        # cannot reach `min_coverage` are dropped before any value is read.
        pairs = [
            ((i, a), (j, b))
            for (i, a), (j, b) in itertools.permutations(enumerate(keys), 2)
            if a.family == b.family and b.distinct >= min_coverage * a.distinct
        ]
        for i in sorted({i for pair in pairs for i, _ in pair}):
            c = keys[i]
            con.execute(
                f"CREATE TEMP TABLE _s{i} AS SELECT DISTINCT CAST({quote_ident(c.name)} AS VARCHAR) v "
                f"FROM {rel} WHERE {quote_ident(c.name)} IS NOT NULL"
            )
        found = []
        for (i, a), (j, b) in pairs:
            common = con.execute(f"SELECT count(*) FROM _s{i} JOIN _s{j} USING (v)").fetchall()[0][0]
            coverage = common / a.distinct if a.distinct else 0.0
            if coverage < min_coverage:
                continue
            unique = b.distinct_ratio >= 0.999
            found.append(
                {
                    "column": a.name,
                    "target": b.name,
                    "kind": "reference" if unique else "counterpart",
                    "value_coverage": round(coverage, 4),
                    "column_null_share": round(a.null_share, 4),
                    "target_uniqueness": round(b.distinct_ratio, 5),
                    "next": "check_referential_integrity" if unique else "check_reciprocity",
                }
            )
    finally:
        _close(con)
    record = _store().save_json(
        f"self_references_{source.name}", {"dataset": source.name, "candidates": found}, parents=[source.name]
    )
    return ToolResult.ok({"dataset": source.name, "candidates": found[:10]}, artifact_id=record.id, warnings=notes)


@tool
def check_reciprocity(
    dataset: str,
    column: str,
    counterpart_column: str,
    time_column: str | None = None,
    max_lag_seconds: Annotated[int, Field(ge=0)] = 300,
) -> ToolResult:
    """Check that rows `A -> B` (column = A, counterpart_column = B) have a matching row `B -> A`.

    With `time_column`, the matching row must be within `max_lag_seconds`.
    Rows without their reciprocal are reported, with where they concentrate
    (e.g. one category value).

    Args:
        dataset: Dataset name or table artifact id.
        column: Column of the row's own party.
        counterpart_column: Column of the other party.
        time_column: Event time used to pair the rows (recommended).
        max_lag_seconds: Maximum time between the two rows of a pair.
    """
    opened = _resolve(dataset)
    if isinstance(opened, ToolResult):
        return opened
    source, con = opened
    try:
        rel, notes = _working_rows(con, source.view)
        _, stats = compute_column_stats(con, rel, with_top=False, with_samples="classify")
        names = {c.name for c in stats}
        for c in (column, counterpart_column, time_column):
            if c and c not in names:
                return ToolResult.error(f"Unknown column '{c}'")
        # The other party is the sparse column (filled only on exchange rows): orient the pair.
        nulls = {c.name: c.null_share for c in stats}
        swapped = nulls[column] > nulls[counterpart_column]
        if swapped:
            column, counterpart_column = counterpart_column, column
        a, b = quote_ident(column), quote_ident(counterpart_column)
        lag = ""
        if time_column:
            t = quote_ident(time_column)
            lag = f" AND abs(epoch(o.{t}) - epoch(x.{t})) <= {int(max_lag_seconds)}"
        # Only the columns used below are copied: the pair, the time and the segment/day columns.
        segments = _segment_columns(stats, exclude={column, counterpart_column})
        used = dict.fromkeys(
            [column, counterpart_column, *([time_column] if time_column else []), *segments, _date_column(stats)]
        )
        kept = ", ".join(quote_ident(c) for c in used if c)
        con.execute(
            f"CREATE TEMP TABLE _pairs AS SELECT row_number() OVER () AS _rid, {kept} FROM {rel} WHERE {b} IS NOT NULL"
        )
        con.execute(
            f"CREATE TEMP TABLE _missing AS SELECT x._rid FROM _pairs x WHERE NOT EXISTS "
            f"(SELECT 1 FROM _pairs o WHERE o.{a} = x.{b} AND o.{b} = x.{a}{lag})"
        )
        total, missing = con.execute(
            "SELECT (SELECT count(*) FROM _pairs), (SELECT count(*) FROM _missing)"
        ).fetchall()[0]
        conc = (
            _concentration(con, "_pairs", "_rid IN (SELECT _rid FROM _missing)", "TRUE", segments, _date_expr(stats))
            if missing
            else {}
        )
    finally:
        _close(con)
    issues = []
    if missing:
        issues.append(
            _issue(
                source.name,
                counterpart_column,
                "missing_reciprocal",
                "medium",
                rows=int(missing),
                share=round(missing / total, 6),
                pair=[column, counterpart_column],
                max_lag_seconds=max_lag_seconds if time_column else None,
                concentration=conc.get("segment"),
                predicate=f"x.{b} IS NOT NULL AND NOT EXISTS (SELECT 1 FROM {source.view} o "
                f"WHERE o.{a} = x.{b} AND o.{b} = x.{a}{lag})",
            )
        )
    result = {
        "dataset": source.name,
        "pair": [column, counterpart_column],
        "rows_with_counterpart": int(total),
        "without_reciprocal": int(missing),
        "share": round(missing / total, 6) if total else 0.0,
        "concentrate_in": conc.get("segment"),
        "time_column": time_column,
        "max_lag_seconds": max_lag_seconds if time_column else None,
    }
    if swapped:
        result["note"] = f"columns swapped: '{counterpart_column}' is the sparse column, the other party"
    record = _store().save_json(
        f"reciprocity_{source.name}_{counterpart_column}",
        {
            **result,
            "reference": {
                "dataset": source.name,
                "column": counterpart_column,
                "ref_dataset": source.name,
                "ref_column": column,
                "kind": "counterpart",
            },
            "issues": issues,
        },
        parents=[source.name],
    )
    return ToolResult.ok(result, artifact_id=record.id, warnings=notes)
