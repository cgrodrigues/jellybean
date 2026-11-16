"""Profiler tools (spec section 7.2).

The profiler understands the data before anyone touches it: it profiles each
dataset, infers semantic types, grain, entity and time column, finds quality
problems and discovers how datasets join. Every tool computes inside DuckDB,
stores its full output as an artifact and returns a compact summary.

Quality problems found by any tool are returned under ``summary["issues"]`` and
stored in the tool's artifact with the same structure, so
:func:`compile_quality_issues` can merge them for the preparer::

    {"dataset": ..., "column": ... | None, "kind": ..., "severity": "low|medium|high",
     "detail": {...}, "source_tool": ...}
"""

from __future__ import annotations

import difflib
import math
import statistics
from typing import Annotated, Any

from pydantic import Field

from jellybean.runtime.context import maybe_current_run
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.skills._shared.artifacts import ArtifactNotFoundError, ArtifactStore
from jellybean.skills._shared.catalog import (
    Source,
    SourceNotFoundError,
    connect,
    dataset_relation_with_files,
    quote_ident,
    resolve_source,
)
from jellybean.skills._shared.column_stats import (
    ColumnStats,
    classify_column,
    compute_column_stats,
    describe,
    name_tokens,
    type_family,
)
from jellybean.skills._shared.keys import (
    BASE_STRATEGIES,
    UnknownStrategyError,
    detect_prefixes,
    is_valid_strategy,
    prefix_strategy,
    strategy_sql,
)
from jellybean.skills._shared.masking import Masker
from jellybean.skills._shared.spec_tools import SpecMissingError, patch_spec, read_spec
from jellybean.skills.profiler._common import _issue, _masker, _open, _settings, _severity, _store
from jellybean.skills.profiler.integrity import (  # noqa: F401  (re-exported: the profiler's tools)
    _segment_columns,
    check_reciprocity,
    check_referential_integrity,
    classify_nulls,
    discover_arithmetic_invariants,
    discover_self_references,
    find_duplicate_keys,
    infer_sign_convention,
    null_structure,
)
from jellybean.spec import assess_join, load_spec

# Columns whose per-file median differs from the typical file by this factor
# are reported as a possible unit error.
_SCALE_SHIFT_FACTOR = 10.0
# Join discovery: minimum distinct values for a column to be a key candidate.
_MIN_KEY_DISTINCT = 20
# Number of values sampled to detect prefixes between key columns.
_PREFIX_SAMPLE = 2000


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _safe_value(masker: Masker, column: str, value: Any, sensitive: bool) -> Any:
    """Mask a single value when the column is sensitive."""
    return masker.mask_value(value) if sensitive and value is not None else value


def _is_sensitive(masker: Masker, col: ColumnStats) -> bool:
    values = [v for v, _ in col.top_values] + list(col.sample_values)
    return masker.is_sensitive_name(col.name) or masker.is_sensitive_values(values)


def _file_relation(source: Source) -> str | None:
    """Relation with an internal ``__file`` column, for multi-file datasets only."""
    if source.dataset is None or len(source.dataset.files) < 2:
        return None
    return dataset_relation_with_files(source.dataset)


def _robust_z(values: list[float]) -> list[float]:
    """Robust z-scores based on the median and the MAD."""
    if len(values) < 3:
        return [0.0] * len(values)
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values]) * 1.4826
    if mad == 0:
        return [0.0 if v == med else math.inf for v in values]
    return [(v - med) / mad for v in values]


# --------------------------------------------------------------------------- #
# profile_table                                                               #
# --------------------------------------------------------------------------- #
def _category_variants(con, view: str, col: ColumnStats) -> tuple[int, list[list[Any]]]:
    """Distinct values collapsing to the same lower/trimmed form."""
    q = quote_ident(col.name)
    groups = con.execute(
        f"SELECT lower(trim({q})) AS k, list(DISTINCT {q} ORDER BY {q}) AS variants FROM {view} "
        f"WHERE {q} IS NOT NULL GROUP BY 1 HAVING count(DISTINCT {q}) > 1 ORDER BY 1"
    ).fetchall()
    return len(groups), [g[1] for g in groups[:5]]


def _scale_shift_by_file(con, source: Source, measures: list[str]) -> list[dict[str, Any]]:
    """Per-file medians of measure columns; files deviating by >= 10x are flagged."""
    relation = _file_relation(source)
    if relation is None or not measures:
        return []
    meds = ", ".join(f"median({quote_ident(m)})" for m in measures)
    rows = con.execute(f"SELECT __file, count(*), {meds} FROM {relation} GROUP BY 1 ORDER BY 1").fetchall()
    findings = []
    for j, measure in enumerate(measures):
        values = [(r[0], r[1], r[2 + j]) for r in rows if r[2 + j] is not None]
        typical = statistics.median([abs(v) for _, _, v in values]) if values else 0
        if not typical:
            continue
        shifted = []
        for file, n, med in values:
            factor = abs(med) / typical
            if factor >= _SCALE_SHIFT_FACTOR or (med and factor <= 1 / _SCALE_SHIFT_FACTOR):
                shifted.append(
                    {
                        "column": measure,
                        "file": file,
                        "rows": int(n),
                        "file_median": med,
                        "typical_median": typical,
                        "factor": round(factor, 2),
                    }
                )
        findings += _collapse_by_partition(shifted, [r[0] for r in rows])
    return findings


def _collapse_by_partition(shifted: list[dict[str, Any]], all_files: list[str]) -> list[dict[str, Any]]:
    """Report one issue per top-level partition when ALL its files are shifted.

    With nested partitions (e.g. date=.../hour=...) a problem that affects a whole
    day shows up in all its files; it is reported once for ``date=.../`` (the
    ``file`` value ends with "/") instead of once per file.
    """

    def top(path: str) -> str | None:
        return path.split("/", 1)[0] + "/" if "/" in path else None

    by_top: dict[str | None, list[dict[str, Any]]] = {}
    for item in shifted:
        by_top.setdefault(top(item["file"]), []).append(item)
    out = []
    for prefix, items in by_top.items():
        files_in_top = [f for f in all_files if prefix and f.startswith(prefix)]
        if prefix and len(items) > 1 and len(items) == len(files_in_top):
            out.append(
                {
                    "column": items[0]["column"],
                    "file": prefix,
                    "files": len(items),
                    "rows": sum(i["rows"] for i in items),
                    "file_median": statistics.median(i["file_median"] for i in items),
                    "typical_median": items[0]["typical_median"],
                    "factor": round(statistics.median(i["factor"] for i in items), 2),
                }
            )
        else:
            out += items
    return out


@tool
def profile_table(dataset: str, seed: int | None = None) -> ToolResult:
    """Profile a dataset or table artifact: nulls, cardinality, quantiles, top values and quality issues.

    Detects exact duplicate rows, nulls, rare negative values, constant columns,
    inconsistent category spellings and per-file scale shifts (possible unit
    errors). The full profile, with histograms, is stored as an artifact.

    Args:
        dataset: Dataset name or table artifact id.
        seed: Seed for sampled values (defaults to the run seed).
    """
    try:
        source = resolve_source(dataset)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        view = source.view
        rows, stats = compute_column_stats(con, view, seed=seed or 0)
        masker = _masker()
        distinct_rows = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {view})").fetchall()[0][0]
        issues: list[dict[str, Any]] = []
        if rows - distinct_rows > 0:
            dup = rows - distinct_rows
            issues.append(
                _issue(source.name, None, "duplicate_rows", _severity(dup / rows), rows=dup, share=round(dup / rows, 5))
            )

        columns_out, histograms, measures = [], {}, []
        for col in stats:
            q = quote_ident(col.name)
            sensitive = _is_sensitive(masker, col)
            semantic = classify_column(col).semantic_type
            if col.non_null < col.rows:
                issues.append(
                    _issue(
                        source.name,
                        col.name,
                        "nulls",
                        _severity(col.null_share),
                        rows=col.rows - col.non_null,
                        share=round(col.null_share, 5),
                    )
                )
            if col.distinct == 1:
                issues.append(_issue(source.name, col.name, "constant_column", "low"))
            if col.family in ("integer", "float") and col.min is not None and col.min < 0:
                negatives = con.execute(f"SELECT count(*) FROM {view} WHERE {q} < 0").fetchall()[0][0]
                share = negatives / max(col.non_null, 1)
                if share < 0.01:  # rare negatives in a mostly positive column look invalid
                    issues.append(
                        _issue(
                            source.name,
                            col.name,
                            "negative_values",
                            "medium",
                            rows=int(negatives),
                            share=round(share, 6),
                        )
                    )
            if col.family == "string" and 1 < col.distinct <= 5000 and not sensitive:
                n_groups, examples = _category_variants(con, view, col)
                if n_groups:
                    issues.append(
                        _issue(
                            source.name,
                            col.name,
                            "inconsistent_categories",
                            "medium",
                            groups=n_groups,
                            examples=examples,
                        )
                    )
            if semantic == "measure":
                measures.append(col.name)
                if col.quantiles and col.quantiles["p1"] is not None and col.quantiles["p99"] is not None:
                    lo, hi = col.quantiles["p1"], col.quantiles["p99"]
                    if hi > lo:
                        width = (hi - lo) / 10
                        histograms[col.name] = {
                            "bin_start": lo,
                            "bin_width": width,
                            "counts": [
                                int(r[1])
                                for r in con.execute(
                                    f"SELECT least(9, greatest(0, floor(({q} - {lo}) / {width})))::INT b, count(*) "
                                    f"FROM {view} WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY 1"
                                ).fetchall()
                            ],
                        }
            entry: dict[str, Any] = {
                "name": col.name,
                "type": col.type,
                "null_pct": round(100 * col.null_share, 2),
                "distinct": col.distinct,
            }
            if col.family in ("integer", "float"):
                entry |= {"min": col.min, "p50": col.quantiles["p50"] if col.quantiles else None, "max": col.max}
            elif col.family == "temporal":
                entry |= {"min": col.min, "max": col.max}
            elif col.family == "string":
                entry["top"] = [[_safe_value(masker, col.name, v, sensitive), n] for v, n in col.top_values[:3]]
            if sensitive:
                entry["masked"] = True
            columns_out.append(entry)

        for shift in _scale_shift_by_file(con, source, measures):
            issues.append(_issue(source.name, shift.pop("column"), "scale_shift_by_file", "high", **shift))

        full = {
            "dataset": source.name,
            "rows": rows,
            "distinct_rows": distinct_rows,
            "columns": [
                {
                    **c.to_dict(),
                    "sample_values": [
                        _safe_value(masker, c.name, v, _is_sensitive(masker, c)) for v in c.sample_values
                    ],
                    "top_values": [
                        [_safe_value(masker, c.name, v, _is_sensitive(masker, c)), n] for v, n in c.top_values
                    ],
                }
                for c in stats
            ],
            "histograms": histograms,
            "issues": issues,
        }
        parents = [source.name]
    finally:
        con.close()

    record = _store().save_json(f"profile_{source.name}", full, parents=parents, params={"seed": seed})
    compact_issues = [
        {
            "column": i["column"],
            "kind": i["kind"],
            "severity": i["severity"],
            **{k: v for k, v in i["detail"].items() if k in ("rows", "share", "file", "factor", "groups")},
        }
        for i in issues
    ]
    return ToolResult.ok(
        {"dataset": source.name, "rows": rows, "columns": columns_out, "issues": compact_issues},
        artifact_id=record.id,
    )


# --------------------------------------------------------------------------- #
# infer_semantic_types                                                        #
# --------------------------------------------------------------------------- #
@tool
def infer_semantic_types(dataset: str, seed: int | None = None) -> ToolResult:
    """Classify each column as identifier, timestamp, measure, category, boolean, free_text, geo or unknown.

    Uses the column name, its statistics and 5 sample values (which are never returned).

    Args:
        dataset: Dataset name or table artifact id.
        seed: Seed used to pick the sample values.
    """
    try:
        with _open(dataset) as (con, source):
            _, stats = compute_column_stats(con, source.view, seed=seed or 0)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    guesses = {c.name: classify_column(c) for c in stats}
    result = {
        name: {"semantic_type": g.semantic_type, "confidence": g.confidence, "reason": g.reason}
        for name, g in guesses.items()
    }
    time_candidates = [n for n, g in guesses.items() if g.semantic_type == "timestamp"]
    record = _store().save_json(
        f"semantic_types_{source.name}", {"dataset": source.name, "columns": result}, parents=[source.name]
    )
    compact = {n: [g.semantic_type, g.confidence] for n, g in guesses.items()}
    return ToolResult.ok(
        {"dataset": source.name, "columns": compact, "time_column_candidates": time_candidates},
        artifact_id=record.id,
    )


# --------------------------------------------------------------------------- #
# infer_grain                                                                 #
# --------------------------------------------------------------------------- #
@tool
def infer_grain(dataset: str) -> ToolResult:
    """Find column combinations that uniquely identify a row (the grain).

    Reports single columns that are unique, unique once exact duplicate rows are
    removed, or nearly unique; tries column pairs only when no single column works.

    Args:
        dataset: Dataset name or table artifact id.
    """
    try:
        with _open(dataset) as (con, source):
            rows, stats = compute_column_stats(con, source.view, with_top=False, with_samples=False)
            distinct_rows = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {source.view})").fetchall()[0][0]
            candidates: list[dict[str, Any]] = []
            for col in stats:
                if col.non_null < rows:
                    continue
                if col.distinct == rows:
                    candidates.append({"columns": [col.name], "uniqueness": 1.0, "status": "unique"})
                elif col.distinct == distinct_rows:
                    candidates.append(
                        {
                            "columns": [col.name],
                            "uniqueness": round(col.distinct / rows, 5),
                            "status": "unique_after_removing_exact_duplicates",
                        }
                    )
                elif col.distinct / rows >= 0.99:
                    candidates.append(
                        {"columns": [col.name], "uniqueness": round(col.distinct / rows, 5), "status": "near_unique"}
                    )
            if not any(c["status"] != "near_unique" for c in candidates):
                # Try pairs among the highest-cardinality complete columns.
                top = [c.name for c in sorted(stats, key=lambda c: -c.distinct) if c.non_null == rows][:8]
                for i, a in enumerate(top):
                    for b in top[i + 1 :]:
                        n = con.execute(
                            f"SELECT count(*) FROM (SELECT DISTINCT {quote_ident(a)}, {quote_ident(b)} "
                            f"FROM {source.view})"
                        ).fetchall()[0][0]
                        if n == rows:
                            candidates.append({"columns": [a, b], "uniqueness": 1.0, "status": "unique"})
                        elif n == distinct_rows:
                            candidates.append(
                                {
                                    "columns": [a, b],
                                    "uniqueness": round(n / rows, 5),
                                    "status": "unique_after_removing_exact_duplicates",
                                }
                            )
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    order = {"unique": 0, "unique_after_removing_exact_duplicates": 1, "near_unique": 2}
    candidates.sort(key=lambda c: (order[c["status"]], len(c["columns"]), -c["uniqueness"]))
    issues = []
    if rows > distinct_rows:
        issues.append(
            _issue(
                source.name, None, "duplicate_rows", _severity((rows - distinct_rows) / rows), rows=rows - distinct_rows
            )
        )
    warnings = [] if candidates else ["No unique column or pair found; the grain may need more columns."]
    record = _store().save_json(
        f"grain_{source.name}",
        {
            "dataset": source.name,
            "rows": rows,
            "distinct_rows": distinct_rows,
            "candidates": candidates,
            "issues": issues,
        },
        parents=[source.name],
    )
    return ToolResult.ok(
        {
            "dataset": source.name,
            "rows": rows,
            "exact_duplicate_rows": rows - distinct_rows,
            "candidates": candidates[:8],
            "recommended": candidates[0]["columns"] if candidates else None,
        },
        artifact_id=record.id,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# suggest_entities                                                            #
# --------------------------------------------------------------------------- #
def _join_key_columns(dataset: str) -> set[str]:
    """Columns of ``dataset`` used as join keys in the run spec (if any)."""
    run = maybe_current_run()
    if run is None or not run.spec_path.exists():
        return set()
    spec = load_spec(run.spec_path)
    cols: set[str] = set()
    for join in spec.joins:
        if join.status == "rejected":
            continue
        for left_col, right_col in join.keys:
            if join.left == dataset:
                cols.add(left_col)
            if join.right == dataset:
                cols.add(right_col)
    return cols


@tool
def suggest_entities(dataset: str) -> ToolResult:
    """Rank columns that could be the analysis entity (the "who" rows belong to).

    Candidates are identifier/category columns whose values repeat across rows.
    The score rewards completeness, values that recur (saturating at 5 rows per
    value), the share of repeated values and cardinality, plus a bonus when the
    column is a join key in the run spec.

    Args:
        dataset: Dataset name or table artifact id.
    """
    try:
        with _open(dataset) as (con, source):
            _rows, stats = compute_column_stats(con, source.view, with_top=True, with_samples=True)
            join_keys = _join_key_columns(source.name)
            candidates = []
            for col in stats:
                semantic = classify_column(col).semantic_type
                if semantic not in ("identifier", "category") or col.family not in ("string", "integer"):
                    continue
                if col.distinct < 50 or col.distinct_ratio > 0.5:
                    continue
                q = quote_ident(col.name)
                repeated = con.execute(
                    f"SELECT avg((n >= 2)::INT), median(n), max(n) FROM (SELECT count(*) AS n FROM {source.view} "
                    f"WHERE {q} IS NOT NULL GROUP BY {q})"
                ).fetchall()[0]
                mean_rows = col.non_null / col.distinct
                share_repeated = float(repeated[0])
                # Repetition saturates at 5 rows per value: it only has to show that
                # values recur. Beyond that, more distinct entities rank higher.
                score = (
                    (1 - col.null_share) ** 2
                    * min(1.0, math.log10(max(mean_rows, 1)) / math.log10(5))
                    * min(1.0, math.log10(col.distinct) / 4)
                    * math.sqrt(share_repeated)
                )
                bonus = col.name in join_keys
                score += 0.1 if bonus else 0.0  # additive, so it is not lost to saturation
                candidates.append(
                    {
                        "column": col.name,
                        "semantic_type": semantic,
                        "distinct": col.distinct,
                        "null_pct": round(100 * col.null_share, 2),
                        "mean_rows_per_value": round(mean_rows, 2),
                        "median_rows_per_value": float(repeated[1]),
                        "max_rows_per_value": int(repeated[2]),
                        "share_values_repeated": round(share_repeated, 4),
                        "join_key_bonus": bonus,
                        "score": round(score, 4),
                    }
                )
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    candidates.sort(key=lambda c: -c["score"])
    record = _store().save_json(
        f"entities_{source.name}", {"dataset": source.name, "candidates": candidates}, parents=[source.name]
    )
    best = candidates[0]["column"] if candidates else None
    return ToolResult.ok(
        {
            "dataset": source.name,
            "recommended": best,
            "alternatives": [c["column"] for c in candidates[1:4]],
            "candidates": candidates[:6],
        },
        artifact_id=record.id,
        warnings=[] if candidates else ["No column repeats enough to be an entity."],
    )


# --------------------------------------------------------------------------- #
# check_temporal_coverage                                                     #
# --------------------------------------------------------------------------- #
def _tv_distance(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True))


@tool
def check_temporal_coverage(dataset: str, time_column: str) -> ToolResult:
    """Check a time column: range, gaps, volume spikes and format/time-zone changes.

    Detects missing days, days and hours with abnormal volume (robust z-scores),
    partitions whose timestamps fall outside the partition date, and days whose
    hour-of-day profile differs from the rest (typical of a time-zone shift).

    Args:
        dataset: Dataset name or table artifact id.
        time_column: Column holding the event time.
    """
    try:
        source = resolve_source(dataset)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        schema = dict(describe(con, source.view))
        if time_column not in schema:
            return ToolResult.error(f"Column '{time_column}' not found. Columns: {list(schema)}")
        q = quote_ident(time_column)
        family = type_family(schema[time_column])
        t = q if family == "temporal" else f"try_cast({q} AS TIMESTAMP)"
        view = source.view
        total, nulls, unparsable, tmin, tmax = con.execute(
            f"SELECT count(*), count(*) - count({q}), count({q}) - count({t}), min({t}), max({t}) FROM {view}"
        ).fetchall()[0]
        issues: list[dict[str, Any]] = []
        if unparsable:
            issues.append(
                _issue(
                    source.name,
                    time_column,
                    "unparsable_timestamps",
                    _severity(unparsable / total),
                    rows=int(unparsable),
                )
            )
        if tmin is None:
            return ToolResult.error(f"Column '{time_column}' has no parsable timestamps.")

        daily = con.execute(
            f"SELECT CAST({t} AS DATE) d, count(*) FROM {view} WHERE {t} IS NOT NULL GROUP BY 1 ORDER BY 1"
        ).fetchall()
        days = [d for d, _ in daily]
        all_days = con.execute(
            f"SELECT CAST(range AS DATE) FROM range(DATE '{days[0]}', DATE '{days[-1]}' + INTERVAL 1 DAY, "
            "INTERVAL 1 DAY)"
        ).fetchall()
        gaps = sorted(set(r[0] for r in all_days) - set(days))
        if gaps:
            issues.append(
                _issue(
                    source.name, time_column, "time_gaps", "medium", days=len(gaps), first=[str(g) for g in gaps[:5]]
                )
            )
        counts = [float(n) for _, n in daily]
        # Weekly seasonality: with two or more weeks of data, compare each day with
        # the median of the same weekday before looking for outliers.
        if len(days) >= 14:
            by_dow: dict[int, list[float]] = {}
            for d, n in zip(days, counts, strict=True):
                by_dow.setdefault(d.weekday(), []).append(n)
            ratios = [n / max(statistics.median(by_dow[d.weekday()]), 1.0) for d, n in zip(days, counts, strict=True)]
        else:
            ratios = counts
        z = _robust_z(ratios)
        # Skip the first/last day (often partial). A day is flagged only if it is a
        # robust outlier AND deviates by at least 5% from its weekday median, so
        # ordinary noise on large volumes is not reported. Increases and drops apart.
        baseline = ratios if len(days) >= 14 else [n / max(statistics.median(counts), 1.0) for n in counts]
        daily_spikes = [
            {
                "date": str(days[i]),
                "rows": int(counts[i]),
                "z": round(z[i], 1),
                "relative_change": round(baseline[i] - 1, 3),
                "direction": "increase" if z[i] > 0 else "drop",
            }
            for i in range(1, len(days) - 1)
            if abs(z[i]) >= 4 and abs(baseline[i] - 1) >= 0.05
        ]
        hourly = con.execute(
            f"SELECT date_trunc('hour', {t}) h, count(*) FROM {view} WHERE {t} IS NOT NULL GROUP BY 1 ORDER BY 1"
        ).fetchall()
        hz = _robust_z([float(n) for _, n in hourly])
        hourly_spikes = sorted(
            (
                {"hour": str(h), "rows": int(n), "z": round(zz, 1)}
                for (h, n), zz in zip(hourly, hz, strict=True)
                if zz >= 8
            ),
            key=lambda s: -s["z"],
        )[:5]
        for spike in daily_spikes:
            kind = "volume_spike_day" if spike["direction"] == "increase" else "volume_drop_day"
            issues.append(_issue(source.name, time_column, kind, "medium", **spike))

        # Partition consistency: timestamps outside their date partition.
        partition_mismatch: list[dict[str, Any]] = []
        group_col = None
        for part in source.dataset.partition_columns if source.dataset else ():
            if type_family(schema.get(part, "")) == "temporal":
                group_col = quote_ident(part)
                rows = con.execute(
                    f"SELECT {group_col}, count(*), avg((CAST({t} AS DATE) <> CAST({group_col} AS DATE))::DOUBLE) "
                    f"FROM {view} WHERE {t} IS NOT NULL GROUP BY 1 ORDER BY 1"
                ).fetchall()
                for value, n, share in rows:
                    if share and share > 0.01:
                        partition_mismatch.append(
                            {"partition": f"{part}={value}", "rows": int(n), "share_outside_partition": round(share, 4)}
                        )
                break
        for mismatch in partition_mismatch:
            issues.append(_issue(source.name, time_column, "timestamp_outside_partition", "high", **mismatch))

        # Hour-of-day profile per day (per partition when available).
        day_expr = f"CAST({group_col} AS DATE)" if group_col else f"CAST({t} AS DATE)"
        profile_rows = con.execute(
            f"SELECT {day_expr} d, hour({t}) h, count(*) FROM {view} WHERE {t} IS NOT NULL GROUP BY 1, 2"
        ).fetchall()
        per_day: dict[Any, list[float]] = {}
        for d, h, n in profile_rows:
            per_day.setdefault(d, [0.0] * 24)[h] += n
        tvs = {}
        for d, hours in per_day.items():
            if not sum(hours):
                continue
            # Reference profile: other days with the same weekday; if too few, days of
            # the same type (weekday vs weekend); as a last resort, all other days.
            peers = [h for o, h in per_day.items() if o != d and o.weekday() == d.weekday()]
            if len(peers) < 2:
                weekend = d.weekday() >= 5
                peers = [h for o, h in per_day.items() if o != d and (o.weekday() >= 5) == weekend]
            if len(peers) < 2:
                peers = [h for o, h in per_day.items() if o != d]
            if not peers:
                continue  # a single day (or one date partition): nothing to compare it with
            # Per-hour median of the peers' normalised profiles: robust to one odd peer.
            normalised = [[v / (sum(h) or 1.0) for v in h] for h in peers]
            reference = [statistics.median(p[i] for p in normalised) for i in range(24)]
            total_ref = sum(reference) or 1.0
            tvs[d] = _tv_distance([v / sum(hours) for v in hours], [v / total_ref for v in reference])
        tv_values = list(tvs.values())
        med = statistics.median(tv_values) if tv_values else 0
        mad = statistics.median([abs(v - med) for v in tv_values]) * 1.4826 if tv_values else 0
        profile_shifts = [
            {"date": str(d), "tv_distance": round(v, 3)}
            for d, v in sorted(tvs.items())
            if v > 0.1 and v > med + 5 * max(mad, 0.01)
        ]
        mismatch_dates = {m["partition"].split("=", 1)[1] for m in partition_mismatch}
        for shift in profile_shifts:
            kind = "possible_timezone_shift" if shift["date"] in mismatch_dates else "hour_profile_shift"
            issues.append(_issue(source.name, time_column, kind, "high", **shift))
    finally:
        con.close()

    full = {
        "dataset": source.name,
        "time_column": time_column,
        "min": tmin,
        "max": tmax,
        "rows": total,
        "null_rows": nulls,
        "daily_counts": [[str(d), n] for d, n in daily],
        "gaps": [str(g) for g in gaps],
        "daily_spikes": daily_spikes,
        "hourly_spikes": hourly_spikes,
        "partition_mismatch": partition_mismatch,
        "hour_profile_tv": {str(k): round(v, 4) for k, v in tvs.items()},
        "issues": issues,
    }
    record = _store().save_json(f"temporal_{source.name}", full, parents=[source.name])
    return ToolResult.ok(
        {
            "dataset": source.name,
            "time_column": time_column,
            "min": tmin,
            "max": tmax,
            "days": len(days),
            "null_rows": nulls,
            "gaps": len(gaps),
            "daily_spikes": daily_spikes[:5],
            "hourly_spikes": hourly_spikes,
            "issues": [
                {"kind": i["kind"], **i["detail"]}
                for i in issues
                if i["kind"] not in ("volume_spike_day", "volume_drop_day")
            ],
        },
        artifact_id=record.id,
    )


# --------------------------------------------------------------------------- #
# Join discovery                                                              #
# --------------------------------------------------------------------------- #
def _name_similarity(a: str, b: str) -> float:
    ta, tb = name_tokens(a), name_tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return round(max(jaccard, difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()), 3)


def _key_columns(stats: list[ColumnStats]) -> list[ColumnStats]:
    """Columns that may act as join keys: identifier/category, text or integer, enough values."""
    return [
        c
        for c in stats
        if c.family in ("string", "integer")
        and classify_column(c).semantic_type in ("identifier", "category")
        and c.distinct >= _MIN_KEY_DISTINCT
    ]


def _overlap(con, left_table: str, right_table: str, strategy: str) -> tuple[int, int, int]:
    """Distinct normalised values on each side and in common."""
    nl = strategy_sql(strategy, "v")
    return con.execute(
        f"WITH a AS (SELECT DISTINCT {nl} AS n FROM {left_table}), b AS (SELECT DISTINCT {nl} AS n FROM {right_table}) "
        f"SELECT (SELECT count(*) FROM a), (SELECT count(*) FROM b), (SELECT count(*) FROM a JOIN b USING (n))"
    ).fetchall()[0]


@tool
def discover_join_candidates(left: str, right: str, seed: int | None = None) -> ToolResult:
    """Find and rank column pairs that could join two datasets.

    Each pair of key-like columns is scored by value overlap (after trying key
    normalisations such as prefix stripping), name similarity and type
    compatibility. The score margin between the best pair and the best
    *different* pair is reported for the join-confidence rules.

    Args:
        left: Primary dataset (rows to keep).
        right: Enrichment dataset.
        seed: Seed for the value samples used to detect prefixes.
    """
    try:
        lsrc, rsrc = resolve_source(left), resolve_source(right)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        _, lstats = compute_column_stats(con, lsrc.view, with_top=False, seed=seed or 0)
        _, rstats = compute_column_stats(con, rsrc.view, with_top=False, seed=seed or 0)
        lkeys, rkeys = _key_columns(lstats), _key_columns(rstats)
        tables: dict[tuple[str, str], str] = {}
        for side, src, cols in (("l", lsrc, lkeys), ("r", rsrc, rkeys)):
            for i, col in enumerate(cols):
                name = f"_k{side}{i}"
                con.execute(
                    f"CREATE TEMP TABLE {name} AS SELECT DISTINCT CAST({quote_ident(col.name)} AS VARCHAR) "
                    f"AS v FROM {src.view} WHERE {quote_ident(col.name)} IS NOT NULL"
                )
                tables[(side, col.name)] = name

        def sample(table: str) -> list[str]:
            return [
                r[0]
                for r in con.execute(
                    f"SELECT v FROM {table} ORDER BY hash(v, {int(seed or 0)}) LIMIT {_PREFIX_SAMPLE}"
                ).fetchall()
            ]

        def value_set(table: str) -> set[str]:
            return {r[0] for r in con.execute(f"SELECT v FROM {table} LIMIT 300000").fetchall()}

        candidates = []
        for lc in lkeys:
            lt = tables[("l", lc.name)]
            lsample = sample(lt)
            for rc in rkeys:
                if lsrc.name == rsrc.name and lc.name == rc.name:
                    continue  # a column trivially "joins" itself
                rt = tables[("r", rc.name)]
                prefixes = detect_prefixes(lsample, value_set(rt)) + detect_prefixes(sample(rt), value_set(lt))
                strategies = ["none", "lower_trim", "strip_leading_zeros"] + [prefix_strategy(p) for p in prefixes]
                best = None
                for strategy in dict.fromkeys(strategies):
                    n_left, n_right, common = _overlap(con, lt, rt, strategy)
                    left_cov = common / n_left if n_left else 0.0
                    right_cont = common / n_right if n_right else 0.0
                    # Prefer the simplest strategy unless another is clearly better.
                    if best is None or left_cov + right_cont > best[1] + best[2] + 1e-9:
                        best = (strategy, left_cov, right_cont, common)
                strategy, left_cov, right_cont, common = best
                if common == 0:
                    continue
                name_sim = _name_similarity(lc.name, rc.name)
                type_compat = 1.0 if lc.family == rc.family else 0.5
                score = 0.55 * left_cov + 0.25 * right_cont + 0.10 * name_sim + 0.10 * type_compat
                candidates.append(
                    {
                        "left_column": lc.name,
                        "right_column": rc.name,
                        "normalization": strategy,
                        "left_value_coverage": round(left_cov, 4),
                        "right_value_containment": round(right_cont, 4),
                        "common_values": int(common),
                        "name_similarity": name_sim,
                        "type_compatibility": type_compat,
                        "score": round(score, 4),
                    }
                )
    finally:
        con.close()
    candidates.sort(key=lambda c: -c["score"])
    margin = None
    if candidates:
        top = candidates[0]
        others = [
            c["score"]
            for c in candidates[1:]
            if (c["left_column"], c["right_column"]) != (top["left_column"], top["right_column"])
        ]
        margin = round(top["score"] - (others[0] if others else 0.0), 4)
    record = _store().save_json(
        f"join_candidates_{lsrc.name}_{rsrc.name}",
        {"left": lsrc.name, "right": rsrc.name, "candidates": candidates, "score_margin": margin},
        parents=[lsrc.name, rsrc.name],
    )
    return ToolResult.ok(
        {
            "left": lsrc.name,
            "right": rsrc.name,
            "candidates": candidates[:8],
            "score_margin": margin,
            "best": candidates[0] if candidates else None,
        },
        artifact_id=record.id,
        warnings=[] if candidates else ["No column pair shares values; the datasets may not be joinable."],
    )


def _temporal_validity_columns(stats: list[ColumnStats]) -> str | None:
    """Detect validity-interval columns (valid_from/valid_to, start/end...) on a dimension."""
    markers = {"valid", "validity", "effective", "from", "to", "start", "end", "expiry", "expires"}
    cols = [c.name for c in stats if c.family == "temporal" and name_tokens(c.name) & markers]
    return "/".join(cols) if len(cols) >= 2 else None


@tool
def test_join(
    left: str,
    right: str,
    keys: Annotated[list[tuple[str, str]], Field(min_length=1)],
    normalization: list[str] | None = None,
    candidates_artifact: str | None = None,
) -> ToolResult:
    """Measure a join: coverage, right-side key uniqueness, fan-out, cardinality and confidence.

    Duplicated right-side keys and unmatched primary rows are reported as
    quality issues. When `candidates_artifact` (from discover_join_candidates)
    is given, the score margin is included and the section-8 confidence rules
    decide whether the join can be auto-confirmed.

    Args:
        left: Primary dataset.
        right: Enrichment dataset.
        keys: Key pairs as [[left_column, right_column], ...].
        normalization: One strategy per key pair, or one for all (e.g. ["strip_prefix_plus351"]).
        candidates_artifact: Artifact id produced by discover_join_candidates.
    """
    norms = normalization or ["none"]
    if len(norms) not in (1, len(keys)):
        return ToolResult.error("normalization must have one entry or one per key pair")
    for n in norms:
        if not is_valid_strategy(n):
            return ToolResult.error(
                f"Unknown normalization '{n}'. Valid: {list(BASE_STRATEGIES)}, strip_prefix_<code> "
                "or strip_prefix:<prefix>"
            )
    norms = norms * len(keys) if len(norms) == 1 else norms
    try:
        lsrc, rsrc = resolve_source(left), resolve_source(right)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        lcols, rcols = dict(describe(con, lsrc.view)), dict(describe(con, rsrc.view))
        missing = [k for k, _ in keys if k not in lcols] + [k for _, k in keys if k not in rcols]
        if missing:
            return ToolResult.error(f"Unknown key columns: {missing}")
        try:
            lk = [
                f"{strategy_sql(n, quote_ident(a))} AS k{i}"
                for i, ((a, _), n) in enumerate(zip(keys, norms, strict=True))
            ]
            rk = [
                f"{strategy_sql(n, quote_ident(b))} AS k{i}"
                for i, ((_, b), n) in enumerate(zip(keys, norms, strict=True))
            ]
        except UnknownStrategyError as exc:
            return ToolResult.error(str(exc))
        names = [f"k{i}" for i in range(len(keys))]
        not_null = " AND ".join(f"{k} IS NOT NULL" for k in names)
        key_tuple = "(" + ", ".join(names) + ")" if len(names) > 1 else names[0]
        on = " AND ".join(f"L.{k} = R.{k}" for k in names)
        con.execute(f"CREATE TEMP TABLE L AS SELECT {', '.join(lk)} FROM {lsrc.view}")
        con.execute(f"CREATE TEMP TABLE R AS SELECT {', '.join(rk)} FROM {rsrc.view}")
        con.execute(f"CREATE TEMP TABLE RD AS SELECT DISTINCT {', '.join(names)} FROM R WHERE {not_null}")
        left_rows, left_null, left_distinct = con.execute(
            f"SELECT count(*), count(*) FILTER (WHERE NOT ({not_null})), count(DISTINCT {key_tuple}) "
            f"FILTER (WHERE {not_null}) FROM L"
        ).fetchall()[0]
        right_rows, right_keyed, right_distinct = con.execute(
            f"SELECT count(*), count(*) FILTER (WHERE {not_null}), (SELECT count(*) FROM RD) FROM R"
        ).fetchall()[0]
        matched_rows = con.execute(f"SELECT count(*) FROM L SEMI JOIN RD ON {on.replace('R.', 'RD.')}").fetchall()[0][0]
        joined_rows = con.execute(f"SELECT count(*) FROM L LEFT JOIN R ON {on}").fetchall()[0][0]
        unmatched_distinct = con.execute(
            f"SELECT count(*) FROM (SELECT DISTINCT {', '.join(names)} FROM L WHERE {not_null}) L "
            f"ANTI JOIN RD ON {on.replace('R.', 'RD.')}"
        ).fetchall()[0][0]
        dups = con.execute(
            f"SELECT {', '.join(names)}, count(*) AS n FROM R WHERE {not_null} GROUP BY ALL HAVING count(*) > 1 "
            f"ORDER BY n DESC, {', '.join(names)} LIMIT 1000"
        ).fetchall()
        _, rstats = compute_column_stats(con, rsrc.view, with_top=False, with_samples=False)
        # Sparse keys: are the null keys decided by another column (structural)?
        nulls = None
        if left_null and len(keys) == 1:
            _, lstats = compute_column_stats(con, lsrc.view, with_top=False, with_samples=True)
            nulls = null_structure(con, lsrc.view, keys[0][0], _segment_columns(lstats))
    finally:
        con.close()

    thresholds = _settings().config.join_confidence
    # Coverage over rows with a non-null key (section 10.3): a sparse key is not "unmatched".
    keyed_rows = left_rows - left_null
    coverage = matched_rows / keyed_rows if keyed_rows else 0.0
    null_key_share = left_null / left_rows if left_rows else 0.0
    nulls_structural = None if not left_null else bool(nulls and nulls["classification"] == "structural")
    right_uniqueness = right_distinct / right_keyed if right_keyed else 0.0
    fanout = joined_rows / left_rows if left_rows else 0.0  # null keys join nothing: no fan-out
    left_side = "1" if left_distinct == left_rows - left_null else "N"
    right_side = "1" if right_uniqueness >= thresholds.min_right_uniqueness else "N"
    cardinality = {("1", "1"): "1:1", ("N", "1"): "N:1", ("1", "N"): "1:N", ("N", "N"): "N:M"}[(left_side, right_side)]
    # Validity intervals version a dimension only when a key has several rows;
    # with unique keys the dates are plain attributes (e.g. a promotion's period).
    validity_columns = _temporal_validity_columns(rstats)
    temporal_validity = validity_columns if right_distinct < right_keyed else None

    margin, warnings = None, []
    if candidates_artifact:
        try:
            cand = _store().load_json(candidates_artifact)
        except (ArtifactNotFoundError, FileNotFoundError, ValueError):
            return ToolResult.error(f"Unknown candidates artifact '{candidates_artifact}'")
        ranked = cand.get("candidates", [])
        this = next((c for c in ranked if [c["left_column"], c["right_column"]] == list(keys[0])), None)
        if this is None or len(keys) > 1:
            warnings.append("Keys are not a single discovered candidate; score margin unavailable.")
        else:
            others = [
                c["score"]
                for c in ranked
                if (c["left_column"], c["right_column"]) != (this["left_column"], this["right_column"])
            ]
            margin = round(this["score"] - max(others, default=0.0), 4)
    elif len(keys) == 1:
        warnings.append("Pass candidates_artifact to include the score margin in the confidence rules.")

    assessment = assess_join(
        coverage,
        right_uniqueness,
        fanout,
        margin,
        len(keys),
        cardinality,
        temporal_validity,
        thresholds,
        null_key_share=null_key_share,
        nulls_structural=nulls_structural,
    )
    issues = []
    # Join-specific issues record which join they belong to, so the cleaning
    # plan only acts on issues of joins that end up in the spec.
    join_ref = {"join_left": lsrc.name, "join_right": rsrc.name, "join_keys": [list(k) for k in keys]}
    if dups:
        extra = sum(d[-1] - 1 for d in dups)
        issues.append(
            _issue(
                rsrc.name,
                ",".join(b for _, b in keys),
                "duplicate_keys",
                "high",
                keys=len(dups),
                extra_rows=int(extra),
                **join_ref,
            )
        )
    if left_null and not nulls_structural:
        issues.append(
            _issue(
                lsrc.name,
                ",".join(a for a, _ in keys),
                "null_keys",
                _severity(null_key_share),
                rows=int(left_null),
                share=round(null_key_share, 4),
                **join_ref,
            )
        )
    if matched_rows < keyed_rows:
        issues.append(
            _issue(
                lsrc.name,
                ",".join(a for a, _ in keys),
                "unmatched_rows",
                _severity(1 - coverage),
                rows=int(keyed_rows - matched_rows),
                distinct_keys=int(unmatched_distinct),
                share=round(1 - coverage, 4),
                **join_ref,
            )
        )
    metrics = {
        "left_rows": left_rows,
        "left_null_key_rows": left_null,
        "matched_rows": matched_rows,
        "coverage": round(coverage, 4),
        "left_distinct_keys": left_distinct,
        "key_coverage": round(1 - unmatched_distinct / left_distinct, 4) if left_distinct else 0.0,
        "unmatched_rows": keyed_rows - matched_rows,
        "null_key_share": round(null_key_share, 4),
        "nulls_structural": nulls_structural,
        "null_key_determined_by": nulls["determined_by"] if nulls else None,
        "unmatched_distinct_keys": unmatched_distinct,
        "right_rows": right_rows,
        "right_distinct_keys": right_distinct,
        "right_uniqueness": round(right_uniqueness, 5),
        "duplicate_keys": len(dups),
        "joined_rows": joined_rows,
        "fanout": round(fanout, 5),
        "cardinality": cardinality,
        "temporal_validity": temporal_validity,
        "validity_columns_on_unique_keys": validity_columns if validity_columns and not temporal_validity else None,
    }
    masker = _masker()
    record = _store().save_json(
        f"join_test_{lsrc.name}_{rsrc.name}",
        {
            "left": lsrc.name,
            "right": rsrc.name,
            "keys": keys,
            "normalization": norms,
            "metrics": metrics,
            "assessment": assessment.model_dump(),
            "duplicate_keys": [list(d) for d in dups],
            "issues": issues,
        },
        parents=[lsrc.name, rsrc.name, *([candidates_artifact] if candidates_artifact else [])],
        params={"keys": keys, "normalization": norms},
    )
    return ToolResult.ok(
        {
            "left": lsrc.name,
            "right": rsrc.name,
            "keys": keys,
            "normalization": norms,
            "metrics": metrics,
            "duplicate_key_examples": [[masker.mask_value(v) for v in d[:-1]] + [d[-1]] for d in dups[:3]],
            "assessment": assessment.model_dump(),
            "issues": [
                {"kind": i["kind"], **{k: v for k, v in i["detail"].items() if not k.startswith("join_")}}
                for i in issues
            ],
        },
        artifact_id=record.id,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# normalize_key                                                               #
# --------------------------------------------------------------------------- #
def _shape_sql(expr: str) -> str:
    """Value shape: digits -> 9, letters -> A (reveals format, not content)."""
    return f"regexp_replace(regexp_replace(CAST({expr} AS VARCHAR), '[0-9]', '9', 'g'), '[A-Za-z]', 'A', 'g')"


@tool
def normalize_key(dataset: str, column: str, strategy: str = "auto", match_against: str | None = None) -> ToolResult:
    """Test normalisation variants of a key column (prefixes, leading zeros, case, spaces).

    With strategy "auto" every base strategy plus any detected common prefix is
    tested. With `match_against` ("dataset.column") the match rate of each
    variant against that column is reported. Value shapes (9 = digit, A = letter)
    are shown instead of values.

    Args:
        dataset: Dataset name or table artifact id.
        column: Key column to analyse.
        strategy: "auto" or a strategy name such as "strip_prefix_plus351".
        match_against: Optional "dataset.column" to measure matches against.
    """
    if strategy != "auto" and not is_valid_strategy(strategy):
        return ToolResult.error(
            f"Unknown strategy '{strategy}'. Use 'auto', {list(BASE_STRATEGIES)}, strip_prefix_<code> "
            "or strip_prefix:<prefix>"
        )
    target = None
    if match_against:
        if "." not in match_against:
            return ToolResult.error("match_against must look like 'dataset.column'")
        target = match_against.rsplit(".", 1)
    try:
        src = resolve_source(dataset)
        tgt = resolve_source(target[0]) if target else None
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        cols = dict(describe(con, src.view))
        if column not in cols:
            return ToolResult.error(f"Column '{column}' not found. Columns: {list(cols)}")
        if tgt is not None and target[1] not in dict(describe(con, tgt.view)):
            return ToolResult.error(f"Column '{target[1]}' not found in '{tgt.name}'")
        q = quote_ident(column)
        con.execute(
            f"CREATE TEMP TABLE kv AS SELECT DISTINCT CAST({q} AS VARCHAR) AS v FROM {src.view} WHERE {q} IS NOT NULL"
        )
        n_distinct, plus, zeros, spaces, upper = con.execute(
            "SELECT count(*), avg(starts_with(v, '+')::INT), avg(starts_with(v, '0')::INT), "
            "avg((v <> trim(v))::INT), avg((v <> lower(v))::INT) FROM kv"
        ).fetchall()[0]
        shapes = con.execute(
            f"SELECT {_shape_sql('v')} s, count(*) n FROM kv GROUP BY 1 ORDER BY 2 DESC LIMIT 5"
        ).fetchall()
        # Common literal prefix: the longest first-k characters shared by >= 90% of values.
        common_prefix = None
        for k in range(1, 7):
            top = con.execute(
                f"SELECT left(v, {k}) p, count(*) n FROM kv GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
            ).fetchall()
            if top and top[0][1] / n_distinct >= 0.9 and all(ch.isalnum() or ch in "+_-.:/#" for ch in top[0][0]):
                common_prefix = top[0][0]
            else:
                break
        if tgt is not None:
            tq = quote_ident(target[1])
            con.execute(
                f"CREATE TEMP TABLE tv AS SELECT DISTINCT CAST({tq} AS VARCHAR) AS v FROM {tgt.view} "
                f"WHERE {tq} IS NOT NULL"
            )
            sample = [r[0] for r in con.execute(f"SELECT v FROM kv ORDER BY hash(v) LIMIT {_PREFIX_SAMPLE}").fetchall()]
            tset = {r[0] for r in con.execute("SELECT v FROM tv LIMIT 300000").fetchall()}
            detected = detect_prefixes(sample, tset)
        else:
            detected = [common_prefix] if common_prefix else []
        if strategy == "auto":
            strategies = list(BASE_STRATEGIES) + [prefix_strategy(p) for p in detected]
        else:
            strategies = [strategy]
        results = []
        for name in dict.fromkeys(strategies):
            expr = strategy_sql(name, "v")
            after, changed = con.execute(f"SELECT count(DISTINCT {expr}), avg(({expr} <> v)::INT) FROM kv").fetchall()[
                0
            ]
            entry = {"strategy": name, "distinct_after": after, "changed_share": round(changed or 0, 4)}
            example = con.execute(
                f"SELECT {_shape_sql('v')}, {_shape_sql(expr)} FROM kv WHERE {expr} <> v LIMIT 1"
            ).fetchall()
            if example:
                entry["shape_example"] = f"{example[0][0]} -> {example[0][1]}"
            if tgt is not None:
                common = con.execute(
                    f"SELECT count(*) FROM (SELECT DISTINCT {expr} n FROM kv) a JOIN "
                    f"(SELECT DISTINCT {expr} n FROM tv) b USING (n)"
                ).fetchall()[0][0]
                entry["match_rate"] = round(common / n_distinct, 4) if n_distinct else 0.0
            results.append(entry)
    finally:
        con.close()
    if tgt is not None:
        results.sort(key=lambda r: (-r["match_rate"], r["strategy"] != "none"))
    return ToolResult.ok(
        {
            "dataset": src.name,
            "column": column,
            "distinct_values": n_distinct,
            "value_shapes": [[s, n] for s, n in shapes],
            "share_with_plus": round(plus or 0, 4),
            "share_leading_zero": round(zeros or 0, 4),
            "share_untrimmed": round(spaces or 0, 4),
            "share_with_uppercase": round(upper or 0, 4),
            "common_prefix": common_prefix,
            "detected_prefixes": detected,
            "strategies": results,
            "recommended": results[0]["strategy"] if tgt is not None and results else None,
        }
    )


# --------------------------------------------------------------------------- #
# correlation_summary                                                         #
# --------------------------------------------------------------------------- #
@tool
def correlation_summary(dataset: str) -> ToolResult:
    """Find redundant columns: strongly correlated measures and functionally dependent categories.

    Args:
        dataset: Dataset name or table artifact id.
    """
    try:
        with _open(dataset) as (con, source):
            _, stats = compute_column_stats(con, source.view, with_top=False, with_samples=True)
            types = {c.name: classify_column(c).semantic_type for c in stats}
            measures = [c.name for c in stats if types[c.name] == "measure"][:30]
            numeric_pairs = []
            pairs = [(a, b) for i, a in enumerate(measures) for b in measures[i + 1 :]]
            if pairs:
                exprs = ", ".join(f"corr({quote_ident(a)}, {quote_ident(b)})" for a, b in pairs)
                values = con.execute(f"SELECT {exprs} FROM {source.view}").fetchall()[0]
                numeric_pairs = sorted(
                    (
                        {"a": a, "b": b, "pearson": round(v, 4)}
                        for (a, b), v in zip(pairs, values, strict=True)
                        if v is not None
                    ),
                    key=lambda p: -abs(p["pearson"]),
                )
            cats = [c for c in stats if types[c.name] in ("category", "identifier") and 1 < c.distinct <= 5000][:15]
            dependencies = []
            cat_pairs = [(a, b) for i, a in enumerate(cats) for b in cats[i + 1 :]]
            if cat_pairs:
                exprs = ", ".join(
                    f"count(DISTINCT ({quote_ident(a.name)}, {quote_ident(b.name)}))" for a, b in cat_pairs
                )
                counts = con.execute(f"SELECT {exprs} FROM {source.view}").fetchall()[0]
                for (a, b), n in zip(cat_pairs, counts, strict=True):
                    if n == a.distinct and n == b.distinct:
                        dependencies.append({"a": a.name, "b": b.name, "relation": "one_to_one"})
                    elif n == a.distinct:
                        dependencies.append({"a": a.name, "b": b.name, "relation": f"{a.name} determines {b.name}"})
                    elif n == b.distinct:
                        dependencies.append({"a": a.name, "b": b.name, "relation": f"{b.name} determines {a.name}"})
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    redundant = [p for p in numeric_pairs if abs(p["pearson"]) >= 0.9]
    record = _store().save_json(
        f"correlations_{source.name}",
        {"dataset": source.name, "numeric_pairs": numeric_pairs, "dependencies": dependencies},
        parents=[source.name],
    )
    return ToolResult.ok(
        {
            "dataset": source.name,
            "redundant_measures": redundant,
            "top_correlations": numeric_pairs[:5],
            "category_dependencies": dependencies[:10],
        },
        artifact_id=record.id,
    )


# --------------------------------------------------------------------------- #
# compile_quality_issues                                                      #
# --------------------------------------------------------------------------- #
@tool
def compile_quality_issues(artifact_ids: Annotated[list[str], Field(min_length=1)]) -> ToolResult:
    """Merge the quality issues found by profiler tools into one artifact for the preparer.

    When classify_nulls covered a column, its classification replaces the generic
    "nulls" issue of profile_table: structural nulls are not a problem, and only
    the unexpected nulls remain an issue.

    Args:
        artifact_ids: Artifacts of profile_table, infer_grain, check_temporal_coverage, test_join and the
            integrity checks (discover_arithmetic_invariants, check_referential_integrity, find_duplicate_keys,
            classify_nulls, infer_sign_convention, check_reciprocity).
    """
    store = _store()
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    by_nulls: dict[tuple[Any, ...], dict[str, Any]] = {}  # issues of classify_nulls
    missing = []
    for artifact_id in artifact_ids:
        try:
            content = store.load_json(artifact_id)
        except (ArtifactNotFoundError, FileNotFoundError, ValueError):
            missing.append(artifact_id)
            continue
        for issue in content.get("issues", []) if isinstance(content, dict) else []:
            detail = issue.get("detail", {})
            key = (
                issue["dataset"],
                issue.get("column"),
                issue["kind"],
                detail.get("file"),
                detail.get("partition"),
                detail.get("date"),
                str(detail.get("join_keys")),
                detail.get("join_right"),
                detail.get("invariant"),
                detail.get("category"),
            )
            target = by_nulls if issue.get("source_tool") == "classify_nulls" else merged
            target.setdefault(key, {**issue, "artifact": artifact_id})
    if missing:
        return ToolResult.error(f"Unknown artifacts: {missing}")
    classified = {(i["dataset"], i.get("column")) for i in by_nulls.values()}
    merged = {
        k: v for k, v in merged.items() if not (v["kind"] == "nulls" and (v["dataset"], v.get("column")) in classified)
    }
    for key, issue in by_nulls.items():
        merged.setdefault(key, issue)
    issues = sorted(
        merged.values(),
        key=lambda i: ({"high": 0, "medium": 1, "low": 2}[i["severity"]], i["dataset"], str(i.get("column"))),
    )
    for n, issue in enumerate(issues, 1):
        issue["id"] = f"Q{n:03d}"
    record = store.save_json("quality_issues", {"issues": issues}, parents=list(artifact_ids))
    return ToolResult.ok(
        {
            "count": len(issues),
            "issues": [
                {
                    "id": i["id"],
                    "dataset": i["dataset"],
                    "column": i.get("column"),
                    "kind": i["kind"],
                    "severity": i["severity"],
                }
                for i in issues
            ],
        },
        artifact_id=record.id,
    )


# --------------------------------------------------------------------------- #
# propose_spec                                                                #
# --------------------------------------------------------------------------- #
def _load_kind(store: ArtifactStore, artifact_id: str, required_key: str) -> dict[str, Any]:
    """Load a JSON artifact and check it is of the expected kind."""
    content = store.load_json(artifact_id)
    if not isinstance(content, dict) or required_key not in content:
        raise ValueError(f"Artifact '{artifact_id}' is not the expected kind (missing '{required_key}')")
    return content


@tool
def propose_spec(
    join_tests: list[str],
    grain: str,
    entities: str,
    semantic_types: Annotated[list[str], Field(min_length=1)],
    temporal: str | None = None,
    quality_issues: str | None = None,
    context_notes: str = "",
    checks: list[str] | None = None,
) -> ToolResult:
    """Write the profiler's proposal into spec.yaml, built only from evidence artifacts.

    Joins are written with status "proposed" and their test metrics; grain,
    entity (with alternatives), time column and column semantic types come
    from the given artifacts. Nothing is confirmed here: that is control 1.

    Args:
        join_tests: Artifact ids from test_join, one per proposed join (may be empty).
        grain: Artifact id from infer_grain on the primary dataset.
        entities: Artifact id from suggest_entities on the primary dataset.
        semantic_types: Artifact ids from infer_semantic_types (one per dataset).
        temporal: Artifact id from check_temporal_coverage; its time column becomes the spec time column.
        quality_issues: Artifact id from compile_quality_issues.
        context_notes: Short notes from the datasets' README files, if any.
        checks: Artifact ids of the integrity checks to record: discover_arithmetic_invariants (invariants),
            check_referential_integrity and check_reciprocity (references), classify_nulls (structural nulls).
    """
    store = _store()
    try:
        spec = read_spec()
        joins = []
        for artifact_id in join_tests:
            test = _load_kind(store, artifact_id, "assessment")
            m = test["metrics"]
            norms = test["normalization"]
            joins.append(
                {
                    "left": test["left"],
                    "right": test["right"],
                    "keys": test["keys"],
                    "key_normalization": [] if norms == ["none"] else norms,
                    "type": "left",
                    "cardinality": m["cardinality"],
                    "coverage": m["coverage"],
                    "fanout": m["fanout"],
                    "right_uniqueness": m["right_uniqueness"],
                    "temporal_validity": m["temporal_validity"],
                    "confidence": test["assessment"]["confidence"],
                    "status": "proposed",
                    "evidence_artifact": artifact_id,
                }
            )
        grain_doc = _load_kind(store, grain, "candidates")
        if not grain_doc["candidates"]:
            return ToolResult.error(f"Grain artifact '{grain}' has no candidates.")
        ent_doc = _load_kind(store, entities, "candidates")
        if not ent_doc["candidates"]:
            return ToolResult.error(f"Entities artifact '{entities}' has no candidates.")
        columns: dict[str, dict[str, str]] = {}
        typed = [(_load_kind(store, a, "columns"), a) for a in semantic_types]
        all_names = [name for doc, _ in typed for name in doc["columns"]]
        for doc, _ in typed:
            for name, info in doc["columns"].items():
                key = name if all_names.count(name) == 1 else f"{doc['dataset']}.{name}"
                columns[key] = {"dataset": doc["dataset"], "semantic_type": info["semantic_type"]}
        evidence = {"grain": grain, "entity": entities, **{f"semantic_types:{doc['dataset']}": a for doc, a in typed}}
        patch: dict[str, Any] = {
            "joins": joins,
            "grain": {"dataset": grain_doc["dataset"], "columns": grain_doc["candidates"][0]["columns"]},
            "entity": {
                "column": ent_doc["candidates"][0]["column"],
                "alternatives": [c["column"] for c in ent_doc["candidates"][1:4]],
            },
            "columns": columns,
            "evidence_artifacts": evidence,
        }
        if spec.entity is not None and spec.entity.source == "user":
            patch.pop("entity")  # fixed by the user with --entity: never overwritten
        if temporal:
            patch["time_column"] = _load_kind(store, temporal, "time_column")["time_column"]
            evidence["temporal"] = temporal
        if quality_issues:
            _load_kind(store, quality_issues, "issues")
            evidence["quality_issues"] = quality_issues
        if context_notes:
            patch["context_notes"] = context_notes[:2000]
        ignored: list[str] = []
        if checks:
            references, invariants, structural = [], [], {}
            for artifact_id in checks:
                doc = store.load_json(artifact_id)
                if not isinstance(doc, dict):
                    return ToolResult.error(f"Artifact '{artifact_id}' is not an integrity check.")
                if "invariants" in doc:
                    invariants += [
                        {
                            "dataset": doc["dataset"],
                            "expression": i["expression"],
                            "holds_share": i["holds_share"],
                            "violations": i["violations"],
                            "evidence_artifact": artifact_id,
                        }
                        for i in doc["invariants"]
                        if i["kind"] != "non_negative"
                    ]
                    evidence[f"invariants:{doc['dataset']}"] = artifact_id
                elif "reference" in doc:
                    ref = doc["reference"]
                    nulls = doc.get("nulls") or {}
                    references.append(
                        {
                            **{k: ref[k] for k in ("dataset", "column", "ref_dataset", "ref_column")},
                            "kind": ref.get("kind", "reference"),
                            "orphan_rows": doc.get("orphan_rows", doc.get("without_reciprocal", 0)),
                            "orphan_share": doc.get("orphan_share", doc.get("share", 0.0)),
                            "null_share": doc.get("null_share", 0.0),
                            "nulls_structural": None if not nulls else nulls.get("classification") == "structural",
                            "evidence_artifact": artifact_id,
                        }
                    )
                    evidence[f"reference:{ref['dataset']}.{ref['column']}"] = artifact_id
                elif "nulls" in doc:
                    structural |= {
                        f"{doc['dataset']}.{column}": info["determined_by"]
                        for column, info in doc["nulls"].items()
                        if info["classification"] != "not_structural"
                    }
                    evidence[f"nulls:{doc['dataset']}"] = artifact_id
                elif {"candidates", "checks", "inversions"} & set(doc):
                    # Other integrity checks (self-references, duplicate keys, signs) feed the
                    # quality issues, not the spec: accepted and ignored here.
                    ignored.append(artifact_id)
                else:
                    return ToolResult.error(f"Artifact '{artifact_id}' is not an integrity check.")
            patch |= {"references": references, "invariants": invariants, "structural_nulls": structural}
        primary = spec.primary()
        if primary and patch["grain"]["dataset"] != primary.name:
            return ToolResult.error(f"Grain must be inferred on the primary dataset '{primary.name}'.")
        new_spec = patch_spec(patch)
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    except (ArtifactNotFoundError, FileNotFoundError) as exc:
        return ToolResult.error(f"Unknown artifact: {exc}")
    except ValueError as exc:
        return ToolResult.error(f"Proposal rejected: {exc}"[:1500])
    warnings = [f"Not recorded in the spec (their issues are in the quality issues): {ignored}"] if ignored else []
    return ToolResult.ok({"proposed": sorted(patch), "spec": new_spec.compact_summary()}, warnings=warnings)
