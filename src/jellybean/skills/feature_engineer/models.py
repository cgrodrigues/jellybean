# ruff: noqa: E501  (SQL templates are kept on one line each, to read as SQL)
"""Feature models for ledgers of typed events (``requirements_2.md`` section 10.4).

Each tool extends the per-entity feature table (``features``) with one family of
features, computed in DuckDB from the event table, and records every new
feature in the dictionary with its formula:

| tool                               | features                                                              |
| ---------------------------------- | --------------------------------------------------------------------- |
| build_category_features            | per value of a type column: count, sum and weight of a measure;       |
|                                    | ratios between the sums of pairs of values                            |
| build_linked_event_features        | rows referenced by others (share, time to the reference), and         |
|                                    | references without their expected response                           |
| build_sequence_features            | time between consecutive events of different types, inactivity        |
|                                    | after the last event                                                  |
| build_conformity_features          | observed value vs a value expected from the reference tables, with a  |
|                                    | formula proposed by the agent (one SQL expression, validated)         |
| build_interaction_graph            | graph of entities linked by a counterpart column (who transacts with  |
|                                    | whom): degree, component size, neighbour degree (stars)               |

Category values are normalised (lower case, trimmed), so spelling variants do
not split a feature. Nothing here knows a domain: the columns, the reference
and the expected-value formula come from the spec and the agent.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

import duckdb
import pandas as pd
from pydantic import Field

from jellybean.runtime.context import current_run
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.skills._shared.artifacts import ArtifactNotFoundError
from jellybean.skills._shared.catalog import SourceNotFoundError, connect, quote_ident, resolve_source, sql_escape
from jellybean.skills._shared.column_stats import describe
from jellybean.skills._shared.detection import component_table
from jellybean.skills._shared.spec_tools import SpecMissingError, patch_spec, read_spec
from jellybean.skills._shared.sql_guard import SQLRejectedError, validate_select
from jellybean.skills.feature_engineer._common import _feature_name, _load_table, _store, latest_features

_MAX_LEVELS = 16
_MAX_GRAPH_EDGES = 50_000_000  # distinct entity pairs an interaction graph can hold in memory
_MIN_SUPPORT = 20
# A response is "expected" when at least this share of the population has it.
_EXPECTED_SHARE = 0.5


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _norm(column: str, alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return f"lower(trim(CAST({prefix}{quote_ident(column)} AS VARCHAR)))"


def _self_reference(spec: Any, columns: dict[str, str]) -> tuple[str, str] | None:
    """(referencing column, target column) of the primary dataset's self-reference, if any."""
    primary = spec.primary().name if spec.primary() else None
    for ref in spec.references:
        if (
            ref.kind == "reference"
            and ref.dataset == ref.ref_dataset == primary
            and ref.column in columns
            and ref.ref_column in columns
        ):
            return ref.column, ref.ref_column
    return None


def _levels(con: duckdb.DuckDBPyConnection, column: str, limit: int) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            f"SELECT {_norm(column)} v FROM ev WHERE {quote_ident(column)} IS NOT NULL GROUP BY 1 "
            f"ORDER BY count(*) DESC, 1 LIMIT {int(limit)}"
        ).fetchall()
    ]


def _extend(
    features: str, entity: str, new: pd.DataFrame, entries: list[dict[str, str]], parents: list[str], fill: float | None
) -> str:
    """Add the columns of ``new`` to the feature table ``features``; record them in the dictionary.

    Returns the new feature-table version (also recorded in the spec).
    """
    store = _store()
    requested = features
    features, rebased = latest_features(features)
    base = _load_table(features)
    if entity not in base.columns:
        raise ValueError(f"Feature table '{features}' has no column '{entity}'.")
    new = new.copy()
    new[entity] = new[entity].astype(base[entity].dtype)
    names = [c for c in new.columns if c != entity]
    merged = base.drop(columns=[c for c in names if c in base.columns]).merge(new, on=entity, how="left")
    if fill is not None:
        merged[names] = merged[names].fillna(fill)
    params = dict(store.get(features).params) | ({"rebased_from": requested} if rebased else {})
    record = store.save_dataframe(
        f"features_{_feature_name(entity)}", merged, parents=[features, *parents], params=params
    )
    spec = read_spec()
    dictionary = store.load_json(spec.features.dictionary)["features"] if spec.features.dictionary else []
    dictionary = [d for d in dictionary if d["name"] not in set(names)] + entries
    dict_id = store.save_json("feature_dictionary", {"entity": entity, "features": dictionary}, parents=[record.id]).id
    patch_spec({"features": {"table": record.id, "dictionary": dict_id, "scaled": None, "embedding": None}})
    return record.id


def _open_events(
    input: str, entity: str, needed: list[str | None], spec: Any = None, expressions: str | None = None
) -> tuple[duckdb.DuckDBPyConnection, dict[str, str]]:
    """Connection with a temp table ``ev`` (rows with an entity); checks the columns exist.

    ``ev`` holds only the columns the tool reads, not the whole event table: the
    entity, ``needed``, the self-reference of ``spec`` (when given) and every
    column named in ``expressions`` (agent-written SQL; ``ref_<column>`` counts
    as ``<column>``). The returned schema is the schema of ``ev``.
    """
    source = resolve_source(input)
    con = connect()
    try:
        schema = dict(describe(con, source.view))
        missing = [c for c in [entity, *needed] if c and c not in schema]
        if missing:
            raise ValueError(f"Unknown columns: {missing}. Columns: {list(schema)}")
        keep = {entity, *(c for c in needed if c)}
        if spec is not None:
            keep |= set(_self_reference(spec, schema) or ())
        if expressions is not None:
            # A name inside the text may be a column: keeping one too many is harmless.
            keep |= {c for c in schema if c in expressions}
        columns = [c for c in schema if c in keep]
        con.execute(
            f"CREATE TEMP TABLE ev AS SELECT {', '.join(quote_ident(c) for c in columns)} FROM {source.view} "
            f"WHERE {quote_ident(entity)} IS NOT NULL"
        )
    except BaseException:
        con.close()  # never leave the connection open when the copy fails (e.g. out of memory)
        raise
    return con, {c: schema[c] for c in columns}


def _safe_expression(expr: str, label: str) -> str:
    """Validate an agent-proposed one-line SQL expression with the run_sql guard."""
    text = expr.strip()
    if not text or len(text) > 600 or "\n" in text or ";" in text:
        raise SQLRejectedError(f"{label} must be one SQL expression on one line (at most 600 characters).")
    if re.search(r"\bselect\b|\bfrom\b", text, re.IGNORECASE):
        raise SQLRejectedError(f"{label} must be an expression over the row's columns, not a query.")
    validate_select(f"SELECT ({text}) AS v FROM t", forbidden_paths=[current_run().settings.ground_truth_root])
    return text


def _result(entity: str, table: str, entries: list[dict[str, str]], **extra: Any) -> ToolResult:
    rebased = _store().get(table).params.get("rebased_from")
    warnings = (
        [
            f"'{rebased}' is not the latest feature table: the features were added to the latest one, so no columns are lost."
        ]
        if rebased
        else []
    )
    return ToolResult.ok(
        {"features_table": table, "new_features": len(entries), "examples": [e["name"] for e in entries[:12]], **extra},
        artifact_id=table,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Model 1 + 2: measure by category value, ratios between values               #
# --------------------------------------------------------------------------- #
@tool
def build_category_features(
    input: str,
    entity: str,
    category: str,
    measure: str,
    features: str,
    ratio_pairs: list[tuple[str, str]] | None = None,
    max_levels: Annotated[int, Field(ge=2, le=30)] = _MAX_LEVELS,
) -> ToolResult:
    """Per value of a low-cardinality column (e.g. a type): count, sum and relative weight of a measure, plus ratios.

    For each of the `max_levels` most frequent values: the entity's number of
    rows, the sum of `measure` and its weight (share of the entity's absolute
    `measure`). Ratios are the sums of two values divided (|sum a| / |sum b|):
    give `ratio_pairs` as [[a, b], ...], or leave it empty to derive the pairs
    from the dataset's self-reference in the spec (rows of type a referring to
    rows of type b, e.g. an answer and what it answers).

    Args:
        input: Event table (the enriched artifact).
        entity: Entity column.
        category: Low-cardinality column whose values split the rows (e.g. a type).
        measure: Numeric column to aggregate.
        features: Feature table to extend.
        ratio_pairs: Pairs of category values [[numerator, denominator], ...] (optional).
        max_levels: Maximum number of category values.
    """
    try:
        spec = read_spec()
        con, schema = _open_events(input, entity, [category, measure], spec=spec)
    except (SpecMissingError, SourceNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    try:
        e, m, c = quote_ident(entity), quote_ident(measure), _norm(category)
        levels = _levels(con, category, max_levels)
        pairs = [(a.strip().lower(), b.strip().lower()) for a, b in ratio_pairs or []]
        ref = _self_reference(spec, schema)
        if not ratio_pairs and ref:
            r, target = (quote_ident(x) for x in ref)
            pairs = [
                (a, b)
                for a, b, n in con.execute(
                    f"SELECT {_norm(category, 'y')} a, {_norm(category, 'x')} b, count(*) n FROM ev y "
                    f"JOIN ev x ON y.{r} = x.{target} GROUP BY 1, 2 HAVING n >= {_MIN_SUPPORT} ORDER BY n DESC LIMIT 12"
                ).fetchall()
            ]
        unknown = sorted({v for p in pairs for v in p} - set(levels))
        if ratio_pairs and unknown:
            return ToolResult.error(f"Unknown or rare values in ratio_pairs: {unknown}. Values: {levels}")
        pairs = [p for p in pairs if set(p) <= set(levels)]
        select, entries = [], []
        for v in levels:
            lit = f"'{sql_escape(v)}'"
            n, s_, w = (_feature_name("cat", category, v, k) for k in ("n", f"{measure}_sum", f"{measure}_weight"))
            select += [
                f"count(*) FILTER (WHERE {c} = {lit}) AS {quote_ident(n)}",
                f"coalesce(sum({m}) FILTER (WHERE {c} = {lit}), 0) AS {quote_ident(s_)}",
                f"coalesce(sum(abs({m})) FILTER (WHERE {c} = {lit}) / nullif(sum(abs({m})), 0), 0) AS {quote_ident(w)}",
            ]
            entries += [
                {
                    "name": n,
                    "source": category,
                    "template": "category_measure",
                    "formula": f"rows with {category} = {v}",
                },
                {
                    "name": s_,
                    "source": measure,
                    "template": "category_measure",
                    "formula": f"sum of {measure} where {category} = {v}",
                },
                {
                    "name": w,
                    "source": measure,
                    "template": "category_measure",
                    "formula": f"|{measure}| where {category} = {v} / total |{measure}|",
                },
            ]
        for a, b in pairs:
            name = _feature_name("ratio", category, a, "over", b)
            select.append(
                f"abs(sum({m}) FILTER (WHERE {c} = '{sql_escape(a)}')) / "
                f"nullif(abs(sum({m}) FILTER (WHERE {c} = '{sql_escape(b)}')), 0) AS {quote_ident(name)}"
            )
            entries.append(
                {
                    "name": name,
                    "source": measure,
                    "template": "category_ratio",
                    "formula": f"|sum {measure} of {a}| / |sum {measure} of {b}|",
                }
            )
        new = con.execute(f"SELECT {e}, {', '.join(select)} FROM ev GROUP BY {e}").df()
    finally:
        con.close()
    try:
        table = _extend(features, entity, new, entries, [input], fill=None)
    except (ArtifactNotFoundError, FileNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    return _result(entity, table, entries, levels=levels, ratio_pairs=[list(p) for p in pairs])


# --------------------------------------------------------------------------- #
# Model 3: events linked by reference                                         #
# --------------------------------------------------------------------------- #
@tool
def build_linked_event_features(
    input: str,
    entity: str,
    category: str,
    time_column: str,
    features: str,
    reference_column: str | None = None,
    target_column: str | None = None,
) -> ToolResult:
    """Features of rows linked by a reference (a row pointing to another row of the same table).

    With the self-reference of the spec (or `reference_column` -> `target_column`):
    for each pair (type of the referenced row, type of the referring row) seen at
    least 20 times: share of the entity's rows of the first type referenced by the
    second, and the median time between them. Expected responses are learnt from
    the population: a type answered by another in at least half of the cases
    (e.g. a request and its settlement), and a response expected after a sibling
    (when a row is referred to by A and B, B is usually answered by C). The
    entity's missing expected responses become features.

    Args:
        input: Event table (the enriched artifact).
        entity: Entity column.
        category: Type column of the rows.
        time_column: Event time.
        features: Feature table to extend.
        reference_column: Referencing column (default: from spec.references).
        target_column: Referenced column (default: from spec.references).
    """
    try:
        spec = read_spec()
        con, schema = _open_events(input, entity, [category, time_column, reference_column, target_column], spec=spec)
    except (SpecMissingError, SourceNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    try:
        ref = (reference_column, target_column) if reference_column and target_column else _self_reference(spec, schema)
        if not ref:
            return ToolResult.error("No self-reference in the spec: pass reference_column and target_column.")
        r, target = (quote_ident(x) for x in ref)
        e, t = quote_ident(entity), quote_ident(time_column)
        con.execute(
            f"CREATE TEMP TABLE rows_ AS SELECT DISTINCT {target} AS id, {r} AS ref, {_norm(category)} AS ty, {e} AS ent, epoch({t}) AS ts FROM ev"
        )
        con.execute(
            "CREATE TEMP TABLE links AS SELECT x.id AS xid, x.ty AS tx, x.ent AS ent, y.id AS yid, y.ty AS ty, "
            "y.ts - x.ts AS lag FROM rows_ y JOIN rows_ x ON y.ref = x.id"
        )
        pairs = con.execute(
            f"SELECT tx, ty, count(*) n FROM links GROUP BY 1, 2 HAVING n >= {_MIN_SUPPORT} ORDER BY n DESC LIMIT 16"
        ).fetchall()
        totals = dict(con.execute("SELECT ty, count(*) FROM rows_ GROUP BY 1").fetchall())
        select, entries, rules = [], [], []
        joins = []
        for i, (tx, ty, n) in enumerate(pairs):
            share, lag = _feature_name("link", tx, "by", ty, "share"), _feature_name("link", tx, "by", ty, "lag_p50_s")
            con.execute(
                f"CREATE TEMP TABLE l{i} AS SELECT ent, count(DISTINCT xid) AS k, median(lag) AS lag FROM links "
                f"WHERE tx = '{sql_escape(tx)}' AND ty = '{sql_escape(ty)}' GROUP BY ent"
            )
            joins.append(f"LEFT JOIN l{i} ON l{i}.ent = b.ent")
            select += [
                f'coalesce(l{i}.k, 0) / nullif(b."{sql_escape(tx)}", 0) AS {quote_ident(share)}',
                f"l{i}.lag AS {quote_ident(lag)}",
            ]
            entries += [
                {
                    "name": share,
                    "source": ",".join(ref),
                    "template": "linked",
                    "formula": f"share of the entity's {tx} rows referred to by a {ty} row",
                },
                {
                    "name": lag,
                    "source": time_column,
                    "template": "linked_lag",
                    "formula": f"median seconds from a {tx} row to the {ty} row referring to it",
                },
            ]
            if n >= _EXPECTED_SHARE * totals.get(tx, 0):
                rules.append(
                    {"kind": "answer", "type": tx, "expected": ty, "population_share": round(n / totals[tx], 4)}
                )
        # Responses expected after a sibling: x referred to by A and B -> B referred to by C.
        siblings = con.execute(
            # The sibling side is reduced to (referenced row, type) first: a row referred to by
            # k rows costs k x types pairs, not k x k.
            f"""WITH sib AS (SELECT DISTINCT a.ty AS ta, b.ty AS tb, b.yid AS bid
                             FROM (SELECT DISTINCT xid, ty FROM links) a JOIN links b
                             ON a.xid = b.xid AND a.ty <> b.ty),
                     ans AS (SELECT DISTINCT s.ta, s.tb, s.bid, c.ty AS tc FROM sib s JOIN links c ON c.xid = s.bid)
                SELECT a.ta, a.tb, a.tc, count(*) n,
                       (SELECT count(DISTINCT bid) FROM sib s WHERE s.ta = a.ta AND s.tb = a.tb) total
                FROM ans a GROUP BY 1, 2, 3 HAVING n >= {_MIN_SUPPORT}"""
        ).fetchall()
        for ta, tb, tc, n, total in siblings:
            if n >= _EXPECTED_SHARE * total:
                rules.append(
                    {
                        "kind": "after_sibling",
                        "sibling": ta,
                        "type": tb,
                        "expected": tc,
                        "population_share": round(n / total, 4),
                    }
                )
        for i, rule in enumerate(rules):
            name = _feature_name("link", rule["type"], "without", rule["expected"])
            if rule["kind"] == "answer":
                sql = (
                    f"SELECT x.ent, count(*) AS k FROM rows_ x WHERE x.ty = '{sql_escape(rule['type'])}' AND NOT EXISTS "
                    f"(SELECT 1 FROM links l WHERE l.xid = x.id AND l.ty = '{sql_escape(rule['expected'])}') GROUP BY 1"
                )
                formula = f"{rule['type']} rows never referred to by a {rule['expected']} row ({rule['population_share']:.0%} of the population are)"
            else:
                name = _feature_name("link", rule["type"], "after", rule["sibling"], "without", rule["expected"])
                sql = f"""SELECT b.ent, count(DISTINCT b.yid) AS k FROM links b
                          WHERE b.ty = '{sql_escape(rule["type"])}'
                          AND EXISTS (SELECT 1 FROM links a WHERE a.xid = b.xid AND a.ty = '{sql_escape(rule["sibling"])}')
                          AND NOT EXISTS (SELECT 1 FROM links c WHERE c.xid = b.yid AND c.ty = '{sql_escape(rule["expected"])}')
                          GROUP BY 1"""
                formula = (
                    f"{rule['type']} rows whose origin also has a {rule['sibling']} row, never referred to by a "
                    f"{rule['expected']} row ({rule['population_share']:.0%} of the population are)"
                )
            con.execute(f"CREATE TEMP TABLE r{i} AS {sql}")
            joins.append(f"LEFT JOIN r{i} ON r{i}.ent = b.ent")
            select.append(f"coalesce(r{i}.k, 0) AS {quote_ident(name)}")
            entries.append({"name": name, "source": ",".join(ref), "template": "linked", "formula": formula})
        con.execute("CREATE TEMP TABLE referenced AS SELECT DISTINCT xid FROM links")
        types = sorted({tx for tx, _, _ in pairs})
        base_cols = ", ".join(f"count(*) FILTER (WHERE ty = '{sql_escape(v)}') AS \"{sql_escape(v)}\"" for v in types)
        con.execute(
            f"CREATE TEMP TABLE b AS SELECT ent, avg((id IN (SELECT xid FROM referenced))::DOUBLE) AS link_referenced_share"
            f"{', ' + base_cols if base_cols else ''} FROM rows_ GROUP BY ent"
        )
        entries.insert(
            0,
            {
                "name": "link_referenced_share",
                "source": ",".join(ref),
                "template": "linked",
                "formula": "share of the entity's rows referred to by another row",
            },
        )
        new = con.execute(
            f"SELECT b.ent AS {e}, b.link_referenced_share{''.join(', ' + s for s in select)} FROM b {' '.join(joins)}"
        ).df()
    finally:
        con.close()
    try:
        table = _extend(features, entity, new, entries, [input], fill=None)
    except (ArtifactNotFoundError, FileNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    return _result(
        entity,
        table,
        entries,
        reference=list(ref),
        pairs=[[a, b, int(n)] for a, b, n in pairs],
        expected_responses=rules,
    )


# --------------------------------------------------------------------------- #
# Model 4: temporal sequences                                                 #
# --------------------------------------------------------------------------- #
@tool
def build_sequence_features(
    input: str,
    entity: str,
    category: str,
    time_column: str,
    features: str,
    pairs: list[tuple[str, str]] | None = None,
    fast_seconds: Annotated[int, Field(ge=1)] = 3600,
    max_pairs: Annotated[int, Field(ge=1, le=30)] = 12,
) -> ToolResult:
    """Sequence features: time between consecutive events of different types, and inactivity after the last event.

    Consecutive events of the same entity (by time) form transitions a -> b
    (a != b). Transitions where the later row refers to the earlier one (the
    spec's self-reference) are system responses and are left to
    build_linked_event_features. For the given `pairs`, or the `max_pairs`
    transitions most entities have: share of the entity's a events followed by b
    within `fast_seconds`, and the median gap. Plus the inactivity after the
    entity's last event (seconds to the end of the period, and as a share of the
    entity's active span).

    Args:
        input: Event table (the enriched artifact).
        entity: Entity column.
        category: Type column of the events.
        time_column: Event time.
        features: Feature table to extend.
        pairs: Transitions [[a, b], ...] to measure (optional).
        fast_seconds: Window of a "fast" transition.
        max_pairs: Number of automatic transitions.
    """
    try:
        spec = read_spec()
        con, schema = _open_events(input, entity, [category, time_column], spec=spec)
    except (SpecMissingError, SourceNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    try:
        e, t = quote_ident(entity), quote_ident(time_column)
        ref = _self_reference(spec, schema)
        ids = f", {quote_ident(ref[1])} AS id, {quote_ident(ref[0])} AS ref" if ref else ", NULL AS id, NULL AS ref"
        con.execute(
            f"CREATE TEMP TABLE seq AS SELECT {e} AS ent, epoch({t}) AS ts, {_norm(category)} AS ty{ids}, "
            f"lead({_norm(category)}) OVER w AS next_ty, lead(epoch({t})) OVER w - epoch({t}) AS gap, "
            f"lead({quote_ident(ref[0]) if ref else 'NULL'}) OVER w AS next_ref "
            f"FROM ev WINDOW w AS (PARTITION BY {e} ORDER BY {t}, {_norm(category)})"
        )
        system = "AND (next_ref IS NULL OR id IS NULL OR next_ref <> id)"
        if pairs:
            chosen = [(a.strip().lower(), b.strip().lower()) for a, b in pairs]
        else:
            chosen = [
                (a, b)
                for a, b, _ in con.execute(
                    f"SELECT ty, next_ty, count(DISTINCT ent) n FROM seq WHERE next_ty IS NOT NULL AND next_ty <> ty {system} "
                    f"GROUP BY 1, 2 ORDER BY n DESC, 1, 2 LIMIT {int(max_pairs)}"
                ).fetchall()
            ]
        end = con.execute("SELECT max(ts) FROM seq").fetchall()[0][0]
        select = [
            f"{end} - max(ts) AS seq_inactivity_after_last_s",
            f"({end} - max(ts)) / nullif({end} - min(ts), 0) AS seq_inactivity_share",
        ]
        entries = [
            {
                "name": "seq_inactivity_after_last_s",
                "source": time_column,
                "template": "sequence",
                "formula": "seconds from the entity's last event to the end of the period",
            },
            {
                "name": "seq_inactivity_share",
                "source": time_column,
                "template": "sequence",
                "formula": "inactivity after the last event / time from the first event to the end",
            },
        ]
        for a, b in chosen:
            la, lb = f"'{sql_escape(a)}'", f"'{sql_escape(b)}'"
            fast, gap = _feature_name("seq", a, "to", b, "fast_share"), _feature_name("seq", a, "to", b, "gap_p50_s")
            cond = f"ty = {la} AND next_ty = {lb} {system}"
            select += [
                f"count(*) FILTER (WHERE {cond} AND gap <= {int(fast_seconds)}) / nullif(count(*) FILTER (WHERE ty = {la}), 0) AS {quote_ident(fast)}",
                f"median(gap) FILTER (WHERE {cond}) AS {quote_ident(gap)}",
            ]
            entries += [
                {
                    "name": fast,
                    "source": category,
                    "template": "sequence",
                    "formula": f"share of {a} events followed by {b} within {fast_seconds} s",
                },
                {
                    "name": gap,
                    "source": time_column,
                    # Heavy-tailed durations: their own family, so they do not drown the shares.
                    "template": "sequence_gap",
                    "formula": f"median seconds from {a} to a following {b}",
                },
            ]
        new = con.execute(f"SELECT ent AS {e}, {', '.join(select)} FROM seq GROUP BY ent").df()
    finally:
        con.close()
    try:
        table = _extend(features, entity, new, entries, [input], fill=None)
    except (ArtifactNotFoundError, FileNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    return _result(entity, table, entries, transitions=[list(p) for p in chosen])


# --------------------------------------------------------------------------- #
# Model 5: conformity with the reference tables                               #
# --------------------------------------------------------------------------- #
@tool
def build_conformity_features(
    input: str,
    entity: str,
    name: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,40}$")],
    observed: str,
    expected: str,
    features: str,
    where: str | None = None,
    tolerance: Annotated[float, Field(ge=0)] = 0.01,
) -> ToolResult:
    """Compare an observed value with the value expected from the enriched columns, per row; aggregate per entity.

    `observed`, `expected` and `where` are single SQL expressions over the row's
    columns (enriched with the reference tables), validated like run_sql (no
    queries, no files). With a self-reference in the spec, the columns of the
    referred row are available with the prefix `ref_` (e.g. `ref_face_value`).
    Propose `expected` from the reference table's columns and the README notes,
    e.g. a bonus as `least(ref_amount * bonus_pct / 100, max_bonus)`.

    Features: rows checked, rows off by more than `tolerance`, their share, the
    sum and the largest absolute deviation, and the median observed/expected ratio.

    Args:
        input: Event table (the enriched artifact).
        entity: Entity column.
        name: Short name of the rule (feature prefix).
        observed: Expression of the observed value (e.g. "amount").
        expected: Expression of the expected value.
        features: Feature table to extend.
        where: Expression selecting the rows the rule applies to (optional).
        tolerance: Absolute deviation still considered conforming.
    """
    try:
        spec = read_spec()
        obs = _safe_expression(observed, "observed")
        exp = _safe_expression(expected, "expected")
        cond = _safe_expression(where, "where") if where else "TRUE"
        con, schema = _open_events(input, entity, [], spec=spec, expressions=f"{obs} {exp} {cond}")
    except (SpecMissingError, SourceNotFoundError, ValueError, SQLRejectedError) as exc:
        return ToolResult.error(str(exc))
    try:
        ref = _self_reference(spec, schema)
        uses_ref = bool(re.search(r"\bref_\w+", f"{obs} {exp} {cond}"))
        if uses_ref and not ref:
            return ToolResult.error("ref_ columns need a self-reference in the spec (spec.references).")
        if uses_ref:
            r, target = (quote_ident(x) for x in ref)
            ref_cols = ", ".join(
                f"p.{quote_ident(c)} AS {quote_ident('ref_' + c)}" for c in schema if "ref_" + c not in schema
            )
            relation = (
                f"(SELECT ev.*, {ref_cols} FROM ev LEFT JOIN (SELECT DISTINCT * FROM ev) p ON ev.{r} = p.{target})"
            )
        else:
            relation = "ev"
        con.execute(
            f"CREATE TEMP TABLE conf AS SELECT {quote_ident(entity)} AS ent, CAST(({obs}) AS DOUBLE) AS o, "
            f"CAST(({exp}) AS DOUBLE) AS x FROM {relation} WHERE coalesce(({cond}), false)"
        )
        checked, off, unknown = con.execute(
            f"SELECT count(*), count(*) FILTER (WHERE abs(o - x) > {tolerance}), count(*) FILTER (WHERE x IS NULL) FROM conf"
        ).fetchall()[0]
        p = _feature_name("conf", name)
        new = con.execute(
            f"""SELECT ent AS {quote_ident(entity)}, count(*) AS {quote_ident(p + "_rows")},
                       count(*) FILTER (WHERE abs(o - x) > {tolerance}) AS {quote_ident(p + "_violations")},
                       count(*) FILTER (WHERE abs(o - x) > {tolerance}) / count(*) AS {quote_ident(p + "_violation_share")},
                       coalesce(sum(o - x) FILTER (WHERE abs(o - x) > {tolerance}), 0) AS {quote_ident(p + "_deviation_sum")},
                       coalesce(max(abs(o - x)), 0) AS {quote_ident(p + "_deviation_max")},
                       median(o / nullif(x, 0)) AS {quote_ident(p + "_ratio_p50")}
                FROM conf GROUP BY ent"""
        ).df()
    except duckdb.Error as exc:
        return ToolResult.error(f"Cannot evaluate the expressions: {str(exc).splitlines()[0]}")
    finally:
        con.close()
    formula = f"observed {obs} vs expected {exp} on rows where {cond} (tolerance {tolerance})"
    # Columns the rule reads (ref_<c> reads column c of the referred row): the critic uses
    # them to know which reference table a rule comes from.
    # The rule itself (already validated), so the critic can re-check it on any subset of rows.
    rule = {"observed": obs, "expected": exp, "where": cond, "tolerance": tolerance, "uses_ref": uses_ref}
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f"{obs} {exp} {cond}"))
    rule_columns = sorted(
        {t for t in tokens if t in schema} | {t[4:] for t in tokens if t.startswith("ref_") and t[4:] in schema}
    )
    entries = [
        {
            "name": p + suffix,
            "source": name,
            "template": "conformity",
            "formula": f"{label}: {formula}",
            "columns": rule_columns,
            "rule": rule,
        }
        for suffix, label in (
            ("_rows", "rows checked"),
            ("_violations", "rows off by more than the tolerance"),
            ("_violation_share", "share of rows off"),
            ("_deviation_sum", "sum of observed - expected on rows off"),
            ("_deviation_max", "largest |observed - expected|"),
            ("_ratio_p50", "median observed / expected"),
        )
    ]
    try:
        table = _extend(features, entity, new, entries, [input], fill=0.0)
    except (ArtifactNotFoundError, FileNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    warnings = [f"{unknown} rows have no expected value (null): check the expression."] if unknown else []
    result = _result(
        entity,
        table,
        entries,
        rows_checked=int(checked),
        rows_off=int(off),
        share_off=round(off / checked, 6) if checked else 0.0,
    )
    result.warnings.extend(warnings)
    return result


# --------------------------------------------------------------------------- #
# Interaction graph (who transacts with whom)                                 #
# --------------------------------------------------------------------------- #
@tool
def build_interaction_graph(input: str, entity: str, counterpart_column: str, features: str) -> ToolResult:
    """Graph of entities linked by a counterpart column (the other party of a row), with node features.

    An edge links the row's entity and its counterpart when the counterpart is
    itself an entity. Node features: number of distinct partners, size of the
    connected component, and the largest and mean partner degree (a leaf of a
    star has one partner with many partners). Use it for the other party of an
    exchange (sender/receiver); use build_graph for shared resources.

    Args:
        input: Event table (the enriched artifact).
        entity: Entity column.
        counterpart_column: Column holding the other party (same kind of values as `entity`).
        features: Feature table to extend.
    """
    try:
        con, _ = _open_events(input, entity, [counterpart_column])
    except (SourceNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    try:
        e, cp = quote_ident(entity), quote_ident(counterpart_column)
        con.execute(
            f"CREATE TEMP TABLE edges AS SELECT DISTINCT least(CAST({e} AS VARCHAR), CAST({cp} AS VARCHAR)) a, "
            f"greatest(CAST({e} AS VARCHAR), CAST({cp} AS VARCHAR)) b FROM ev "
            f"WHERE {cp} IS NOT NULL AND CAST({cp} AS VARCHAR) IN (SELECT DISTINCT CAST({e} AS VARCHAR) FROM ev) "
            f"AND CAST({cp} AS VARCHAR) <> CAST({e} AS VARCHAR)"
        )
        n_edges = con.execute("SELECT count(*) FROM edges").fetchall()[0][0]
        if n_edges > _MAX_GRAPH_EDGES:
            return ToolResult.error(
                f"{n_edges} distinct pairs of entities: too many for a graph (limit {_MAX_GRAPH_EDGES}). "
                "Filter the events first (e.g. one type of exchange)."
            )
        # Partner counts in SQL: each pair is stored once (a < b), so both directions are listed.
        con.execute(
            "CREATE TEMP TABLE adj AS SELECT a AS n, b AS p FROM edges UNION ALL SELECT b AS n, a AS p FROM edges"
        )
        con.execute("CREATE TEMP TABLE deg AS SELECT n, count(*) AS k FROM adj GROUP BY n")
        node_stats = con.execute(
            "SELECT adj.n AS node, any_value(mine.k) AS partners, max(d.k) AS degree_max, "
            "sum(d.k) / count(*) AS degree_mean FROM adj JOIN deg d ON adj.p = d.n JOIN deg mine ON adj.n = mine.n "
            "GROUP BY adj.n"
        ).df()
        pairs = con.execute("SELECT a AS src, b AS dst FROM edges").df()
        entities = con.execute(f"SELECT DISTINCT {e} FROM ev").df()[entity]
    finally:
        con.close()
    components = component_table(pairs, "node")
    del pairs
    stats = node_stats.merge(components[["node", "group_size"]], on="node", how="left").set_index("node")
    keys = entities.astype(str)
    new = pd.DataFrame(
        {
            entity: entities.to_numpy(),
            "igraph_partners": keys.map(stats["partners"]).fillna(0).astype(int).to_numpy(),
            "igraph_component_size": keys.map(stats["group_size"]).fillna(1).astype(int).to_numpy(),
            "igraph_partner_degree_max": keys.map(stats["degree_max"]).fillna(0).astype(int).to_numpy(),
            "igraph_partner_degree_mean": keys.map(stats["degree_mean"]).fillna(0.0).astype(float).to_numpy(),
        }
    )
    entries = [
        {"name": n, "source": counterpart_column, "template": "interaction_graph", "formula": f}
        for n, f in (
            ("igraph_partners", f"distinct entities linked through {counterpart_column}"),
            ("igraph_component_size", "size of the connected component"),
            ("igraph_partner_degree_max", "largest number of partners of a partner"),
            ("igraph_partner_degree_mean", "mean number of partners of the partners"),
        )
    ]
    try:
        table = _extend(features, entity, new, entries, [input], fill=0.0)
    except (ArtifactNotFoundError, FileNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    sizes = components.drop_duplicates("group")["group_size"]
    return _result(
        entity,
        table,
        entries,
        nodes=len(stats),
        edges=int(n_edges),
        largest_components=sizes.head(6).tolist(),
        largest_degrees=stats["partners"].sort_values(ascending=False).head(6).astype(int).tolist(),
    )
