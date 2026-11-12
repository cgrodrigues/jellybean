"""Preparer tools (spec section 7.3).

The preparer proposes a justified cleaning plan from the profiler's quality
issues, applies it once approved (control 2) and materialises the confirmed
joins. Every transformation runs in DuckDB and produces a **new** artifact;
inputs are never modified.

Each operation's affected rows are logged (with the entity/grain keys when
available) in a ``transform_log`` artifact, so the critic can later test
whether a finding is a cleaning artefact.

Policy decisions encoded here (documented in the plan's justifications):

* identifiers and categories are never imputed with a constant (that would
  create an artificial group of entities);
* values are never winsorised automatically (it would hide anomalies);
* fixes inferred from evidence (e.g. a unit conversion factor) are proposed but
  require explicit approval: they are not part of the recommended plan.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

import duckdb
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter, ValidationError, model_validator

from jellybean.runtime.context import current_run
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.skills._shared.artifacts import ArtifactNotFoundError, ArtifactStore
from jellybean.skills._shared.catalog import (
    FILE_COLUMN,
    Source,
    SourceNotFoundError,
    connect,
    dataset_relation_with_files,
    quote_ident,
    resolve_source,
    sql_escape,
)
from jellybean.skills._shared.column_stats import compute_column_stats, describe, type_family
from jellybean.skills._shared.keys import is_valid_strategy, strategy_sql
from jellybean.skills._shared.spec_tools import SpecMissingError, patch_spec, read_spec
from jellybean.skills._shared.sql_guard import SQLRejectedError, validate_select

_CAST_TYPES = {"BIGINT", "INTEGER", "DOUBLE", "VARCHAR", "DATE", "TIMESTAMP", "BOOLEAN"}


# --------------------------------------------------------------------------- #
# Declarative operations                                                      #
# --------------------------------------------------------------------------- #
class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Plan operation id (e.g. "P03"), used to label the affected-rows log.
    _plan_id: str | None = PrivateAttr(default=None)


class DedupeOp(_Op):
    """Remove duplicate rows: exact duplicates, or duplicates of ``keys`` keeping one row."""

    op: Literal["dedupe"]
    keys: list[str] | None = None
    keep: Literal["first", "latest"] = "first"
    order_by: str | None = None


class CastOp(_Op):
    """Convert a column to another type (values that cannot be converted become null)."""

    op: Literal["cast"]
    column: str
    to: str


class ImputeOp(_Op):
    """Fill nulls of a column with its median, mean or a constant."""

    op: Literal["impute"]
    column: str
    method: Literal["median", "mean", "constant"] = "median"
    value: float | str | None = None


class WinsorizeOp(_Op):
    """Clip a numeric column to its [lower_q, upper_q] quantiles."""

    op: Literal["winsorize"]
    column: str
    lower_q: float = Field(0.01, ge=0, le=0.5)
    upper_q: float = Field(0.99, ge=0.5, le=1)


class NormalizeOp(_Op):
    """Normalise text values; ``canonical`` maps spelling variants to the most frequent spelling."""

    op: Literal["normalize"]
    column: str
    method: Literal["trim", "lower", "upper", "trim_lower", "canonical"] = "canonical"


class DropRowsOp(_Op):
    """Drop rows matching a SQL boolean condition."""

    op: Literal["drop_rows"]
    where: str


class FixUnitsOp(_Op):
    """Rescale a numeric column (e.g. milliseconds -> seconds) on matching rows, a file or a partition.

    ``file`` is a path relative to the dataset folder; ending it with "/" selects
    every file of that partition (e.g. "date=2026-10-17/").
    """

    op: Literal["fix_units"]
    column: str
    factor: float = Field(gt=0)
    operation: Literal["divide", "multiply"] = "divide"
    where: str | None = None
    file: str | None = None


class FlipSignOp(_Op):
    """Invert the sign of a numeric column on the rows of one segment and date range.

    The segment is ``category_column`` in ``values`` (compared case- and
    space-insensitively); ``date_column`` between ``date_from`` and ``date_to``
    (inclusive) limits it in time. ``only_if`` is an optional SQL condition on top
    (e.g. "the row breaks the invariant"), so rows that were right are left alone.
    """

    op: Literal["flip_sign"]
    column: str
    category_column: str | None = None
    values: list[str] = Field(default_factory=list)
    date_column: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    only_if: str | None = None

    @model_validator(mode="after")
    def _has_a_filter(self) -> FlipSignOp:
        if not (self.values or self.date_from or self.only_if):
            raise ValueError("flip_sign needs a filter: values of a category, a date range or only_if")
        if bool(self.category_column) != bool(self.values):
            raise ValueError("flip_sign: category_column and values go together")
        if (self.date_from or self.date_to) and not self.date_column:
            raise ValueError("flip_sign: a date range needs date_column")
        return self


Operation = Annotated[
    DedupeOp | CastOp | ImputeOp | WinsorizeOp | NormalizeOp | DropRowsOp | FixUnitsOp | FlipSignOp,
    Field(discriminator="op"),
]
_OPERATIONS = TypeAdapter(list[Operation])


class PlanOp(BaseModel):
    """One step of a cleaning plan."""

    id: str
    dataset: str
    operation: dict[str, Any]
    issue_ids: list[str]
    justification: str
    requires_explicit_approval: bool = False


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _store() -> ArtifactStore:
    return ArtifactStore(current_run())


def _check_condition(where: str) -> None:
    """Validate a user-supplied SQL condition by embedding it in a SELECT."""
    if ";" in where:
        raise SQLRejectedError("Conditions cannot contain ';'.")
    validate_select(f"SELECT 1 FROM t WHERE ({where})", forbidden_paths=[current_run().settings.ground_truth_root])


def _select_list(columns: list[str], overrides: dict[str, str]) -> str:
    return ", ".join(f"{overrides[c]} AS {quote_ident(c)}" if c in overrides else quote_ident(c) for c in columns)


def _key_columns(columns: list[str]) -> list[str]:
    """Columns used to log affected rows: the spec entity and grain, else the join keys."""
    try:
        spec = read_spec()
    except SpecMissingError:
        return []
    keys = []
    if spec.entity and spec.entity.column in columns:
        keys.append(spec.entity.column)
    if spec.grain:
        keys += [c for c in spec.grain.columns if c in columns and c not in keys]
    if not keys:
        # Enrichment tables: log by their join key columns.
        for join in spec.joins:
            keys += [r for _, r in join.keys if r in columns and r not in keys]
    return keys


def _apply(con: duckdb.DuckDBPyConnection, source: Source, ops: list[Any]) -> tuple[str, list[dict[str, Any]], bool]:
    """Apply ``ops`` as a chain of temp tables. Returns (final table, per-op log, has_key_log)."""
    needs_file = any(isinstance(o, FixUnitsOp) and o.file for o in ops)
    if needs_file and source.dataset is None:
        raise ValueError("fix_units with 'file' is only possible on a raw dataset (file provenance is needed).")
    base = dataset_relation_with_files(source.dataset) if needs_file else f"(SELECT * FROM {source.view})"
    con.execute(f"CREATE OR REPLACE TEMP TABLE s0 AS SELECT * FROM {base}")
    schema = dict(describe(con, "s0"))
    columns = list(schema)
    keys = _key_columns(columns)
    key_sql = ", ".join(quote_ident(k) for k in keys)
    if keys:
        log_cols = ", ".join(f"{quote_ident(k)}" for k in keys)
        con.execute(f"CREATE OR REPLACE TEMP TABLE __log AS SELECT ''::VARCHAR AS op_id, {log_cols} FROM s0 LIMIT 0")
    log: list[dict[str, Any]] = []

    def need(col: str) -> str:
        if col not in schema:
            raise ValueError(f"Unknown column '{col}'. Columns: {[c for c in columns if c != FILE_COLUMN]}")
        return quote_ident(col)

    for i, op in enumerate(ops, 1):
        prev, cur = f"s{i - 1}", f"s{i}"
        op_id = op._plan_id or f"op{i}"
        changed_where: str | None = None  # condition on `prev` identifying affected rows
        removed = False
        if isinstance(op, DedupeOp):
            removed = True
            if op.keys:
                part = ", ".join(need(k) for k in op.keys)
                order_col = need(op.order_by) if op.order_by else None
                direction = "DESC" if op.keep == "latest" else "ASC"
                tiebreak = ", ".join(quote_ident(c) for c in columns)
                order = f"{order_col} {direction} NULLS LAST, {tiebreak}" if order_col else tiebreak
                con.execute(
                    f"CREATE TEMP TABLE {cur} AS SELECT * FROM {prev} "
                    f"QUALIFY row_number() OVER (PARTITION BY {part} ORDER BY {order}) = 1"
                )
            else:
                con.execute(f"CREATE TEMP TABLE {cur} AS SELECT DISTINCT * FROM {prev}")
        elif isinstance(op, DropRowsOp):
            _check_condition(op.where)
            removed = True
            con.execute(f"CREATE TEMP TABLE {cur} AS SELECT * FROM {prev} WHERE NOT coalesce(({op.where}), false)")
        elif isinstance(op, CastOp):
            target = op.to.upper()
            if target not in _CAST_TYPES:
                raise ValueError(f"Unsupported cast type '{op.to}'. Use one of {sorted(_CAST_TYPES)}")
            q = need(op.column)
            changed_where = f"{q} IS NOT NULL AND try_cast({q} AS {target}) IS NULL"
            select = _select_list(columns, {op.column: f"try_cast({q} AS {target})"})
            con.execute(f"CREATE TEMP TABLE {cur} AS SELECT {select} FROM {prev}")
            schema[op.column] = target
        elif isinstance(op, ImputeOp):
            q = need(op.column)
            if op.method == "constant":
                if op.value is None:
                    raise ValueError("impute with method 'constant' needs a value")
                fill = f"'{sql_escape(op.value)}'" if isinstance(op.value, str) else str(op.value)
            else:
                agg = "median" if op.method == "median" else "avg"
                value = con.execute(f"SELECT {agg}({q}) FROM {prev}").fetchall()[0][0]
                if value is None:
                    raise ValueError(f"Cannot impute '{op.column}': no non-null values")
                fill = repr(float(value))
            changed_where = f"{q} IS NULL"
            select = _select_list(columns, {op.column: f"coalesce({q}, {fill})"})
            con.execute(f"CREATE TEMP TABLE {cur} AS SELECT {select} FROM {prev}")
        elif isinstance(op, WinsorizeOp):
            q = need(op.column)
            lo, hi = con.execute(
                f"SELECT quantile_cont({q}, {op.lower_q}), quantile_cont({q}, {op.upper_q}) FROM {prev}"
            ).fetchall()[0]
            changed_where = f"{q} < {lo} OR {q} > {hi}"
            select = _select_list(columns, {op.column: f"least(greatest({q}, {lo}), {hi})"})
            con.execute(f"CREATE TEMP TABLE {cur} AS SELECT {select} FROM {prev}")
        elif isinstance(op, NormalizeOp):
            q = need(op.column)
            if op.method == "canonical":
                con.execute(
                    f"CREATE OR REPLACE TEMP TABLE __map AS SELECT k, arg_max(v, n) AS canon FROM ("
                    f"SELECT lower(trim({q})) AS k, {q} AS v, count(*) AS n FROM {prev} WHERE {q} IS NOT NULL "
                    f"GROUP BY 1, 2 ORDER BY 1, 3 DESC, 2) GROUP BY k"
                )
                expr = f"coalesce((SELECT canon FROM __map WHERE k = lower(trim({q}))), {q})"
            else:
                expr = {
                    "trim": f"trim({q})",
                    "lower": f"lower({q})",
                    "upper": f"upper({q})",
                    "trim_lower": f"lower(trim({q}))",
                }[op.method]
            changed_where = f"{q} IS DISTINCT FROM {expr}"
            select = _select_list(columns, {op.column: expr})
            con.execute(f"CREATE TEMP TABLE {cur} AS SELECT {select} FROM {prev}")
        elif isinstance(op, FixUnitsOp):
            q = need(op.column)
            conditions = []
            if op.where:
                _check_condition(op.where)
                conditions.append(f"({op.where})")
            if op.file and op.file.endswith("/"):  # a whole partition, e.g. "date=2026-10-17/"
                conditions.append(f"starts_with({FILE_COLUMN}, '{sql_escape(op.file)}')")
            elif op.file:
                conditions.append(f"{FILE_COLUMN} = '{sql_escape(op.file)}'")
            cond = " AND ".join(conditions) or "true"
            scaled = f"{q} / {op.factor}" if op.operation == "divide" else f"{q} * {op.factor}"
            if type_family(schema[op.column]) == "integer":
                scaled = f"CAST(round({scaled}) AS {schema[op.column]})"
            changed_where = f"{q} IS NOT NULL AND {cond}"
            expr = f"CASE WHEN {cond} THEN {scaled} ELSE {q} END"
            select = _select_list(columns, {op.column: expr})
            con.execute(f"CREATE TEMP TABLE {cur} AS SELECT {select} FROM {prev}")
        elif isinstance(op, FlipSignOp):
            q = need(op.column)
            if type_family(schema[op.column]) not in ("integer", "float"):
                raise ValueError(f"flip_sign: '{op.column}' is not numeric")
            conditions = [f"{q} IS NOT NULL"]
            if op.category_column:
                listed = ", ".join(f"'{sql_escape(v.strip().lower())}'" for v in op.values)
                conditions.append(f"lower(trim(CAST({need(op.category_column)} AS VARCHAR))) IN ({listed})")
            if op.date_column:
                d = f"CAST({need(op.date_column)} AS DATE)"
                if op.date_from:
                    conditions.append(f"{d} >= DATE '{sql_escape(op.date_from)}'")
                if op.date_to:
                    conditions.append(f"{d} <= DATE '{sql_escape(op.date_to)}'")
            if op.only_if:
                _check_condition(op.only_if)
                conditions.append(f"coalesce(({op.only_if}), false)")
            cond = " AND ".join(conditions)
            changed_where = cond
            select = _select_list(columns, {op.column: f"CASE WHEN {cond} THEN -{q} ELSE {q} END"})
            con.execute(f"CREATE TEMP TABLE {cur} AS SELECT {select} FROM {prev}")
        else:  # pragma: no cover - the discriminated union prevents this
            raise ValueError(f"Unsupported operation {op}")

        before = con.execute(f"SELECT count(*) FROM {prev}").fetchall()[0][0]
        after = con.execute(f"SELECT count(*) FROM {cur}").fetchall()[0][0]
        if removed:
            affected = before - after
            if keys and affected:
                con.execute(
                    f"INSERT INTO __log SELECT '{op_id}', {key_sql} FROM "
                    f"(SELECT {key_sql} FROM {prev} EXCEPT ALL SELECT {key_sql} FROM {cur})"
                )
        else:
            affected = con.execute(f"SELECT count(*) FROM {prev} WHERE {changed_where}").fetchall()[0][0]
            if keys and affected:
                con.execute(f"INSERT INTO __log SELECT '{op_id}', {key_sql} FROM {prev} WHERE {changed_where}")
        log.append(
            {"op_id": op_id, "op": op.op, "rows_before": before, "rows_after": after, "affected_rows": int(affected)}
        )

    final = f"s{len(ops)}"
    if needs_file:
        con.execute(f"CREATE TEMP TABLE s_final AS SELECT * EXCLUDE ({FILE_COLUMN}) FROM {final}")
        final = "s_final"
    return final, log, bool(keys)


def _run_transform(
    input_name: str, ops: list[Any], prefix_name: str | None = None, extra_params: dict[str, Any] | None = None
) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Apply operations and store the result. Returns (artifact id, log artifact id, per-op log)."""
    source = resolve_source(input_name)
    con = connect()
    store = _store()
    try:
        final, log, has_keys = _apply(con, source, ops)
        name = prefix_name or source.name
        params = {"operations": [o.model_dump() for o in ops], "log": log, **(extra_params or {})}
        record = store.save_table_from_sql(
            con, f"SELECT * FROM {final}", f"clean_{name}", parents=[source.name], params=params
        )
        log_id = None
        if has_keys:
            log_id = store.save_table_from_sql(
                con, "SELECT * FROM __log", f"transform_log_{name}", parents=[record.id], params={"log": log}
            ).id
    finally:
        con.close()
    return record.id, log_id, log


# --------------------------------------------------------------------------- #
# Tools                                                                        #
# --------------------------------------------------------------------------- #
def _semantic_type(spec: Any, dataset: str, column: str) -> str | None:
    for key in (column, f"{dataset}.{column}"):
        info = spec.columns.get(key)
        if info and info.dataset == dataset:
            return info.semantic_type
    return None


def _temporal_column(spec: Any, dataset: str) -> str | None:
    """First timestamp column of a dataset according to the spec."""
    for key, info in spec.columns.items():
        if info.dataset == dataset and info.semantic_type == "timestamp":
            return key.split(".", 1)[-1]
    return None


def _nearest_power_of_ten(factor: float) -> float | None:
    """10, 100, 1000... when ``factor`` (or its inverse) is within 15% of one."""
    if factor <= 0:
        return None
    exponent = round(math.log10(factor))
    if exponent == 0:
        return None
    candidate = 10.0**exponent
    return candidate if abs(factor / candidate - 1) <= 0.15 else None


def _belongs_to_spec_join(issue: dict[str, Any], joins: list[Any]) -> bool:
    """True when a join-specific issue comes from a join that is in the spec.

    Issues carry the join identity (``join_left``, ``join_right``, ``join_keys``)
    since phase 4; older ones are matched on dataset and key columns.
    """
    detail = issue.get("detail", {})
    columns = (issue.get("column") or "").split(",")
    for join in joins:
        keys = [list(k) for k in join.keys]
        if "join_keys" in detail:
            if (detail.get("join_left"), detail.get("join_right"), detail["join_keys"]) == (
                join.left,
                join.right,
                keys,
            ):
                return True
        elif issue["kind"] == "duplicate_keys":
            if issue["dataset"] == join.right and columns == [r for _, r in keys]:
                return True
        elif issue["dataset"] == join.left and columns == [lk for lk, _ in keys]:
            return True
    return False


def operation_signature(ops: list[dict[str, Any]]) -> list[str]:
    """Order-independent identity of a set of plan operations (ignores their ids)."""
    import json

    return sorted(json.dumps({"dataset": o["dataset"], "operation": o["operation"]}, sort_keys=True) for o in ops)


# Order in which operations are applied within a dataset.
_OP_ORDER = {"fix_units": 0, "flip_sign": 0, "normalize": 1, "cast": 2, "drop_rows": 3, "dedupe": 4, "impute": 5}


@tool
def propose_cleaning_plan(issues_artifact: str, skip_issues: list[str] | None = None) -> ToolResult:
    """Propose a justified cleaning plan, operation by operation, from the compiled quality issues.

    The plan is stored as an artifact and recorded in the spec (status "proposed").
    Inferred value changes (unit fixes) are marked as requiring explicit
    approval. Issues that should not be "fixed" (nulls in identifiers, time-zone
    shifts, unmatched rows, spikes) are listed with the reason.

    Args:
        issues_artifact: Artifact id from compile_quality_issues.
        skip_issues: Ids to leave out: issue ids ("Q003") or operation ids of a previous plan ("P02").
    """
    store = _store()
    try:
        spec = read_spec()
        issues = store.load_json(issues_artifact)["issues"]
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    except (ArtifactNotFoundError, FileNotFoundError, KeyError, TypeError):
        return ToolResult.error(f"'{issues_artifact}' is not a quality-issues artifact.")
    skip = set(skip_issues or [])
    # Operation ids (P..) refer to the current plan: translate them into issue ids.
    if any(x.startswith("P") for x in skip) and spec.cleaning_plan.artifact_id:
        try:
            previous = store.load_json(spec.cleaning_plan.artifact_id)["ops"]
            skip |= {i for o in previous if o["id"] in skip for i in o["issue_ids"]}
        except (ArtifactNotFoundError, FileNotFoundError, KeyError):
            pass
    active_joins = [j for j in spec.joins if j.status != "rejected"]
    ops: list[dict[str, Any]] = []
    not_addressed: list[dict[str, str]] = []
    flip_dates: dict[str, set[str]] = {}  # dates already fixed by a flip_sign, per dataset
    # Systematic invariant errors first: a sign inversion on the same dates is the same problem.
    issues = sorted(issues, key=lambda i: i["kind"] != "invariant_violation_systematic")

    def add(dataset: str, operation: dict[str, Any], issue: dict[str, Any], why: str, explicit: bool = False) -> None:
        ops.append(
            {
                "dataset": dataset,
                "operation": operation,
                "issue_ids": [issue["id"]],
                "justification": why,
                "requires_explicit_approval": explicit,
            }
        )

    for issue in issues:
        kind, dataset, column, detail = issue["kind"], issue["dataset"], issue.get("column"), issue.get("detail", {})
        if issue["id"] in skip:
            not_addressed.append({"issue_id": issue["id"], "reason": "skipped on request"})
            continue
        if kind in ("duplicate_keys", "unmatched_rows") and not _belongs_to_spec_join(issue, active_joins):
            not_addressed.append(
                {
                    "issue_id": issue["id"],
                    "reason": (
                        "belongs to a join candidate that is not in the spec (e.g. a rejected alternative); ignored"
                    ),
                }
            )
            continue
        semantic = _semantic_type(spec, dataset, column) if column else None
        if kind == "duplicate_rows":
            add(dataset, {"op": "dedupe"}, issue, f"{detail.get('rows')} exact duplicate rows inflate counts.")
        elif kind == "duplicate_keys":
            order_by = _temporal_column(spec, dataset)
            keys = column.split(",")
            add(
                dataset,
                {"op": "dedupe", "keys": keys, "keep": "latest" if order_by else "first", "order_by": order_by},
                issue,
                f"{detail.get('keys')} duplicated join keys would duplicate primary rows (fan-out); keep one row "
                f"per key ({'the most recent by ' + order_by if order_by else 'deterministic first'}).",
            )
        elif kind == "negative_values":
            add(
                dataset,
                {"op": "drop_rows", "where": f"{quote_ident(column)} < 0"},
                issue,
                f"{detail.get('rows')} negative values ({detail.get('share')} of rows) in a non-negative "
                "measure are invalid records.",
            )
        elif kind == "inconsistent_categories":
            add(
                dataset,
                {"op": "normalize", "column": column, "method": "canonical"},
                issue,
                f"{detail.get('groups')} values have spelling variants (case/spaces); map each to its most "
                "frequent spelling.",
            )
        elif kind == "scale_shift_by_file":
            factor = _nearest_power_of_ten(float(detail.get("factor", 0)))
            if factor is None:
                not_addressed.append(
                    {"issue_id": issue["id"], "reason": "scale shift is not a power of ten; cannot infer a unit fix"}
                )
                continue
            op = "divide" if detail["factor"] > 1 else "multiply"
            add(
                dataset,
                {
                    "op": "fix_units",
                    "column": column,
                    "factor": factor if op == "divide" else 1 / factor,
                    "operation": op,
                    "file": detail.get("file"),
                },
                issue,
                f"One file has a median {detail.get('factor')}x the others: likely a unit error "
                f"(factor {factor:g}). Inferred fix: needs explicit approval.",
                explicit=True,
            )
        elif kind == "invariant_violation_systematic" and (detail.get("sign_flipped") or {}).get("column"):
            segment, dates = detail.get("segment") or {}, sorted(detail.get("dates") or [])
            flipped = detail["sign_flipped"]["column"]
            add(
                dataset,
                {
                    "op": "flip_sign",
                    "column": flipped,
                    "category_column": segment.get("column"),
                    "values": [str(v) for v in segment.get("values", [])],
                    "date_column": detail.get("date_column"),
                    "date_from": dates[0] if dates else None,
                    "date_to": dates[-1] if dates else None,
                    "only_if": detail.get("violation"),
                },
                issue,
                f"{detail.get('rows')} rows of {segment.get('column')}={segment.get('values')} on {len(dates)} "
                f"dates break '{detail.get('invariant')}' and satisfy it with the sign of {flipped} inverted: "
                "a systematic sign error. Inferred fix: needs explicit approval.",
                explicit=True,
            )
            flip_dates[dataset] = flip_dates.get(dataset, set()) | set(dates)
        elif kind == "sign_inversion" and set(detail.get("dates") or []) <= flip_dates.get(dataset, set()):
            not_addressed.append(
                {"issue_id": issue["id"], "reason": "same rows as the flip_sign of the systematic invariant error"}
            )
        elif kind == "nulls":
            if semantic == "measure" and detail.get("share", 1) <= 0.05:
                add(
                    dataset,
                    {"op": "impute", "column": column, "method": "median"},
                    issue,
                    f"{detail.get('share')} nulls in a measure; median imputation keeps rows usable.",
                )
            else:
                not_addressed.append(
                    {
                        "issue_id": issue["id"],
                        "reason": (
                            "nulls kept: imputing an identifier/category with a constant would create an artificial "
                            "group of entities; features ignore nulls"
                        ),
                    }
                )
        elif kind == "unparsable_timestamps":
            add(
                dataset,
                {"op": "cast", "column": column, "to": "TIMESTAMP"},
                issue,
                "Text timestamps converted; unparsable values become null.",
            )
        else:
            reasons = {
                "unmatched_rows": "kept: unmatched rows are flagged by materialize_join and analysed as a group",
                "possible_timezone_shift": "flagged as a limitation: the offset cannot be fixed with the "
                "available operations without guessing",
                "timestamp_outside_partition": "flagged as a limitation (see the time-zone issue)",
                "hour_profile_shift": "kept: may be real behaviour",
                "volume_spike_day": "kept: a spike may be a real anomaly",
                "volume_drop_day": "kept: a drop may be real; reported as a limitation",
                "constant_column": "kept: constant columns are ignored by feature building",
                "time_gaps": "kept: reported as a limitation",
                "structural_nulls": "not a problem: the value does not exist for these rows; never imputed",
                "null_keys": "kept: rows with a null join key are kept unmatched and flagged",
                "invariant_violation": "kept and reported: isolated records that break an invariant "
                "(reconciliation problem); not changed without knowing the right value",
                "invariant_violation_systematic": "reported as a systematic error of one segment; no automatic "
                "operation fixes it yet",
                "sign_inversion": "reported as a systematic sign error; no automatic operation fixes it yet",
                "orphan_references": "kept and reported: references to records that do not exist",
                "duplicate_business_keys": "kept and reported: rows sharing a key that should be unique "
                "(e.g. the same payment twice); removing them would hide the problem",
                "missing_reciprocal": "kept and reported: rows whose counterpart row is missing",
            }
            not_addressed.append({"issue_id": issue["id"], "reason": reasons.get(kind, "no automatic operation")})

    ops.sort(key=lambda o: (o["dataset"], _OP_ORDER[o["operation"]["op"]], bool(o["operation"].get("keys"))))
    plan_ops = []
    for n, op in enumerate(ops, 1):
        try:
            _OPERATIONS.validate_python([op["operation"]])
        except ValidationError as exc:  # pragma: no cover - defensive
            return ToolResult.error(f"Internal plan error: {exc}")
        plan_ops.append(PlanOp(id=f"P{n:02d}", **op).model_dump())
    record = store.save_json(
        "cleaning_plan", {"ops": plan_ops, "not_addressed": not_addressed}, parents=[issues_artifact]
    )
    # A re-proposal keeps `applied_signature`: what was applied stays in effect
    # until a different set of operations is approved and applied.
    patch_spec({"cleaning_plan": {"artifact_id": record.id, "status": "proposed", "approved_ops": []}})
    return ToolResult.ok(
        {
            "ops": [
                {
                    "id": o["id"],
                    "dataset": o["dataset"],
                    "op": o["operation"]["op"],
                    "column": o["operation"].get("column") or o["operation"].get("keys"),
                    "issue_ids": o["issue_ids"],
                    "explicit_approval": o["requires_explicit_approval"],
                    "justification": o["justification"][:160],
                }
                for o in plan_ops
            ],
            "not_addressed": not_addressed,
        },
        artifact_id=record.id,
    )


@tool
def apply_transform(input: str, operations: list[Operation]) -> ToolResult:
    """Apply declarative operations to a dataset or table artifact, creating a new version.

    Operations: dedupe, cast, impute, winsorize, normalize, drop_rows, fix_units, flip_sign.
    Each operation's affected rows are logged for the critic.

    Args:
        input: Dataset name or table artifact id.
        operations: Ordered list of operations, e.g. [{"op": "dedupe"}, {"op": "drop_rows", "where": "x < 0"}].
    """
    try:
        artifact_id, log_id, log = _run_transform(input, operations)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    except (ValueError, SQLRejectedError, duckdb.Error) as exc:
        return ToolResult.error(f"Transform failed: {exc}"[:1500])
    return ToolResult.ok(
        {"input": input, "output": artifact_id, "transform_log": log_id, "operations": log}, artifact_id=artifact_id
    )


@tool
def apply_cleaning_plan() -> ToolResult:
    """Apply the operations approved at control 2 to each dataset and record the cleaned versions in the spec."""
    store = _store()
    try:
        spec = read_spec()
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    plan_ref = spec.cleaning_plan
    if plan_ref.status != "approved" or not plan_ref.artifact_id:
        return ToolResult.error(f"The cleaning plan is not approved (status '{plan_ref.status}').")
    plan = store.load_json(plan_ref.artifact_id)
    approved = set(plan_ref.approved_ops)
    by_dataset: dict[str, list[Any]] = {}
    for op in plan["ops"]:
        if op["id"] in approved:
            parsed = _OPERATIONS.validate_python([op["operation"]])[0]
            parsed._plan_id = op["id"]
            by_dataset.setdefault(op["dataset"], []).append(parsed)
    results, prepared = [], {}
    for dataset, ops in by_dataset.items():
        try:
            artifact_id, log_id, log = _run_transform(dataset, ops, extra_params={"plan": plan_ref.artifact_id})
        except (ValueError, SQLRejectedError, duckdb.Error, SourceNotFoundError) as exc:
            return ToolResult.error(f"Applying the plan to '{dataset}' failed: {exc}"[:1500])
        prepared[dataset] = artifact_id
        results.append({"dataset": dataset, "output": artifact_id, "transform_log": log_id, "operations": log})
    applied = [o for o in plan["ops"] if o["id"] in approved]
    patch_spec(
        {
            "prepared": prepared,
            "cleaning_plan": {"status": "applied", "applied_signature": operation_signature(applied)},
        }
    )
    skipped = [o["id"] for o in plan["ops"] if o["id"] not in approved]
    return ToolResult.ok(
        {"results": results, "not_applied": skipped}, artifact_id=results[0]["output"] if len(results) == 1 else None
    )


@tool
def materialize_join(join_index: int = 0, left_artifact: str | None = None) -> ToolResult:
    """Execute a confirmed join from the spec (always a left join) and flag rows without a match.

    Uses the cleaned versions from the spec when available. Right-side columns
    whose names collide with left columns get the suffix "_<right dataset>". A
    boolean column "matched_<right dataset>" marks rows with a match; unmatched
    rows are kept.

    Args:
        join_index: Index of the join in spec.joins.
        left_artifact: Optional left input (e.g. the result of a previous join, to chain joins).
    """
    try:
        spec = read_spec()
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    if not 0 <= join_index < len(spec.joins):
        return ToolResult.error(f"No join at index {join_index}.")
    join = spec.joins[join_index]
    if join.status != "confirmed":
        return ToolResult.error(f"Join {join_index} is '{join.status}', not confirmed.")
    norms = join.key_normalization or ["none"]
    norms = norms * len(join.keys) if len(norms) == 1 else norms
    if not all(is_valid_strategy(n) for n in norms):
        return ToolResult.error(f"Invalid normalisation in spec: {norms}")
    left_name = left_artifact or spec.prepared.get(join.left, join.left)
    right_name = spec.prepared.get(join.right, join.right)
    try:
        left, right = resolve_source(left_name), resolve_source(right_name)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        lcols, rcols = [c for c, _ in describe(con, left.view)], [c for c, _ in describe(con, right.view)]
        rkeys = [r for _, r in join.keys]
        rk = ", ".join(
            f"{strategy_sql(n, quote_ident(r))} AS __k{i}"
            for i, ((_, r), n) in enumerate(zip(join.keys, norms, strict=True))
        )
        key_names = ", ".join(f"__k{i}" for i in range(len(join.keys)))
        dup = con.execute(
            f"SELECT {key_names} FROM (SELECT {rk} FROM {right.view}) GROUP BY {key_names} HAVING count(*) > 1"
        ).fetchall()
        if dup:
            return ToolResult.error(
                f"The right side still has {len(dup)} duplicated keys: dedupe it first "
                "(the join must not create extra rows)."
            )
        flag = f"matched_{join.right}"
        added = []
        select_right = []
        for c in rcols:
            if c in rkeys:
                continue
            alias = f"{c}_{join.right}" if c in lcols else c
            added.append(alias)
            select_right.append(f"R.{quote_ident(c)} AS {quote_ident(alias)}")
        on = " AND ".join(
            f"{strategy_sql(n, 'L.' + quote_ident(lk))} = R.__k{i}"
            for i, ((lk, _), n) in enumerate(zip(join.keys, norms, strict=True))
        )
        query = (
            f"SELECT L.*, {', '.join([*select_right, f'(R.__k0 IS NOT NULL) AS {quote_ident(flag)}'])} "
            f"FROM {left.view} AS L LEFT JOIN (SELECT {rk}, * FROM {right.view}) AS R ON {on}"
        )
        before = con.execute(f"SELECT count(*) FROM {left.view}").fetchall()[0][0]
        store = _store()
        record = store.save_table_from_sql(
            con,
            query,
            f"enriched_{join.left}",
            parents=[left.name, right.name],
            params={"join_index": join_index, "keys": join.keys, "normalization": norms, "flag": flag},
        )
        matched = con.execute(
            f"SELECT count(*) FILTER (WHERE {quote_ident(flag)}), count(*) FROM "
            f"read_parquet('{sql_escape(str(store.path_of(record.id)))}')"
        ).fetchall()[0]
    finally:
        con.close()
    if record.row_count != before:
        return ToolResult.error(f"Join changed the row count ({before} -> {record.row_count}).")
    patch_spec({"enriched_artifact": record.id, "phase": "features"})
    return ToolResult.ok(
        {
            "left": left.name,
            "right": right.name,
            "rows": before,
            "matched_rows": matched[0],
            "unmatched_rows": before - matched[0],
            "coverage": round(matched[0] / before, 4) if before else 0.0,
            "flag_column": flag,
            "added_columns": added,
        },
        artifact_id=record.id,
    )


@tool
def diff_versions(a: str, b: str) -> ToolResult:
    """Compare two versions of a table: row counts, added/removed columns, null and distribution changes.

    Args:
        a: Dataset name or table artifact id (before).
        b: Dataset name or table artifact id (after).
    """
    try:
        sa, sb = resolve_source(a), resolve_source(b)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        rows_a, stats_a = compute_column_stats(con, sa.view, with_top=False, with_samples=False)
        rows_b, stats_b = compute_column_stats(con, sb.view, with_top=False, with_samples=False)
    finally:
        con.close()
    da, db = {c.name: c for c in stats_a}, {c.name: c for c in stats_b}
    changes = []
    for name in da.keys() & db.keys():
        ca, cb = da[name], db[name]
        entry: dict[str, Any] = {"column": name}
        if ca.non_null != cb.non_null or ca.rows != cb.rows:
            entry["nulls"] = [ca.rows - ca.non_null, cb.rows - cb.non_null]
        if ca.distinct != cb.distinct:
            entry["distinct"] = [ca.distinct, cb.distinct]
        if ca.quantiles and cb.quantiles and ca.quantiles != cb.quantiles:
            entry["p50"] = [ca.quantiles["p50"], cb.quantiles["p50"]]
            entry["max"] = [ca.max, cb.max]
        if ca.min != cb.min and ca.family in ("integer", "float"):
            entry["min"] = [ca.min, cb.min]
        if len(entry) > 1:
            changes.append(entry)
    return ToolResult.ok(
        {
            "a": sa.name,
            "b": sb.name,
            "rows": [rows_a, rows_b],
            "row_change": rows_b - rows_a,
            "added_columns": sorted(db.keys() - da.keys()),
            "removed_columns": sorted(da.keys() - db.keys()),
            "changed_columns": sorted(changes, key=lambda c: c["column"]),
        }
    )


class Rule(BaseModel):
    """A data quality rule."""

    model_config = ConfigDict(extra="forbid")

    column: str | None = None
    check: Literal["not_null", "unique", "min", "max", "in_set", "regex", "no_duplicate_rows"]
    value: Any = None


@tool
def validate_constraints(artifact: str, rules: list[Rule]) -> ToolResult:
    """Check quality rules on a dataset or table artifact and count violations.

    Args:
        artifact: Dataset name or table artifact id.
        rules: e.g. [{"column": "customer_id", "check": "unique"}, {"column": "x", "check": "min", "value": 0}].
    """
    try:
        source = resolve_source(artifact)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    results = []
    try:
        columns = {c for c, _ in describe(con, source.view)}
        for rule in rules:
            if rule.check != "no_duplicate_rows" and (rule.column is None or rule.column not in columns):
                results.append({**rule.model_dump(), "passed": False, "error": "unknown or missing column"})
                continue
            q = quote_ident(rule.column) if rule.column else ""
            if rule.check == "not_null":
                sql = f"SELECT count(*) FROM {source.view} WHERE {q} IS NULL"
            elif rule.check == "unique":
                sql = (
                    f"SELECT count(*) FROM (SELECT {q} FROM {source.view} WHERE {q} IS NOT NULL "
                    "GROUP BY 1 HAVING count(*) > 1)"
                )
            elif rule.check == "min":
                sql = f"SELECT count(*) FROM {source.view} WHERE {q} < {float(rule.value)}"
            elif rule.check == "max":
                sql = f"SELECT count(*) FROM {source.view} WHERE {q} > {float(rule.value)}"
            elif rule.check == "in_set":
                values = ", ".join(f"'{sql_escape(str(v))}'" for v in (rule.value or []))
                sql = (
                    f"SELECT count(*) FROM {source.view} WHERE {q} IS NOT NULL "
                    f"AND CAST({q} AS VARCHAR) NOT IN ({values or 'NULL'})"
                )
            elif rule.check == "regex":
                pattern = sql_escape(str(rule.value))
                sql = (
                    f"SELECT count(*) FROM {source.view} WHERE {q} IS NOT NULL "
                    f"AND NOT regexp_full_match(CAST({q} AS VARCHAR), '{pattern}')"
                )
            else:
                sql = (
                    f"SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT * FROM {source.view})) "
                    f"FROM {source.view}"
                )
            violations = int(con.execute(sql).fetchall()[0][0])
            results.append({**rule.model_dump(exclude_none=True), "violations": violations, "passed": violations == 0})
    finally:
        con.close()
    return ToolResult.ok({"artifact": source.name, "passed": all(r["passed"] for r in results), "rules": results})
