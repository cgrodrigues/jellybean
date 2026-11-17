# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""The analysis entity chosen by the user (``jellybean run --entity <dataset>.<column>``).

Every analysis is about one entity: the "who" that rows belong to (a customer,
a machine, a dealer...). Without ``--entity`` the profiler proposes it and
control 1 confirms it. With ``--entity`` the user's choice is validated against
the data before the run starts, recorded as a human decision, and cannot be
changed by the agents. To analyse several entities, run one analysis per entity.

The entity must repeat across rows: the system aggregates the events of each
entity. A column with (almost) one value per row is refused, because row-level
anomaly detection is not supported.
"""

from __future__ import annotations

import re
from typing import Any

import duckdb

from jellybean.settings import Settings
from jellybean.skills._shared.catalog import connect, discover_datasets, quote_ident

_ENTITY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$")
# Above this share of distinct values per row, a column is a row identifier, not an entity.
MAX_DISTINCT_RATIO = 0.5


class EntityError(ValueError):
    """An invalid --entity (message is shown to the user)."""


def parse_entity(expression: str, settings: Settings) -> tuple[str, str]:
    """Split ``dataset.column`` and check the dataset exists."""
    match = _ENTITY_RE.match(expression.strip())
    if not match:
        raise EntityError(f"Use the form <dataset>.<column>, e.g. traffic.msisdn (got {expression!r}).")
    dataset, column = match.groups()
    datasets = discover_datasets(settings.data_root)
    if dataset not in datasets:
        raise EntityError(f"Unknown dataset '{dataset}'. Available: {sorted(datasets)}")
    return dataset, column


def validate_entity(settings: Settings, dataset: str, column: str, where: str | None = None) -> dict[str, Any]:
    """Check that ``column`` can be the analysis entity and return its activity statistics.

    Raises:
        EntityError: Unknown column, too few entities, or (almost) one value per row.
    """
    con = connect(settings=settings, ctx=None)
    try:
        columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {quote_ident(dataset)}").fetchall()]
        if column not in columns:
            raise EntityError(f"Unknown column '{column}' in '{dataset}'. Columns: {columns}")
        q = quote_ident(column)
        relation = f"(SELECT * FROM {quote_ident(dataset)} {f'WHERE {where}' if where else ''})"
        rows, non_null, distinct = con.execute(
            f"SELECT count(*), count({q}), count(DISTINCT {q}) FROM {relation}"
        ).fetchall()[0]
        if distinct < 2:
            raise EntityError(f"'{dataset}.{column}' has {distinct} distinct value(s): nothing to compare.")
        ratio = distinct / max(non_null, 1)
        if ratio > MAX_DISTINCT_RATIO:
            raise EntityError(
                f"'{dataset}.{column}' has {distinct:,} distinct values for {non_null:,} rows (almost one per row): "
                "it identifies rows, not an entity with several events. Row-level analysis is not supported; "
                "choose a column whose values repeat (e.g. a customer, machine or dealer)."
            )
        median, low, high = con.execute(
            f"SELECT median(n), min(n), max(n) FROM (SELECT count(*) AS n FROM {relation} "
            f"WHERE {q} IS NOT NULL GROUP BY {q})"
        ).fetchall()[0]
    except duckdb.Error as exc:
        raise EntityError(f"Cannot validate '{dataset}.{column}': {str(exc).splitlines()[0]}") from exc
    finally:
        con.close()
    return {
        "entities": int(distinct),
        "rows": int(rows),
        "null_rows": int(rows - non_null),
        "median_events": float(median),
        "min_events": int(low),
        "max_events": int(high),
    }
