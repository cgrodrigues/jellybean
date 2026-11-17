# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Row filters for a run (``jellybean run --filter "<dataset>.<column> ..."``).

A filter is a SQL boolean condition whose columns are prefixed with the dataset
name, for example ``traffic.date >= '2026-10-01' and traffic.date < '2026-10-03'``.
Each expression must refer to exactly one dataset. The prefixes are removed and
the condition becomes a ``WHERE`` clause of that dataset's view, so every agent
and tool of the run sees the same filtered rows. Several expressions for the
same dataset are combined with ``AND``.
"""

from __future__ import annotations

import re

import duckdb

from jellybean.settings import Settings
from jellybean.skills._shared.catalog import connect, discover_datasets, quote_ident
from jellybean.skills._shared.sql_guard import SQLRejectedError, validate_select

_LITERAL = re.compile(r"'(?:[^']|'')*'")
_QUALIFIED = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")


class FilterError(ValueError):
    """An invalid --filter expression (message is shown to the user)."""


def _split_literals(expression: str) -> list[tuple[bool, str]]:
    """Split into (is_literal, text) parts so prefixes inside strings are left alone."""
    parts, pos = [], 0
    for m in _LITERAL.finditer(expression):
        parts.append((False, expression[pos : m.start()]))
        parts.append((True, m.group(0)))
        pos = m.end()
    parts.append((False, expression[pos:]))
    return parts


def parse_filters(expressions: list[str], datasets: set[str]) -> dict[str, str]:
    """Turn ``--filter`` expressions into ``{dataset: condition}`` (prefixes removed)."""
    result: dict[str, list[str]] = {}
    for raw in expressions:
        expression = raw.strip()
        if not expression:
            continue
        if ";" in expression:
            raise FilterError(f"Filter must be a single condition (no ';'): {raw}")
        used: set[str] = set()
        rebuilt = []
        for is_literal, text in _split_literals(expression):
            if is_literal:
                rebuilt.append(text)
                continue

            def strip(m: re.Match[str], used: set[str] = used) -> str:
                if m.group(1) in datasets:
                    used.add(m.group(1))
                    return quote_ident(m.group(2))
                return m.group(0)

            rebuilt.append(_QUALIFIED.sub(strip, text))
        if not used:
            raise FilterError(f"Prefix the columns with a dataset name ({', '.join(sorted(datasets))}): {raw}")
        if len(used) > 1:
            raise FilterError(
                f"A filter must refer to one dataset only (found {sorted(used)}): {raw}. Pass one --filter per dataset."
            )
        result.setdefault(used.pop(), []).append("".join(rebuilt).strip())
    return {name: " AND ".join(f"({c})" for c in conditions) for name, conditions in result.items()}


def validate_filters(settings: Settings, filters: dict[str, str]) -> dict[str, tuple[int, int]]:
    """Check each filter on its dataset; returns ``{dataset: (rows_after, rows_before)}``.

    Raises:
        FilterError: If a condition is rejected, fails to run or keeps no rows.
    """
    counts: dict[str, tuple[int, int]] = {}
    con = connect(settings=settings, ctx=None)
    try:
        for name, condition in filters.items():
            query = f"SELECT count(*) FROM {quote_ident(name)} WHERE {condition}"
            try:
                validate_select(query, forbidden_paths=[settings.ground_truth_root])
                after = con.execute(query).fetchall()[0][0]
            except SQLRejectedError as exc:
                raise FilterError(f"Filter on {name} rejected: {exc}") from exc
            except duckdb.Error as exc:
                raise FilterError(f"Filter on {name} is not valid: {str(exc).splitlines()[0]}") from exc
            before = con.execute(f"SELECT count(*) FROM {quote_ident(name)}").fetchall()[0][0]
            if after == 0:
                raise FilterError(f"Filter on {name} keeps no rows: {condition}")
            counts[name] = (int(after), int(before))
    finally:
        con.close()
    return counts


def available_datasets(settings: Settings) -> set[str]:
    return set(discover_datasets(settings.data_root))
