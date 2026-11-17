# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Static validation of SQL submitted by agents.

This is the first of two defence layers for ``run_sql`` (the second is the
sandboxed DuckDB connection in :mod:`catalog`, which blocks file access outside
the allowed directories even if something slips through here).

Rules enforced here:

* exactly one statement, and it must be a ``SELECT`` (``WITH ... SELECT`` and
  set operations parse as ``SELECT`` in DuckDB);
* no table functions that execute dynamic SQL (``query``, ``query_table``);
* no string literal pointing to the ground-truth folder.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb


class SQLRejectedError(ValueError):
    """Raised when a query violates the read-only policy."""


# Functions that would let a SELECT run arbitrary SQL text or read raw files
# outside the curated views. File readers are also blocked by the sandbox; they
# are listed here to give the agent a clearer error.
_FORBIDDEN_FUNCTIONS = (
    "query",
    "query_table",
    "read_text",
    "read_blob",
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_json_auto",
    "glob",
    "sniff_csv",
)
_FORBIDDEN_FUNC_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_FUNCTIONS) + r")\s*\(", re.IGNORECASE)
_STRING_LITERAL_RE = re.compile(r"'((?:[^']|'')*)'")


def validate_select(sql: str, forbidden_paths: list[Path] | None = None) -> str:
    """Validate ``sql`` and return it without a trailing semicolon.

    Raises:
        SQLRejectedError: If the query is not a single read-only ``SELECT``.
    """
    query = sql.strip().rstrip(";").strip()
    if not query:
        raise SQLRejectedError("Empty query.")

    try:
        statements = duckdb.extract_statements(query)
    except duckdb.Error as exc:
        raise SQLRejectedError(f"SQL parse error: {exc}") from exc
    if len(statements) != 1:
        raise SQLRejectedError("Exactly one statement is allowed.")
    if statements[0].type != duckdb.StatementType.SELECT:
        kind = statements[0].type.name
        raise SQLRejectedError(f"Only SELECT / WITH queries are allowed (got {kind}).")

    match = _FORBIDDEN_FUNC_RE.search(query)
    if match:
        raise SQLRejectedError(f"Function '{match.group(1)}' is not allowed; query the dataset views instead.")

    for path in forbidden_paths or []:
        markers = {str(path).lower(), path.name.lower()}
        for literal in _STRING_LITERAL_RE.findall(query):
            text = literal.lower()
            if any(m and m in text for m in markers):
                raise SQLRejectedError("Access to this path is not allowed.")
    return query
