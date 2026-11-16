"""Resource use of the profiler's integrity checks: scans, query width, large-table path.

A recording connection shows *how* a check queries the data, which is what makes
it slow or heavy on large datasets (the results are covered by other tests).
"""

from __future__ import annotations

import re
from typing import Any

import duckdb
import pytest

from jellybean.runtime.context import RunContext
from jellybean.skills._shared import catalog
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills.profiler import integrity

FULL_DISTINCT = re.compile(r"SELECT\s+DISTINCT\s+\*", re.IGNORECASE)


class RecordingConnection:
    """Wraps a DuckDB connection and keeps every statement it executes."""

    def __init__(self, con: duckdb.DuckDBPyConnection, log: list[str]) -> None:
        self._con, self._log = con, log

    def execute(self, query: str, *args: Any, **kwargs: Any) -> Any:
        self._log.append(query)
        return self._con.execute(query, *args, **kwargs)

    def sql(self, query: str, *args: Any, **kwargs: Any) -> Any:
        self._log.append(query)
        return self._con.sql(query, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._con, name)


@pytest.fixture
def statements(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    log: list[str] = []
    real = catalog.connect

    def recording_connect(*args: Any, **kwargs: Any) -> RecordingConnection:
        return RecordingConnection(real(*args, **kwargs), log)

    monkeypatch.setattr(integrity, "connect", recording_connect)
    return log


@pytest.fixture
def ledger(settings) -> str:
    """12 measures (so 738 candidate relations), two categories, a date; c = a + b except on one bad segment."""
    folder = settings.data_root / "ledger"
    folder.mkdir()
    extra = ", ".join(f"((range * {i + 3}) % {97 + i}) * 1.25 AS m{i}" for i in range(9))
    con = duckdb.connect()
    # Every 'refund' row of day 12 has the sign of b flipped (2.5% of the rows).
    con.execute(
        f"""COPY (SELECT range AS id,
                         CASE WHEN range % 4 = 0 THEN 'refund' ELSE 'sale' END AS kind,
                         'S' || (range % 5) AS shop,
                         DATE '2026-10-01' + CAST(range % 40 AS INTEGER) AS date,
                         (range % 50) * 1.5 AS a, (range % 7 + 1) * 0.5 AS b,
                         CASE WHEN range % 40 = 12 THEN (range % 50) * 1.5 - (range % 7 + 1) * 0.5
                              ELSE (range % 50) * 1.5 + (range % 7 + 1) * 0.5 END AS c,
                         {extra}
                  FROM range(4000)) TO '{folder / "part-0.parquet"}' (FORMAT parquet)"""
    )
    con.close()
    return "ledger"


def test_invariants_scan_the_distinct_rows_once(run_ctx: RunContext, ledger: str, statements: list[str]) -> None:
    result = integrity.discover_arithmetic_invariants(ledger)
    assert result.status == "ok", result.summary
    assert "a + b = c" in {i["expression"] for i in result.summary["invariants"]}
    # The full-row DISTINCT is the expensive part: once, materialised, never inline per query.
    assert sum(bool(FULL_DISTINCT.search(q)) for q in statements) <= 1
    # Candidates are tested in batches, not as one statement with ~1,500 filtered aggregates.
    assert max(q.count("FILTER") for q in statements) <= 200
    # Locating the violations must not cost one scan per category column and invariant.
    assert len(statements) <= 60


def test_invariants_result_is_the_same_when_screened_on_a_sample(run_ctx: RunContext, ledger: str) -> None:
    exact = integrity.discover_arithmetic_invariants(ledger)
    run_ctx.settings.config.profiling.sample_rows = 1000  # 4,000 rows: candidates screened on a sample first
    screened = integrity.discover_arithmetic_invariants(ledger)
    assert screened.status == "ok", screened.summary
    assert screened.summary["invariants"] == exact.summary["invariants"]
    assert screened.summary["issues"] == exact.summary["issues"]


def test_large_tables_skip_the_full_row_distinct(run_ctx: RunContext, ledger: str, statements: list[str]) -> None:
    run_ctx.settings.config.profiling.distinct_max_rows = 1000  # the 4,000-row ledger is now "large"
    result = integrity.discover_arithmetic_invariants(ledger)
    assert result.status == "ok", result.summary
    assert not any(FULL_DISTINCT.search(q) for q in statements)
    assert any("duplicate rows were not removed" in w for w in result.warnings)
    systematic = [i for i in ArtifactStore().load_json(result.artifact_id)["issues"] if "systematic" in i["kind"]]
    assert systematic and systematic[0]["detail"]["sign_flipped"]["column"] == "b"


@pytest.mark.parametrize(
    ("tool_name", "kwargs", "max_statements"),
    [
        ("classify_nulls", {}, 40),
        ("find_duplicate_keys", {}, 40),
        ("discover_self_references", {}, 50),
        ("infer_sign_convention", {"measure": "c", "category": "kind"}, 40),
    ],
)
def test_other_integrity_checks_materialise_once(
    run_ctx: RunContext, ledger: str, statements: list[str], tool_name: str, kwargs: dict, max_statements: int
) -> None:
    result = getattr(integrity, tool_name)(ledger, **kwargs)
    assert result.status == "ok", result.summary
    assert sum(bool(FULL_DISTINCT.search(q)) for q in statements) <= 1
    assert len(statements) <= max_statements
