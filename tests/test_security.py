"""Security tests for run_sql / sample_rows: read-only, sandboxed, no ground truth."""

from __future__ import annotations

from pathlib import Path

import pytest

from jellybean.runtime.context import RunContext
from jellybean.skills._shared import data_tools as dt
from jellybean.skills._shared.catalog import connect
from jellybean.skills._shared.sql_guard import SQLRejectedError, validate_select

WRITE_OR_ADMIN = [
    "INSERT INTO events VALUES (1)",
    "UPDATE events SET amount = 0",
    "DELETE FROM events",
    "CREATE TABLE x AS SELECT 1",
    "DROP VIEW events",
    "COPY (SELECT 1) TO 'out.csv'",
    "ATTACH 'other.duckdb' AS other",
    "DETACH other",
    "INSTALL httpfs",
    "LOAD httpfs",
    "PRAGMA version",
    "SET threads = 1",
    "SELECT 1; DROP VIEW events",
    "EXPLAIN SELECT 1",
    "CALL pragma_version()",
    "EXPORT DATABASE 'dump'",
    "SELECT * FROM query('SELECT 42')",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_text('/etc/hostname')",
    "SELECT * FROM glob('/*')",
]


@pytest.mark.parametrize("query", WRITE_OR_ADMIN)
def test_run_sql_rejects_writes_and_admin(run_ctx: RunContext, query: str) -> None:
    result = dt.run_sql(query)
    assert result.status == "error"
    assert "rejected" in result.summary["error"].lower()


def test_allowed_queries(run_ctx: RunContext) -> None:
    assert dt.run_sql("WITH t AS (SELECT * FROM events) SELECT count(*) FROM t").status == "ok"
    assert dt.run_sql("SELECT 1 UNION ALL SELECT 2;").status == "ok"


def test_ground_truth_is_unreadable(run_ctx: RunContext) -> None:
    gt = run_ctx.settings.ground_truth_root / "anomalies.parquet"
    attempts = [
        f"SELECT * FROM read_parquet('{gt}')",
        f"SELECT * FROM '{gt}'",
        "SELECT * FROM read_parquet('../ground_truth/anomalies.parquet')",
        f"SELECT * FROM read_parquet('{run_ctx.settings.data_root}/../ground_truth/*.parquet')",
    ]
    for query in attempts:
        result = dt.run_sql(query)
        assert result.status == "error", query
        assert "TOP SECRET" not in result.to_json()


def test_sandbox_blocks_paths_outside_allowed_even_without_guard(run_ctx: RunContext, tmp_path: Path) -> None:
    """Second defence layer: the DuckDB connection itself refuses outside files."""
    outside = run_ctx.settings.ground_truth_root / "anomalies.parquet"
    con = connect()
    try:
        with pytest.raises(Exception, match=r"(?i)permission|disabled"):
            con.execute(f"SELECT * FROM read_parquet('{outside}')").fetchall()
        with pytest.raises(Exception, match=r"(?i)lock|configuration"):
            con.execute("SET enable_external_access = true")
        # Allowed: the data root and the run folder.
        assert con.execute("SELECT count(*) FROM events").fetchall()[0][0] == 1000
    finally:
        con.close()


def test_sandbox_allows_data_root_paths(run_ctx: RunContext) -> None:
    path = run_ctx.settings.data_root / "customers" / "part-0.parquet"
    result = dt.run_sql(f"SELECT count(*) FROM read_parquet('{path}')")
    assert result.status == "ok" and result.summary["rows"][0][0] == 50


def test_sample_rows_filter_cannot_escape(run_ctx: RunContext) -> None:
    gt = run_ctx.settings.ground_truth_root / "anomalies.parquet"
    for where in ["1=1; DROP VIEW events", f"1=1 UNION SELECT * FROM read_parquet('{gt}')"]:
        result = dt.sample_rows("events", where=where)
        assert result.status == "error"
        assert "TOP SECRET" not in result.to_json()


def test_validate_select_directly() -> None:
    assert validate_select(" SELECT 1 ; ") == "SELECT 1"
    with pytest.raises(SQLRejectedError):
        validate_select("")
    with pytest.raises(SQLRejectedError):
        validate_select("SELECT 'x/ground_truth/y'", forbidden_paths=[Path("/a/ground_truth")])
