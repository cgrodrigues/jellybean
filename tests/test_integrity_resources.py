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


# --------------------------------------------------------------------------- #
# Column statistics, duplicate rows and size caps (performance audit)         #
# --------------------------------------------------------------------------- #
def test_column_stats_are_cached_for_the_run(run_ctx: RunContext, ledger: str) -> None:
    from jellybean.skills._shared.column_stats import compute_column_stats

    log: list[str] = []
    con = RecordingConnection(catalog.connect(), log)
    try:
        first = compute_column_stats(con, '"ledger"', with_top=False, with_samples=False)
        scans = len(log)
        second = compute_column_stats(con, '"ledger"', with_top=False, with_samples=False)
        assert len(log) == scans  # nothing is recomputed
        assert [c.to_dict() for c in first[1]] == [c.to_dict() for c in second[1]]
        second[1][0].distinct = -1  # callers get copies: the cache is not corrupted
        assert compute_column_stats(con, '"ledger"', with_top=False, with_samples=False)[1][0].distinct != -1
        # Temporary tables can change between calls: never cached.
        con.execute("CREATE TEMP TABLE _t AS SELECT 1 AS a")
        compute_column_stats(con, "_t", with_top=False, with_samples=False)
        before = len(log)
        compute_column_stats(con, "_t", with_top=False, with_samples=False)
        assert len(log) > before
    finally:
        con.close()


def test_large_table_stats_keep_exact_distinct_counts(run_ctx: RunContext, ledger: str) -> None:
    from jellybean.skills._shared.column_stats import clear_stats_cache, compute_column_stats

    con = catalog.connect()
    try:
        _, exact = compute_column_stats(con, '"ledger"', with_top=False, with_samples=False)
        clear_stats_cache()
        run_ctx.settings.config.profiling.exact_stats_max_rows = 100  # the ledger is now "large"
        log: list[str] = []
        _, large = compute_column_stats(RecordingConnection(con, log), '"ledger"', with_top=False, with_samples=False)
    finally:
        con.close()
    assert any("approx_quantile" in q for q in log) and not any("quantile_cont" in q for q in log)
    assert {c.name: c.distinct for c in large} == {c.name: c.distinct for c in exact}
    assert {c.name: c.non_null for c in large} == {c.name: c.non_null for c in exact}
    for a, b in zip(exact, large, strict=True):  # approximate quantiles stay close
        if a.quantiles:
            assert abs(float(a.quantiles["p50"]) - float(b.quantiles["p50"])) <= 0.1 * (float(a.max) - float(a.min))


def test_duplicate_rows_are_counted_exactly_without_a_full_distinct(run_ctx: RunContext, settings) -> None:
    from jellybean.skills._shared.column_stats import count_duplicate_rows

    folder = settings.data_root / "dups"
    folder.mkdir()
    con = duckdb.connect()
    con.execute(
        f"""COPY (SELECT range % 900 AS id, 'v' || (range % 900) AS label FROM range(1000))
            TO '{folder / "part-0.parquet"}' (FORMAT parquet)"""
    )
    con.close()
    log: list[str] = []
    con = RecordingConnection(catalog.connect(), log)
    try:
        assert count_duplicate_rows(con, '"dups"') == 100
        assert count_duplicate_rows(con, '"dups"') == 100  # cached
    finally:
        con.close()
    # The full-width DISTINCT only ever runs on the rows whose hash repeats.
    assert all("hash(_t) IN" in q for q in log if FULL_DISTINCT.search(q))


def test_hdbscan_is_fitted_on_a_sample_above_the_cap() -> None:
    import numpy as np

    from jellybean.skills._shared.detection import cluster_labels

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (300, 2)), rng.normal(5, 0.1, (300, 2))])
    capped = cluster_labels(X, "hdbscan", {}, seed=3, max_rows=200)
    assert len(capped) == 600 and int((capped != -1).sum()) <= 200  # rows outside the sample are not clustered
    assert set(capped[capped != -1]) == {0, 1}
    assert (capped == cluster_labels(X, "hdbscan", {}, seed=3, max_rows=200)).all()  # seeded
    assert int((cluster_labels(X, "hdbscan", {}, seed=3, max_rows=10_000) != -1).sum()) > 500  # below the cap: all rows


def test_grouping_of_pooled_outliers_is_capped(run_ctx: RunContext) -> None:
    import numpy as np
    import pandas as pd

    from jellybean.skills.detector import tools as det

    run_ctx.settings.config.detection.group_max_pool = 100
    rng = np.random.default_rng(1)
    strong = rng.normal(0, 0.05, (100, 3)) + np.array([8.0, 0.0, 0.0])  # the 100 largest deviations, one direction
    weak = rng.normal(0, 0.05, (400, 3)) + np.array([0.0, 1.0, 0.0])
    matrix = pd.DataFrame(np.vstack([strong, weak]), index=[f"e{i}" for i in range(500)])
    groups = det._group_entities(matrix, list(matrix.index), 0.5, 3)
    grouped = {e for g in groups for e in g}
    assert grouped == {f"e{i}" for i in range(100)}  # only the most deviating entities are compared pairwise


def test_plots_thin_the_population_but_keep_what_matters() -> None:
    import numpy as np
    import pandas as pd

    from jellybean.skills._shared.plots import _thin

    always = np.zeros(10_000, dtype=bool)
    always[:50] = True
    keep = _thin(pd.RangeIndex(10_000), always, limit=1000, seed=4)
    assert int(keep.sum()) == 1000 and keep[:50].all()
    assert (keep == _thin(pd.RangeIndex(10_000), always, limit=1000, seed=4)).all()
    assert _thin(pd.RangeIndex(100), np.zeros(100, dtype=bool), limit=1000).all()  # small populations are untouched
