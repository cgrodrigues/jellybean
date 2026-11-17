# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Tests for key normalisation strategies and semantic typing rules."""

from __future__ import annotations

import duckdb
import pytest

from jellybean.skills._shared.column_stats import ColumnStats, classify_column, name_tokens
from jellybean.skills._shared.keys import (
    UnknownStrategyError,
    detect_prefixes,
    is_valid_strategy,
    prefix_strategy,
    strategy_sql,
)


@pytest.mark.parametrize(
    ("strategy", "value", "expected"),
    [
        ("none", "+351912", "+351912"),
        ("trim", "  ab ", "ab"),
        ("lower_trim", " AbC ", "abc"),
        ("strip_leading_zeros", "000123", "123"),
        ("digits_only", "+351-91 2", "351912"),
        ("strip_prefix_plus351", "+351912345678", "912345678"),
        ("strip_prefix_plus351", "912345678", "912345678"),  # unchanged when absent
        ("strip_prefix_PT", "PT123", "123"),
        ("strip_prefix:MCH-", "MCH-00123", "00123"),
        ("strip_prefix:ACC_", "ACC_9", "9"),
    ],
)
def test_strategy_sql(strategy: str, value: str, expected: str) -> None:
    con = duckdb.connect()
    literal = "'" + value.replace("'", "''") + "'"
    assert con.execute(f"SELECT {strategy_sql(strategy, literal)}").fetchall()[0][0] == expected


def test_strategy_names() -> None:
    assert prefix_strategy("+351") == "strip_prefix_plus351"
    assert is_valid_strategy("strip_prefix_plus351") and not is_valid_strategy("explode")
    with pytest.raises(UnknownStrategyError):
        strategy_sql("explode", "x")
    with pytest.raises(ValueError):
        prefix_strategy("'; DROP")


def test_detect_prefixes() -> None:
    targets = {f"9{i:08d}" for i in range(100)}
    values = [f"+351{t}" for t in sorted(targets)]
    assert detect_prefixes(values, targets) == ["+351"]
    assert detect_prefixes(["abc", "def"], targets) == []
    # Prefixes with separators (regression: "MCH-" used to be rejected).
    codes = {f"{i:05d}" for i in range(100)}
    assert detect_prefixes([f"MCH-{c}" for c in sorted(codes)], codes) == ["MCH-"]
    assert prefix_strategy("MCH-") == "strip_prefix:MCH-" and is_valid_strategy("strip_prefix:MCH-")
    assert prefix_strategy("+351") == "strip_prefix_plus351"  # legacy name kept


def _col(name: str, family: str, distinct: int, non_null: int = 1000, samples=(), avg_length=None) -> ColumnStats:
    return ColumnStats(
        name=name,
        type=family.upper(),
        family=family,
        rows=1000,
        non_null=non_null,
        distinct=distinct,
        sample_values=list(samples),
        avg_length=avg_length,
    )


@pytest.mark.parametrize(
    ("col", "expected"),
    [
        (_col("customer_id", "string", 900, samples=["912345678", "961234567"]), "identifier"),
        (_col("msisdn", "string", 900, samples=["+351912345678"]), "identifier"),
        (_col("event_day", "string", 30, samples=["2026-10-01", "2026-10-02"]), "timestamp"),
        (_col("compact_day", "string", 30, samples=["20261001", "20261002"]), "timestamp"),
        (_col("ts", "temporal", 1000), "timestamp"),
        (_col("flag", "string", 2, samples=["Y", "N"]), "boolean"),
        (_col("region_code", "integer", 96), "category"),
        (_col("segment", "string", 12, samples=["Business"]), "category"),
        (_col("phone", "string", 50, samples=["+351910000001", "+351910000002"]), "identifier"),
        (_col("ticket", "string", 50, samples=["C0000012", "C0000099"]), "identifier"),
        (_col("comment", "string", 900, samples=["x" * 80], avg_length=80), "free_text"),
        (_col("duration_sec", "integer", 4000), "measure"),
        (_col("charge", "float", 400), "measure"),
        (_col("latitude", "float", 900), "geo"),
        (_col("empty", "string", 0, non_null=0), "unknown"),
    ],
)
def test_classify_column(col: ColumnStats, expected: str) -> None:
    assert classify_column(col).semantic_type == expected


def test_name_tokens() -> None:
    assert name_tokens("customerId") == {"customer", "id"}
    assert name_tokens("Customer ID") == {"customer", "id"}
