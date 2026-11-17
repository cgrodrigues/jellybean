# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Generator tests: determinism, planted anomalies and planted problems."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

from jellybean.datagen import GeneratorConfig, generate

SMALL = {"customers": 2000, "days": 7, "scale": 0.4}


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate the same small dataset twice and a third time with another seed."""
    roots = {}
    for label, seed in (("a", 42), ("b", 42), ("c", 7)):
        base = tmp_path_factory.mktemp(f"gen_{label}")
        generate(GeneratorConfig(out=base / "data", truth=base / "truth", seed=seed, with_readme=True, **SMALL))
        roots[label] = base
    return roots


def _read(base: Path, dataset: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        f"SELECT * FROM read_parquet('{base}/data/{dataset}/**/*.parquet', hive_partitioning=true) ORDER BY ALL"
    ).df()
    con.close()
    return df


@pytest.fixture(scope="module")
def con(generated: dict[str, Path]) -> duckdb.DuckDBPyConnection:
    base = generated["a"]
    c = duckdb.connect()
    c.execute(
        f"CREATE VIEW t AS SELECT * FROM read_parquet('{base}/data/traffic/**/*.parquet', hive_partitioning=true)"
    )
    c.execute(f"CREATE VIEW s AS SELECT * FROM read_parquet('{base}/data/cust_segmentation/*.parquet')")
    c.execute(f"CREATE VIEW o AS SELECT * FROM read_parquet('{base}/data/orders/*.parquet')")
    c.execute(f"CREATE VIEW g AS SELECT * FROM read_parquet('{base}/truth/anomalies.parquet')")
    return c


def _one(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchall()[0]


@pytest.mark.parametrize("dataset", ["traffic", "cust_segmentation", "orders"])
def test_same_seed_same_data(generated: dict[str, Path], dataset: str) -> None:
    pd.testing.assert_frame_equal(_read(generated["a"], dataset), _read(generated["b"], dataset))


def test_ground_truth_is_deterministic(generated: dict[str, Path]) -> None:
    for name in ("anomalies.parquet",):
        pd.testing.assert_frame_equal(
            pd.read_parquet(generated["a"] / "truth" / name), pd.read_parquet(generated["b"] / "truth" / name)
        )
    for name in ("quality_issues.json", "traps.json", "expected_spec.yaml", "anomaly_details.json"):
        assert (generated["a"] / "truth" / name).read_text() == (generated["b"] / "truth" / name).read_text()


def test_different_seed_different_data(generated: dict[str, Path]) -> None:
    assert not _read(generated["a"], "orders").equals(_read(generated["c"], "orders"))


def test_layout_and_files(generated: dict[str, Path]) -> None:
    base = generated["a"]
    partitions = sorted(p.name for p in (base / "data" / "traffic").iterdir() if p.is_dir())
    assert partitions == [f"date=2026-10-0{d}" for d in range(1, 8)]
    # Partitioned by day and hour (default).
    hours = sorted(p.name for p in (base / "data" / "traffic" / "date=2026-10-02").iterdir())
    assert hours[0] == "hour=00" and hours[-1] == "hour=23"
    assert all((base / "data" / "traffic" / "date=2026-10-02" / h / "part-0.parquet").exists() for h in hours)
    for name in ("traffic", "cust_segmentation", "orders"):
        assert (base / "data" / name / "README.md").exists()
    for name in ("anomalies.parquet", "quality_issues.json", "expected_spec.yaml", "traps.json"):
        assert (base / "truth" / name).exists()
    assert not any("truth" in str(p) for p in (base / "data").rglob("*"))


def test_traffic_columns(con: duckdb.DuckDBPyConnection) -> None:
    cols = [r[0] for r in con.execute("DESCRIBE t").fetchall()]
    assert cols == [
        "call_id",
        "msisdn",
        "destination_number",
        "destination_country",
        "start_time",
        "duration_sec",
        "charge",
        "cell_id",
        "imei",
        "call_type",
        "date",
        "hour",
    ]


def test_anomaly_sizes(con: duckdb.DuckDBPyConnection) -> None:
    sizes = dict(con.execute("SELECT anomaly_id, count(DISTINCT entity_id) FROM g GROUP BY 1").fetchall())
    assert sizes == {"A": 25, "B": 15, "C": 40, "D": 30, "E": 1, "F": 60}


def test_join_difficulties(con: duckdb.DuckDBPyConnection) -> None:
    # msisdn has +351, customer_id does not.
    assert _one(con, "SELECT bool_and(msisdn LIKE '+351%') FROM t")[0]
    assert _one(con, "SELECT bool_or(customer_id LIKE '+%') FROM s")[0] is False
    # 3% of traffic customers are unknown to cust_segmentation.
    coverage = _one(
        con,
        """SELECT avg((s.customer_id IS NOT NULL)::INT) FROM (SELECT DISTINCT msisdn FROM t) x
                            LEFT JOIN (SELECT DISTINCT customer_id FROM s) s ON substr(x.msisdn, 5) = s.customer_id""",
    )[0]
    assert coverage == pytest.approx(0.97, abs=0.001)
    # 15 duplicated keys.
    assert _one(con, "SELECT count(*) FROM (SELECT customer_id FROM s GROUP BY 1 HAVING count(*) > 1)")[0] == 15
    # region_code is an integer that only partially overlaps cell_id.
    assert _one(con, "SELECT typeof(region_code) FROM s LIMIT 1")[0] == "INTEGER"
    overlap = _one(
        con,
        """SELECT avg((region_code IN (SELECT DISTINCT cell_id FROM t WHERE cell_id IS NOT NULL))::INT)
                           FROM (SELECT DISTINCT region_code FROM s)""",
    )[0]
    assert 0.1 < overlap < 0.9


def test_quality_issues(con: duckdb.DuckDBPyConnection, generated: dict[str, Path]) -> None:
    n, distinct_ids, null_cell, null_imei, negative = _one(
        con,
        """SELECT count(*), count(DISTINCT call_id), avg((cell_id IS NULL)::INT),
                       avg((imei IS NULL)::INT), sum((duration_sec < 0)::INT) FROM t""",
    )
    assert (n - distinct_ids) / n == pytest.approx(0.005, abs=0.001)
    assert null_cell == pytest.approx(0.02, abs=0.003)
    assert null_imei == pytest.approx(0.01, abs=0.002)
    assert negative >= 200
    variants = {r[0] for r in con.execute("SELECT DISTINCT segment FROM s").fetchall()}
    assert {"Business", "business", " BUSINESS  "} <= variants

    issues = {i["id"]: i for i in json.loads((generated["a"] / "truth" / "quality_issues.json").read_text())}
    assert {
        "null_cell_id",
        "null_imei",
        "negative_duration",
        "timezone_shift",
        "duration_in_ms",
        "duplicate_rows",
        "segment_spelling",
        "duplicate_keys",
    } <= set(issues)
    assert len(issues["negative_duration"]["call_ids"]) == 200


def test_timezone_and_units_files(con: duckdb.DuckDBPyConnection, generated: dict[str, Path]) -> None:
    issues = {i["id"]: i for i in json.loads((generated["a"] / "truth" / "quality_issues.json").read_text())}
    units_date = issues["duration_in_ms"]["file"].split("date=")[1].split("/")[0]
    medians = dict(con.execute("SELECT date::VARCHAR, median(duration_sec) FROM t GROUP BY 1").fetchall())
    assert medians[units_date] > 100 * max(v for k, v in medians.items() if k != units_date)
    tz_date = issues["timezone_shift"]["file"].split("date=")[1].split("/")[0]
    mismatched = dict(
        con.execute(
            "SELECT date::VARCHAR, count(*) FROM t WHERE date <> CAST(start_time AS DATE) GROUP BY 1"
        ).fetchall()
    )
    assert set(mismatched) == {tz_date}


def test_traps(generated: dict[str, Path], con: duckdb.DuckDBPyConnection) -> None:
    traps = {t["id"]: t for t in json.loads((generated["a"] / "truth" / "traps.json").read_text())}
    assert traps["units_error"]["expected_verdict"] == "artifact"
    trap = traps["imputation_artifact"]
    assert len(trap["entity_ids"]) == 300
    ids = ",".join(f"'{i}'" for i in trap["entity_ids"])
    share = _one(con, f"SELECT avg((msisdn IN ({ids}))::INT) FROM t WHERE cell_id IS NULL")[0]
    assert share > 0.5  # nulls are concentrated in the trap customers


def test_anomaly_signatures(con: duckdb.DuckDBPyConnection, generated: dict[str, Path]) -> None:
    # A: long night calls to premium-rate countries.
    a = _one(
        con,
        """SELECT count(DISTINCT t.msisdn) FROM t JOIN g ON g.anomaly_id = 'A' AND g.entity_id = t.msisdn
                     WHERE hour(start_time) BETWEEN 1 AND 4 AND destination_country IN ('CU','SO','TV','NU')
                       AND abs(duration_sec) >= 1200""",
    )[0]
    assert a == 25
    # B: hundreds of 1-3 s calls each (1000-3000 on the file stored in milliseconds).
    b_min = _one(
        con,
        """SELECT min(c) FROM (SELECT t.msisdn, count(*) c FROM t JOIN g ON g.anomaly_id = 'B'
                     AND g.entity_id = t.msisdn WHERE duration_sec BETWEEN 1 AND 3 OR duration_sec IN (1000, 2000, 3000)
                     GROUP BY 1)""",
    )[0]
    assert b_min >= 250
    # C: 3 IMEIs, 2 cells, outgoing only.
    imeis, cells, outgoing = _one(
        con,
        """SELECT count(DISTINCT imei), count(DISTINCT cell_id),
                     bool_and(call_type = 'outgoing') FROM t JOIN g ON g.anomaly_id = 'C' AND g.entity_id = t.msisdn""",
    )
    assert (imeis, cells, outgoing) == (3, 2, True)
    # D: 4 devices, 3 addresses, high amounts.
    dev, addr, amount = _one(
        con,
        """SELECT count(DISTINCT device_id), count(DISTINCT shipping_address_id),
                     min(amount) FROM o JOIN g ON g.anomaly_id = 'D' AND g.entity_id = o.customer_id""",
    )
    assert (dev, addr) == (4, 3) and amount >= 900
    # E: the planted cell/day/6h window is far above the usual hourly volume.
    details = json.loads((generated["a"] / "truth" / "anomaly_details.json").read_text())["E"]
    spike = _one(
        con,
        f"""SELECT count(*) / 6 FROM t WHERE cell_id = {details["cell_id"]}
                     AND date = DATE '{details["date"]}' AND hour(start_time) BETWEEN 10 AND 15""",
    )[0]
    typical = _one(
        con, "SELECT median(c) FROM (SELECT cell_id, date, hour(start_time), count(*) c FROM t GROUP BY ALL)"
    )[0]
    assert spike > 10 * typical
    # F: recorded as Consumer/basic but with a business-like international share.
    seg = _one(
        con,
        """SELECT bool_and(trim(lower(segment)) = 'consumer' AND tariff_plan = 'basic') FROM s
                     JOIN g ON g.anomaly_id = 'F' AND '+351' || s.customer_id = g.entity_id""",
    )[0]
    assert seg
    f_intl, consumer_intl = _one(
        con,
        """SELECT
          avg((destination_country <> 'PT')::INT) FILTER (WHERE g.entity_id IS NOT NULL),
          avg((destination_country <> 'PT')::INT) FILTER (
              WHERE g.entity_id IS NULL AND trim(lower(s.segment)) = 'consumer')
        FROM t JOIN s ON substr(t.msisdn, 5) = s.customer_id
        LEFT JOIN g ON g.anomaly_id = 'F' AND g.entity_id = t.msisdn""",
    )
    assert f_intl > 3 * consumer_intl


def test_expected_spec(generated: dict[str, Path]) -> None:
    spec = yaml.safe_load((generated["a"] / "truth" / "expected_spec.yaml").read_text())
    join = spec["joins"][0]
    assert join["keys"] == [["msisdn", "customer_id"]]
    assert join["key_normalization"] == ["strip_prefix_plus351"]
    assert spec["entity"]["column"] == "msisdn" and spec["time_column"] == "start_time"


def test_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        GeneratorConfig(out=tmp_path / "data", truth=tmp_path / "data" / "gt").validate()
    with pytest.raises(ValueError):
        GeneratorConfig(out=tmp_path / "d", truth=tmp_path / "t", days=3).validate()


def test_refuses_to_overwrite_without_flag(generated: dict[str, Path]) -> None:
    base = generated["a"]
    with pytest.raises(FileExistsError):
        generate(GeneratorConfig(out=base / "data", truth=base / "truth", **SMALL))


def test_partition_hour_matches_the_true_time(con: duckdb.DuckDBPyConnection, generated: dict[str, Path]) -> None:
    """Rows sit in the partition of their (true) hour, except on the time-zone-shifted day."""
    issues = {i["id"]: i for i in json.loads((generated["a"] / "truth" / "quality_issues.json").read_text())}
    tz_date = issues["timezone_shift"]["file"].split("date=")[1].split("/")[0]
    mismatched = dict(
        con.execute("SELECT date::VARCHAR, count(*) FROM t WHERE hour(start_time) <> hour GROUP BY 1").fetchall()
    )
    assert set(mismatched) == {tz_date}


def test_daily_partitioning_option(tmp_path: Path) -> None:
    generate(
        GeneratorConfig(out=tmp_path / "d", truth=tmp_path / "t", customers=1000, days=7, scale=0.1, partition="day")
    )
    day = tmp_path / "d" / "traffic" / "date=2026-10-01"
    assert [p.name for p in day.iterdir()] == ["part-0.parquet"]
