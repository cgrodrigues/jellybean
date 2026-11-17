# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501  (long SQL assertions read better on one line)
"""Top-up ledger generator (requirements_2.md): sizes, invariants, anomalies and ground truth.

Generated once with the default arguments (the counts of the document are exact
for them); the checks query the written Parquet files with DuckDB, as an agent
would see them.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

from jellybean.datagen import GeneratorConfig, generate
from jellybean.datagen.topup import COLUMNS_BY_TYPE


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("topup")
    generate(GeneratorConfig(out=root / "data", truth=root / "truth", datasets=("topup",), with_readme=True))
    return root


@pytest.fixture(scope="module")
def con(base: Path) -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    d = base / "data"
    c.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{d}/topup/**/*.parquet', hive_partitioning=true)")
    # Clean view: no exact duplicates, normalised movement types.
    c.execute("CREATE VIEW t AS SELECT DISTINCT * REPLACE (upper(trim(movement_type)) AS movement_type) FROM raw")
    for name in ("promotions", "dealers", "voucher_batches"):
        c.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{d}/{name}/*.parquet')")
    c.execute(f"CREATE VIEW g AS SELECT * FROM read_parquet('{base}/truth/anomalies.parquet')")
    c.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{base}/truth/reconciliation_issues.parquet')")
    c.execute(
        "CREATE VIEW gm AS SELECT anomaly_id, entity_id AS movement_id FROM g WHERE entity_column = 'movement_id'"
    )
    return c


def _one(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchall()[0]


def _truth(base: Path, name: str):
    return json.loads((base / "truth" / name).read_text())


# ------------------------------------------------------------------ layout & sizes
def test_datasets_and_sizes(con: duckdb.DuckDBPyConnection, base: Path) -> None:
    rows = _one(con, "SELECT count(*) FROM raw")[0]
    assert 500_000 < rows < 600_000
    assert len(list((base / "data" / "topup").glob("date=*/part-0.parquet"))) == 90
    sizes = [_one(con, f"SELECT count(*) FROM {n}")[0] for n in ("promotions", "dealers", "voucher_batches")]
    assert sizes == [24, 600, 400]
    types = dict(con.execute("DESCRIBE raw").fetchall()[i][:2] for i in range(20))
    assert types["amount"] == types["balance_after"] == "DECIMAL(12,2)" and types["event_time"] == "TIMESTAMP"
    assert _one(con, "SELECT bool_and(msisdn LIKE '+351%') FROM t")[0]


def test_same_seed_same_ledger(base: Path, tmp_path: Path) -> None:
    generate(GeneratorConfig(out=tmp_path / "d", truth=tmp_path / "t", datasets=("topup",)))
    q = "SELECT * FROM read_parquet('{}/topup/**/*.parquet') ORDER BY movement_id, event_time, movement_type"
    a = duckdb.sql(q.format(base / "data")).df()
    b = duckdb.sql(q.format(tmp_path / "d")).df()
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- normal behaviour
def test_balance_invariant_and_no_negative_balances(con: duckdb.DuckDBPyConnection, base: Path) -> None:
    assert _one(con, "SELECT count(*) FROM t WHERE balance_before < 0 OR balance_after < 0")[0] == 0
    tr2 = next(x for x in _truth(base, "traps.json") if x["id"] == "TR2_sign_error")
    violations = dict(
        con.execute(
            "SELECT channel = 'USSD' AND date::VARCHAR IN (SELECT unnest(?)), count(*) FROM t "
            "WHERE balance_after <> balance_before + amount GROUP BY 1",
            [tr2["dates"]],
        ).fetchall()
    )
    # TR2 (5 days of USSD debits with a positive sign) plus R1's 300 scattered rows.
    assert violations[True] >= tr2["rows"] and 3_500 < tr2["rows"] < 4_700
    assert violations[True] + violations[False] == tr2["rows"] + 300


def test_paid_recharges_and_bonuses(con: duckdb.DuckDBPyConnection) -> None:
    faces = {r[0] for r in con.execute("SELECT DISTINCT face_value FROM t WHERE face_value IS NOT NULL").fetchall()}
    assert faces == {5, 10, 15, 20, 25, 50}
    above = _one(con, "SELECT count(*) FROM t WHERE movement_type LIKE 'RCH_%' AND amount <> face_value")[0]
    assert above == 80  # R4 only
    share = _one(
        con,
        "SELECT count(*) FILTER (WHERE movement_type = 'BONUS_RCH') / count(*) FILTER (WHERE movement_type LIKE 'RCH_%') "
        "FROM t",
    )[0]
    assert 0.37 < share < 0.45
    # Regular bonuses follow the promotion's rule exactly (the tz-shifted day can move a recharge out of dates).
    wrong = _one(
        con,
        """SELECT count(*) FROM t b JOIN t rc ON rc.movement_id = b.ref_movement_id
           JOIN promotions p ON p.promo_id = b.promo_id
           WHERE b.movement_type = 'BONUS_RCH' AND b.movement_id NOT IN (SELECT movement_id FROM gm)
             AND (rc.face_value < p.min_recharge OR rc.event_time::DATE NOT BETWEEN p.start_date - 1 AND p.end_date
                  OR b.amount <> least(round(rc.face_value * p.bonus_pct / 100, 2), p.max_bonus))""",
    )[0]
    assert wrong == 0
    over_cap = _one(
        con,
        """SELECT count(*) FROM (SELECT msisdn, promo_id, count(*) n FROM t WHERE promo_id IS NOT NULL
             AND movement_id NOT IN (SELECT movement_id FROM gm) GROUP BY ALL) x JOIN promotions USING (promo_id)
           WHERE n > max_per_customer""",
    )[0]
    assert over_cap == 0


def test_reversals_are_clawed_back_except_t2(con: duckdb.DuckDBPyConnection) -> None:
    missing = _one(
        con,
        """SELECT count(*) FROM t v JOIN t b ON b.ref_movement_id = v.ref_movement_id AND b.movement_type = 'BONUS_RCH'
           WHERE v.movement_type = 'REVERSAL' AND NOT EXISTS
             (SELECT 1 FROM t k WHERE k.movement_type = 'BONUS_CLAWBACK' AND k.ref_movement_id = b.movement_id)
             AND v.movement_id NOT IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T2')""",
    )[0]
    assert missing == 0


def test_dealers_vouchers_and_shares(con: duckdb.DuckDBPyConnection) -> None:
    chain_share = _one(
        con,
        "SELECT sum(amount) FILTER (WHERE dealer_type = 'CHAIN') / sum(amount) FROM t JOIN dealers USING (dealer_id) "
        "WHERE movement_type = 'RCH_DEALER'",
    )[0]
    assert 0.40 < chain_share < 0.50
    early = _one(
        con,
        """SELECT count(*) FROM t JOIN voucher_batches b ON b.batch_id = t.voucher_batch_id
           WHERE t.event_time::DATE <= b.distribution_date
             AND voucher_batch_id NOT IN (SELECT entity_id FROM g WHERE anomaly_id = 'T4')""",
    )[0]
    assert early == 0
    over = _one(
        con,
        "SELECT count(*) FROM (SELECT voucher_batch_id, count(*) n FROM t WHERE voucher_batch_id IS NOT NULL GROUP BY 1) "
        "JOIN voucher_batches b ON b.batch_id = voucher_batch_id WHERE n > b.n_vouchers",
    )[0]
    assert over == 0
    p2p, adv = _one(
        con,
        "SELECT count(DISTINCT msisdn) FILTER (WHERE movement_type = 'P2P_OUT'), "
        "count(DISTINCT msisdn) FILTER (WHERE movement_type = 'ADV_GRANT') FROM t",
    )
    assert p2p == 600 + 120 and adv == 1000 + 80  # 3% and 5% of customers, plus T6 feeders and T7


def test_structural_nulls(con: duckdb.DuckDBPyConnection) -> None:
    for column, types in COLUMNS_BY_TYPE.items():
        listed = ", ".join(f"'{x}'" for x in types)
        outside = _one(con, f"SELECT count(*) FROM t WHERE movement_type NOT IN ({listed}) AND {column} IS NOT NULL")
        inside = _one(con, f"SELECT count(*) FROM t WHERE movement_type IN ({listed}) AND {column} IS NULL")
        assert outside[0] == 0, column
        assert inside[0] == (135 if column == "reason_code" else 0), column  # 5% of adjustments lack a reason


def test_sparse_joins_cover_every_non_null_key(con: duckdb.DuckDBPyConnection) -> None:
    coverage = _one(
        con,
        """SELECT avg((p.promo_id IS NOT NULL)::INT) FILTER (WHERE t.promo_id IS NOT NULL),
                  avg((d.dealer_id IS NOT NULL)::INT) FILTER (WHERE t.dealer_id IS NOT NULL),
                  avg((b.batch_id IS NOT NULL)::INT) FILTER (WHERE t.voucher_batch_id IS NOT NULL),
                  avg((t.promo_id IS NULL)::INT), avg((t.dealer_id IS NULL)::INT), avg((t.voucher_batch_id IS NULL)::INT)
           FROM t LEFT JOIN promotions p USING (promo_id) LEFT JOIN dealers d USING (dealer_id)
           LEFT JOIN voucher_batches b ON b.batch_id = t.voucher_batch_id""",
    )
    assert coverage[:3] == (1.0, 1.0, 1.0)
    assert all(0.75 < share < 0.95 for share in coverage[3:])
    assert (
        _one(con, "SELECT count(*) FROM voucher_batches WHERE dealer_id NOT IN (SELECT dealer_id FROM dealers)")[0] == 0
    )


# --------------------------------------------------------------------- anomalies
def test_anomaly_sizes_and_disjoint_members(con: duckdb.DuckDBPyConnection) -> None:
    entities = dict(
        con.execute("SELECT anomaly_id, count(*) FROM g WHERE entity_column <> 'movement_id' GROUP BY 1").fetchall()
    )
    assert entities == {"T1": 3, "T2": 35, "T3": 1, "T4": 2, "T5": 4, "T6": 123, "T7": 80, "T8": 60}
    movements = dict(con.execute("SELECT anomaly_id, count(*) FROM gm GROUP BY 1").fetchall())
    assert movements["T1"] == 240 and movements["T3"] == 1500 and movements["T4"] == 600
    assert movements["T5"] == 1200 and movements["T8"] == 280 and movements["T7"] == 160
    shared = _one(
        con,
        "SELECT count(*) FROM (SELECT entity_id FROM g WHERE entity_column = 'msisdn' GROUP BY 1 HAVING count(DISTINCT anomaly_id) > 1)",
    )
    assert shared[0] == 0
    # Every planted movement exists in the data and respects the balance invariant.
    bad = _one(
        con,
        "SELECT count(*) FROM gm LEFT JOIN t USING (movement_id) WHERE t.movement_id IS NULL OR t.balance_after <> t.balance_before + t.amount",
    )
    assert bad[0] == 0


def test_t1_insider_adjustments(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        """SELECT agent_user, count(*), count(DISTINCT msisdn), max(hour(event_time)), any_value(reason_code)
           FROM t WHERE movement_id IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T1') GROUP BY 1"""
    ).fetchall()
    assert len(rows) == 3 and all(r[1] == 80 and r[2] == 15 and r[3] < 5 and r[4] == "OTHER" for r in rows)
    assert (
        _one(
            con,
            "SELECT count(DISTINCT msisdn) FROM t WHERE movement_id IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T1')",
        )[0]
        == 15
    )


def test_t2_cycles_without_clawback(con: duckdb.DuckDBPyConnection) -> None:
    cycles = con.execute(
        """SELECT msisdn, count(*) FILTER (WHERE movement_type = 'REVERSAL'), count(*) FILTER (WHERE movement_type = 'BONUS_CLAWBACK')
           FROM t WHERE movement_id IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T2') GROUP BY 1"""
    ).fetchall()
    assert len(cycles) == 35 and all(4 <= r[1] <= 8 and r[2] == 0 for r in cycles)


def test_t3_misconfigured_promotion(con: duckdb.DuckDBPyConnection, base: Path) -> None:
    n, customers, ratio, hours = _one(
        con,
        """SELECT count(*), count(DISTINCT b.msisdn), avg(b.amount / rc.amount), date_diff('hour', min(b.event_time), max(b.event_time))
           FROM t b JOIN t rc ON rc.movement_id = b.ref_movement_id
           WHERE b.movement_id IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T3')""",
    )
    assert n == 1500 and 1_350 < customers < 1_500 and ratio == 1 and hours <= 36
    assert _one(con, "SELECT bonus_pct FROM promotions WHERE promo_id = 'P017'")[0] == 10


def test_t4_vouchers_redeemed_before_distribution(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        """SELECT voucher_batch_id, count(*), count(DISTINCT msisdn), date_diff('minute', min(event_time), max(event_time)),
                  date_diff('day', max(event_time)::DATE, any_value(b.distribution_date)), any_value(b.n_vouchers)
           FROM t JOIN voucher_batches b ON b.batch_id = t.voucher_batch_id
           WHERE movement_id IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T4') GROUP BY 1"""
    ).fetchall()
    assert len(rows) == 2 and all(r[1] == 300 and r[3] <= 240 and r[4] == 6 and r[5] == 300 for r in rows)
    assert (
        _one(
            con,
            "SELECT count(DISTINCT msisdn) FROM t WHERE movement_id IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T4')",
        )[0]
        == 20
    )


def test_t5_dealers(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        """WITH x AS (SELECT dealer_id, msisdn, sum(amount) v, bool_and(hour(event_time) = d.closing_hour - 1) last_hour
                      FROM t JOIN dealers d USING (dealer_id)
                      WHERE movement_id IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T5') GROUP BY dealer_id, msisdn)
           SELECT dealer_id, sum(v) FILTER (WHERE rk <= 10) / sum(v), bool_and(last_hour)
           FROM (SELECT *, row_number() OVER (PARTITION BY dealer_id ORDER BY v DESC) rk FROM x) GROUP BY 1"""
    ).fetchall()
    assert len(rows) == 4 and all(0.69 < r[1] < 0.72 and r[2] for r in rows)
    kinds = {
        r[0]
        for r in con.execute(
            "SELECT dealer_type FROM dealers WHERE dealer_id IN (SELECT entity_id FROM g WHERE anomaly_id = 'T5')"
        ).fetchall()
    }
    assert kinds == {"INDEPENDENT"}


def test_t6_collectors(con: duckdb.DuckDBPyConnection) -> None:
    top = con.execute(
        "SELECT count(DISTINCT counterparty_msisdn) k FROM t WHERE movement_type = 'P2P_IN' GROUP BY msisdn ORDER BY k DESC LIMIT 4"
    ).fetchall()
    assert [r[0] for r in top[:3]] == [40, 40, 40] and top[3][0] < 5
    late = _one(
        con,
        """SELECT count(*) FROM t v JOIN t rc ON rc.movement_id = v.ref_movement_id
           WHERE v.movement_id IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T6') AND v.movement_type = 'REVERSAL'
             AND date_diff('second', rc.event_time, v.event_time) BETWEEN 3 * 86400 AND 10 * 86400""",
    )[0]
    assert late == 120


def test_t7_advance_then_silence(con: duckdb.DuckDBPyConnection) -> None:
    after = _one(
        con,
        """WITH grant_ AS (SELECT msisdn, event_time FROM t WHERE movement_type = 'ADV_GRANT'
                          AND movement_id IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T7'))
           SELECT count(*), max(grant_.event_time) FROM t JOIN grant_ USING (msisdn) WHERE t.event_time > grant_.event_time + INTERVAL 1 HOUR""",
    )
    assert after[0] == 0
    last = _one(
        con,
        "SELECT max(event_time)::DATE FROM t WHERE movement_id IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T7')",
    )[0]
    assert (last - pd.Timestamp("2026-10-01").date()).days <= 60


def test_t8_bonuses_not_eligible(con: duckdb.DuckDBPyConnection) -> None:
    below, out_of_dates, n, right_pct = _one(
        con,
        """SELECT sum((rc.face_value < p.min_recharge)::INT),
                  sum((rc.event_time::DATE NOT BETWEEN p.start_date AND p.end_date)::INT), count(*),
                  sum((b.amount = least(round(rc.face_value * p.bonus_pct / 100, 2), p.max_bonus))::INT)
           FROM t b JOIN t rc ON rc.movement_id = b.ref_movement_id JOIN promotions p ON p.promo_id = b.promo_id
           WHERE b.movement_type = 'BONUS_RCH' AND b.movement_id IN (SELECT movement_id FROM gm WHERE anomaly_id = 'T8')""",
    )
    assert n == 140 and below + out_of_dates == 140 and right_pct == 140


# ------------------------------------------------------- reconciliation & quality
def test_reconciliation_issues(con: duckdb.DuckDBPyConnection) -> None:
    assert dict(con.execute("SELECT issue_id, count(*) FROM r GROUP BY 1").fetchall()) == {
        "R1": 300,
        "R2": 300,
        "R3": 100,
        "R4": 80,
        "R5": 60,
    }
    assert (
        _one(
            con,
            "SELECT count(*) FROM (SELECT payment_ref FROM t WHERE payment_ref IS NOT NULL GROUP BY 1 HAVING count(*) > 1)",
        )[0]
        == 150
    )
    orphans = _one(
        con,
        "SELECT count(*) FROM t WHERE ref_movement_id IS NOT NULL AND ref_movement_id NOT IN (SELECT movement_id FROM t)",
    )[0]
    assert orphans == 100
    r1 = _one(
        con,
        "SELECT count(*) FROM t JOIN r USING (movement_id) WHERE issue_id = 'R1' AND balance_after <> balance_before + amount",
    )[0]
    assert r1 == 300


def test_quality_issues_and_traps(con: duckdb.DuckDBPyConnection, base: Path) -> None:
    raw, distinct = _one(con, "SELECT count(*), count(DISTINCT movement_id) FROM raw")
    assert raw - distinct == round(0.005 * distinct)
    assert _one(con, "SELECT count(DISTINCT movement_id) FROM raw WHERE movement_type <> upper(trim(movement_type))")[
        0
    ] == round(0.01 * distinct)
    issues = {i["id"]: i for i in _truth(base, "quality_issues.json")}
    assert {
        "topup_duplicate_rows",
        "topup_timezone_shift",
        "topup_null_reason_code",
        "topup_structural_nulls",
        "topup_sign_error",
    } <= set(issues)
    traps = {t["id"]: t for t in _truth(base, "traps.json")}
    assert {t["expected_verdict"] for t in traps.values()} == {"explained", "artifact"}
    day, _p009, total = _one(
        con,
        "SELECT date, count(*) FILTER (WHERE promo_id = 'P009'), count(*) FROM t WHERE movement_type = 'PROMO_CREDIT' GROUP BY 1 ORDER BY 3 DESC LIMIT 1",
    )
    normal = _one(
        con, "SELECT median(n) FROM (SELECT date, count(*) n FROM t WHERE movement_type = 'PROMO_CREDIT' GROUP BY 1)"
    )[0]
    assert str(day) == traps["TR1_legitimate_campaign"]["date"] and 2.5 < total / normal < 3.8
    assert len(traps["TR3_retail_chains"]["entity_ids"]) == 30


def test_expected_spec_and_readmes(base: Path) -> None:
    spec = yaml.safe_load((base / "truth" / "expected_spec_topup.yaml").read_text())
    assert len(spec["joins"]) == 6 and spec["grain"]["columns"] == ["movement_id"]
    assert spec["entities"] == ["msisdn", "dealer_id", "agent_user", "voucher_batch_id", "promo_id"]
    readme = (base / "data" / "topup" / "README.md").read_text()
    assert "BONUS_CLAWBACK" in readme and "Consumption" in readme
    assert not any(word in readme.lower() for word in ("anomal", "fraud", "trap", "T1"))


# ----------------------------------------------------------- options and merging
def test_opaque_codes(tmp_path: Path) -> None:
    generate(
        GeneratorConfig(
            out=tmp_path / "d", truth=tmp_path / "t", customers=1000, scale=0.2, datasets=("topup",), opaque_codes=True
        )
    )
    types = {
        r[0].strip()
        for r in duckdb.sql(
            f"SELECT DISTINCT movement_type FROM read_parquet('{tmp_path}/d/topup/**/*.parquet')"
        ).fetchall()
    }
    assert len(types) == 16 and all(t.isdigit() for t in types)
    mapping = yaml.safe_load((tmp_path / "t" / "expected_spec_topup.yaml").read_text())["opaque_codes"]
    assert set(mapping) == {"movement_type", "channel", "reason_code"}


def test_ground_truth_is_merged_by_dataset(tmp_path: Path) -> None:
    small = {"customers": 1000, "days": 7, "scale": 0.2}
    generate(GeneratorConfig(out=tmp_path / "d", truth=tmp_path / "t", datasets=("topup",), **small))
    generate(GeneratorConfig(out=tmp_path / "d", truth=tmp_path / "t", **small))  # telecom afterwards
    gt = pd.read_parquet(tmp_path / "t" / "anomalies.parquet")
    assert set(gt["dataset"]) == {"traffic", "orders", "topup"}
    details = json.loads((tmp_path / "t" / "anomaly_details.json").read_text())
    assert {"A", "E", "topup"} <= set(details)
    traps = {t["id"] for t in json.loads((tmp_path / "t" / "traps.json").read_text())}
    assert {"units_error", "TR2_sign_error"} <= traps
    # Regenerating topup keeps the telecom truth.
    generate(GeneratorConfig(out=tmp_path / "d", truth=tmp_path / "t", datasets=("topup",), overwrite=True, **small))
    assert set(pd.read_parquet(tmp_path / "t" / "anomalies.parquet")["dataset"]) == {"traffic", "orders", "topup"}


def test_ground_truth_without_telecom_anomalies(tmp_path: Path) -> None:
    """cust_segmentation alone plants no anomaly: the truth file still has its columns."""
    small = {"customers": 1000, "scale": 0.2}
    generate(
        GeneratorConfig(out=tmp_path / "d", truth=tmp_path / "t", datasets=("cust_segmentation", "topup"), **small)
    )
    gt = pd.read_parquet(tmp_path / "t" / "anomalies.parquet")
    assert set(gt["dataset"]) == {"topup"}


def test_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown datasets"):
        GeneratorConfig(out=tmp_path / "d", truth=tmp_path / "t", datasets=("weather",)).validate()
    with pytest.raises(ValueError, match="topup_days"):
        GeneratorConfig(out=tmp_path / "d", truth=tmp_path / "t", datasets=("topup",), topup_days=30).validate()
