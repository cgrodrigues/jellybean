"""Prepaid top-up ledger with planted anomalies (``requirements_2.md``).

One row per movement on a customer's wallet (``topup``, partitioned by
``date=YYYY-MM-DD``), plus three reference tables (``promotions``, ``dealers``,
``voucher_batches``). The customers are the population of the telecom
generator (same ``msisdn`` with the ``+351`` prefix as ``traffic``), so the
ledger joins ``cust_segmentation`` like ``traffic`` does.

On top of a realistic background (section 5) the generator plants:

* eight behavioural anomalies T1-T8 (section 6), each on its own entity;
* five reconciliation problems R1-R5 and quality problems (section 7);
* three traps for the critic TR1-TR3 (section 8).

Planted anomalies respect the balance invariant
(``balance_after = balance_before + amount``) and referential integrity: only
behavioural analysis can find them. Counts are exact for the default arguments
(20,000 customers, 90 days, scale 1.0). For other sizes the volumes and the
size of T3 scale with ``customers * scale``; the other anomalies keep their size.

Consumption (calls, data) is not in the ledger, so balances are only consistent
row by row, not between consecutive movements.

The generator draws from its own random stream (derived from the seed), so
adding the ledger never changes the telecom datasets.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------- #
# Codes                                                                       #
# --------------------------------------------------------------------------- #
MOVEMENT_TYPES = [
    "RCH_VOUCHER",
    "RCH_DEALER",
    "RCH_ATM",
    "RCH_APP",
    "BONUS_RCH",
    "PROMO_CREDIT",
    "ADJ_CREDIT",
    "ADJ_DEBIT",
    "REVERSAL",
    "BONUS_CLAWBACK",
    "P2P_OUT",
    "P2P_IN",
    "ADV_GRANT",
    "ADV_RECOVERY",
    "BUNDLE_PURCHASE",
    "EXPIRY",
]
TOPUP_CHANNELS = ["VOUCHER", "POS", "ATM", "APP", "USSD", "BACKOFFICE", "SYSTEM"]
REASON_CODES = [
    "BILLING_ERROR",
    "NETWORK_ISSUE",
    "GOODWILL",
    "COMPENSATION",
    "CORRECTION",
    "OTHER",
    "PAYMENT_FAILED",
    "CUSTOMER_REQUEST",
    "DUPLICATE_CHARGE",
    "CARD_CHARGEBACK",
]
ADJ_REASONS, ADJ_REASON_P = REASON_CODES[:6], [0.30, 0.20, 0.20, 0.15, 0.10, 0.05]
REV_REASONS, REV_REASON_P = REASON_CODES[6:], [0.40, 0.30, 0.20, 0.10]

PAID = ("RCH_VOUCHER", "RCH_DEALER", "RCH_ATM", "RCH_APP")
RECHARGE_CHANNEL = {"RCH_VOUCHER": "VOUCHER", "RCH_DEALER": "POS", "RCH_ATM": "ATM", "RCH_APP": "APP"}
FACE_VALUES = np.array([5, 10, 15, 20, 25, 50])
# Face value preferences: basic plans, business customers, everybody else.
FACE_P = {
    "basic": [0.30, 0.35, 0.12, 0.13, 0.05, 0.05],
    "business": [0.05, 0.15, 0.10, 0.25, 0.20, 0.25],
    "other": [0.15, 0.30, 0.15, 0.20, 0.10, 0.10],
}
# Recharge channel preferences (VOUCHER, POS, ATM, APP), same three groups.
RCH_CHANNEL_P = {
    "basic": [0.40, 0.35, 0.15, 0.10],
    "business": [0.05, 0.15, 0.35, 0.45],
    "other": [0.25, 0.30, 0.25, 0.20],
}
COLUMNS_BY_TYPE = {
    "face_value": PAID,
    "voucher_serial": ("RCH_VOUCHER",),
    "voucher_batch_id": ("RCH_VOUCHER",),
    "dealer_id": ("RCH_DEALER",),
    "agent_user": ("ADJ_CREDIT", "ADJ_DEBIT"),
    "promo_id": ("BONUS_RCH", "PROMO_CREDIT"),
    "reason_code": ("ADJ_CREDIT", "ADJ_DEBIT", "REVERSAL"),
    "ref_movement_id": ("BONUS_RCH", "REVERSAL", "BONUS_CLAWBACK", "ADV_RECOVERY"),
    "counterparty_msisdn": ("P2P_OUT", "P2P_IN"),
    "payment_ref": ("RCH_DEALER", "RCH_ATM", "RCH_APP"),
}
TOPUP_DATASETS = ("topup", "promotions", "dealers", "voucher_batches")


def _hourly(values: list[float]) -> np.ndarray:
    p = np.asarray(values, float)
    return p / p.sum()


# Hourly profiles per channel (24 weights).
HOURLY = {
    "shop": _hourly([0, 0, 0, 0, 0, 0, 0, 0.2, 1, 2, 2.6, 3, 3.2, 3, 3, 3.2, 3.6, 4.2, 4.8, 4.6, 3.6, 2.2, 0.8, 0]),
    "ATM": _hourly(
        [
            0.3,
            0.1,
            0.05,
            0.05,
            0.05,
            0.1,
            0.4,
            1.2,
            2.4,
            2.8,
            3,
            3.2,
            3.6,
            3.2,
            3,
            3,
            3.2,
            3.8,
            4.4,
            4.2,
            3.4,
            2.4,
            1.4,
            0.7,
        ]
    ),
    "APP": _hourly(
        [1, 0.6, 0.3, 0.2, 0.2, 0.3, 0.8, 1.6, 2.2, 2.4, 2.4, 2.6, 2.8, 2.6, 2.4, 2.4, 2.6, 3, 3.6, 4.4, 5, 4.8, 3.6, 2]
    ),
}
WEEKDAY = np.array([1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 0.85])

# Volumes for the default arguments (20,000 customers, 90 days, scale 1).
RECHARGES_PER_CUSTOMER_DAY = 250_000 / (20_000 * 90)
BONUS_SHARE = 0.40
BUNDLES_PER_RECHARGE = 0.6
PROMO_CREDITS_PER_DAY = 110 / 20_000  # per customer
REVERSAL_SHARE = 0.003
ADJUSTMENT_SHARE = 0.005
N_AGENTS = 40
USSD_BUNDLE_SHARE = 0.48


@dataclass(frozen=True)
class Population:
    """The telecom customers the ledger is about."""

    msisdn: np.ndarray  # "+351..." strings
    segment: np.ndarray  # index into SEGMENTS (1 = Business)
    tariff: np.ndarray  # index into TARIFFS (0 = basic)
    region: np.ndarray  # index into regions
    regions: np.ndarray  # region names


@dataclass(frozen=True)
class TopupConfig:
    days: int = 90
    start_date: date = date(2026, 10, 1)
    scale: float = 1.0
    opaque_codes: bool = False
    with_readme: bool = False


class TopupGenerator:
    """Builds the ledger, the reference tables and their ground truth."""

    def __init__(self, population: Population, cfg: TopupConfig, seed: int) -> None:
        if cfg.days < 60:
            raise ValueError("topup days must be >= 60 (T7 needs 60 days and a quiet period after them)")
        self.pop, self.cfg = population, cfg
        self.seed = seed
        self.rng = np.random.default_rng([seed, 2026, 11])  # independent of the telecom stream
        self._pay_used, self._pay_offset = 0, int(self.rng.integers(0, 10**11))
        self.n = population.msisdn.size
        self.days = cfg.days
        self.start = np.datetime64(cfg.start_date.isoformat(), "s")
        self.dates = [cfg.start_date + timedelta(days=d) for d in range(cfg.days)]
        self.volume = self.n / 20_000 * cfg.scale  # volume factor relative to the defaults
        # Columns of every movement, appended part by part (see _add).
        self.parts: list[dict[str, np.ndarray]] = []
        self.next_key = 0
        self.quality: list[dict[str, Any]] = []
        self.traps: list[dict[str, Any]] = []
        self.details: dict[str, Any] = {}
        self.recon: dict[str, list[int]] = {}

        # Calendar of the planted events (positions scale with the window length).
        def day(x: int) -> int:
            return int(min(self.days - 1, round(x * self.days / 90)))

        self.tz_day = day(70)
        self.tr2_days = list(range(day(9), day(9) + 5))
        self.t3_day = day(33)
        self.t4_day = day(56)
        self.tr1_day = day(44)  # day 45 counting from 1
        self.anomaly_free_days = {self.tz_day}

    # ------------------------------------------------------------ helpers --
    def _add(self, n: int, **cols: Any) -> np.ndarray:
        """Append ``n`` movements; return their internal keys.

        Columns not given are null. ``cust``/``counterparty`` are customer
        indices (-1 = none), ``ref`` an internal key (-1 = none, -2 = dangling).
        """
        keys = np.arange(self.next_key, self.next_key + n)
        self.next_key += n

        def col(name: str, default: Any, dtype: Any = object) -> np.ndarray:
            value = cols.get(name, default)
            arr = np.asarray(value, dtype=dtype) if not np.isscalar(value) or value is None else None
            if arr is None or arr.ndim == 0:
                arr = np.full(n, value, dtype=dtype)
            assert arr.shape[0] == n, name
            return arr

        self.parts.append(
            {
                "_key": keys,
                "cust": col("cust", -1, np.int64),
                "ts": col("ts", np.datetime64("NaT", "s"), "datetime64[s]"),
                "mtype": col("mtype", None),
                "channel": col("channel", None),
                "wallet": col("wallet", "MAIN"),
                "amount": col("amount", 0, np.int64),  # cents, signed
                "face": col("face", -1, np.int64),  # cents, -1 = null
                "serial": col("serial", None),
                "batch": col("batch", None),
                "dealer": col("dealer", None),
                "agent": col("agent", None),
                "promo": col("promo", None),
                "reason": col("reason", None),
                "ref": col("ref", -1, np.int64),
                "counterparty": col("counterparty", -1, np.int64),
                "payment_ref": col("payment_ref", None),
                "tag": col("tag", ""),
            }
        )
        return keys

    def _count(self, n: int) -> int:
        """A reconciliation problem size: exact for the defaults, proportional otherwise (min 10)."""
        return n if self.volume == 1 else max(10, round(n * self.volume))

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({k: np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]})

    def _days(self, n: int, low: int = 0, high: int | None = None, avoid: set[int] | None = None) -> np.ndarray:
        """``n`` day indices following the seasonality (month ends/starts, weekdays)."""
        high = self.days if high is None else high
        idx = np.arange(low, high)
        w = np.array([self._day_weight(d) for d in idx])
        if avoid:
            w[np.isin(idx, list(avoid))] = 0
        return self.rng.choice(idx, n, p=w / w.sum())

    def _day_weight(self, d: int) -> float:
        day = self.dates[d]
        last = (day.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        boundary = day.day <= 3 or day.day >= last.day - 1
        return (1.8 if boundary else 1.0) * WEEKDAY[day.weekday()]

    def _ts(self, days: np.ndarray, profile: np.ndarray | None = None, hours: np.ndarray | None = None) -> np.ndarray:
        rng = self.rng
        if hours is None:
            hours = rng.choice(24, days.size, p=profile)
        secs = days.astype(np.int64) * 86_400 + hours.astype(np.int64) * 3_600 + rng.integers(0, 3_600, days.size)
        # Nothing in the last hour of the window: linked movements (bonus, recovery,
        # transfer) follow seconds later and must stay inside it.
        secs = np.minimum(secs, self.days * 86_400 - 3_601)
        return self.start + secs.astype("timedelta64[s]")

    def _group(self, cust: np.ndarray) -> np.ndarray:
        """0 = basic plan, 1 = business, 2 = other (face value and channel preferences)."""
        g = np.full(cust.size, 2)
        g[self.pop.tariff[cust] == 0] = 0
        g[self.pop.segment[cust] == 1] = 1
        return g

    def _faces(self, cust: np.ndarray) -> np.ndarray:
        probs = np.array([FACE_P["basic"], FACE_P["business"], FACE_P["other"]])
        return FACE_VALUES[_choice_rows(self.rng, probs, self._group(cust))] * 100

    def _payment_refs(self, n: int) -> np.ndarray:
        """``n`` new payment references: unique and random-looking for any volume.

        A counter times a multiplier coprime with 10**11 is a bijection modulo 10**11.
        """
        counter = np.arange(self._pay_used, self._pay_used + n, dtype=np.int64)
        self._pay_used += n
        values = (counter * 48_271 * 1_000_003 + self._pay_offset) % 10**11
        return np.char.add("PR", pd.Series(values).astype(str).str.zfill(11).to_numpy().astype(str)).astype(object)

    # -------------------------------------------------------- reference --
    def _roles(self) -> None:
        """Disjoint customer sets for the anomalies (no customer is in two)."""
        rng = self.rng
        order = rng.permutation(self.n)
        sizes = {
            "T1": 15,
            "T2": 35,
            "T4": 20,
            "T5": 40,
            "T6c": 3,
            "T6f": 120,
            "T7": 80,
            "T8": 60,
        }
        self.roles: dict[str, np.ndarray] = {}
        start = 0
        for role, size in sizes.items():
            self.roles[role] = np.sort(order[start : start + size])
            start += size
        self.role_mask = np.zeros(self.n, bool)
        self.role_mask[order[:start]] = True
        rest = order[start:]
        # Normal behaviours: P2P senders (3%) and advance users (5%), disjoint from the anomalies.
        n_p2p, n_adv = round(0.03 * self.n), round(0.05 * self.n)
        self.p2p_senders = np.sort(rest[:n_p2p])
        self.adv_users = np.sort(rest[n_p2p : n_p2p + n_adv])
        self.normal = np.sort(rest)  # customers that may appear in any normal movement

    def _dealers(self) -> pd.DataFrame:
        rng = self.rng
        n = 600
        kind = np.array(["INDEPENDENT"] * 530 + ["CHAIN"] * 30 + ["ONLINE"] * 40, dtype=object)
        kind = kind[rng.permutation(n)]
        closing = np.where(kind == "CHAIN", 23, np.where(kind == "ONLINE", 24, rng.integers(19, 23, n)))
        commission = np.where(
            kind == "CHAIN", rng.uniform(2.5, 3.5, n), np.where(kind == "ONLINE", 3.0, rng.uniform(4, 6, n))
        )
        activation = [
            (date(2015, 1, 1) + timedelta(days=int(d))).isoformat()
            for d in rng.integers(0, (date(2026, 8, 1) - date(2015, 1, 1)).days, n)
        ]
        self.dealer_ids = np.array([f"D{i:04d}" for i in range(1, n + 1)], dtype=object)
        self.dealer_kind, self.dealer_closing = kind, closing
        # Long-tailed volume: the 30 chains take 45% of RCH_DEALER, the rest a lognormal share.
        w = rng.lognormal(0, 1.9, n)
        w[kind == "CHAIN"] = 0
        w = w / w.sum() * 0.55
        w[kind == "CHAIN"] = 0.45 / 30
        self.dealer_weight = w
        return pd.DataFrame(
            {
                "dealer_id": self.dealer_ids,
                "dealer_type": kind,
                "region": self.pop.regions[rng.integers(0, self.pop.regions.size, n)],
                "commission_pct": commission.round(2),
                "activation_date": pd.to_datetime(activation).date,
                "closing_hour": closing.astype(np.int32),
            }
        )

    def _promotions(self) -> pd.DataFrame:
        rng, days = self.rng, self.days
        credit_ids = {3, 6, 9, 12, 15, 19, 22, 24}
        rows = []
        for i in range(1, 25):
            pid = f"P{i:03d}"
            if i in credit_ids:
                length = int(rng.integers(10, 30))
                start = int(rng.integers(0, days - length))
                if i == 9:  # TR1: the campaign covers the trap day
                    start, length = self.tr1_day - 14, 30
                rows.append(
                    {
                        "promo_id": pid,
                        "name": f"Free credit {i}",
                        "bonus_pct": None,
                        "min_recharge": None,
                        "max_bonus": float(rng.choice([1, 2, 3, 5])),
                        "start": start,
                        "end": start + length,
                        "max_per_customer": int(1 if i == 9 else rng.choice([1, 2])),
                        "kind": "credit",
                    }
                )
                continue
            if i == 1:  # evergreen base bonus
                start, end, pct, low, cap, per = -30, days + 30, 10, 10, 5.0, 30
            elif i == 17:  # T3: defined at 10%
                start, end, pct, low, cap, per = self.t3_day - 13, self.t3_day + 17, 10, 5, 5.0, 10
            else:
                length = int(rng.integers(14, 40))
                start = int(rng.integers(-10, days - 15))
                end = start + length
                pct = int(rng.choice([10, 15, 20, 25, 50]))
                low = int(rng.choice([5, 10, 15, 20]))
                cap, per = float(rng.choice([2, 3, 5, 10])), int(rng.choice([3, 5, 10, 30]))
            rows.append(
                {
                    "promo_id": pid,
                    "name": f"Recharge bonus {i}",
                    "bonus_pct": float(pct),
                    "min_recharge": float(low),
                    "max_bonus": cap,
                    "start": start,
                    "end": end,
                    "max_per_customer": per,
                    "kind": "recharge",
                }
            )
        promos = pd.DataFrame(rows)
        self.promos = promos
        day0 = self.cfg.start_date
        return pd.DataFrame(
            {
                "promo_id": promos["promo_id"],
                "name": promos["name"],
                "bonus_pct": promos["bonus_pct"],
                "min_recharge": promos["min_recharge"],
                "max_bonus": promos["max_bonus"],
                "start_date": [day0 + timedelta(days=int(d)) for d in promos["start"]],
                "end_date": [day0 + timedelta(days=int(d)) for d in promos["end"]],
                "max_per_customer": promos["max_per_customer"].astype(np.int32),
            }
        )

    # ---------------------------------------------------------- recharges --
    def _normal_recharges(self) -> None:
        rng = self.rng
        cust_pool = self.normal
        activity = rng.lognormal(0, 0.6, cust_pool.size)
        expected = activity / activity.mean() * RECHARGES_PER_CUSTOMER_DAY * self.cfg.scale * self.days
        counts = rng.poisson(expected)
        cust = np.repeat(cust_pool, counts)
        # Role customers also recharge normally (their anomalies come on top).
        role_cust = np.where(self.role_mask)[0]
        role_counts = rng.poisson(RECHARGES_PER_CUSTOMER_DAY * self.cfg.scale * self.days, role_cust.size)
        cust = np.concatenate([cust, np.repeat(role_cust, role_counts)])
        n = cust.size
        # Channel: segment preference mixed with a personal favourite.
        group = self._group(np.arange(self.n))
        probs = np.array([RCH_CHANNEL_P["basic"], RCH_CHANNEL_P["business"], RCH_CHANNEL_P["other"]])
        favourite = _choice_rows(rng, probs, group)
        personal = 0.6 * probs[group] + 0.4 * np.eye(4)[favourite]
        channel_idx = _choice_rows(rng, personal, cust)
        mtype = np.array(PAID, dtype=object)[channel_idx]
        days = self._days(n)
        ts = np.empty(n, "datetime64[s]")
        for ch, profile in ((0, "shop"), (1, "shop"), (2, "ATM"), (3, "APP")):
            m = channel_idx == ch
            ts[m] = self._ts(days[m], HOURLY[profile])
        face = self._faces(cust)
        dealer = np.full(n, None, dtype=object)
        # Dealers: a favourite dealer per customer (80%), otherwise any, by volume.
        fav_dealer = rng.choice(self.dealer_ids.size, self.n, p=self.dealer_weight)
        pos = np.where(channel_idx == 1)[0]
        pick = np.where(
            rng.random(pos.size) < 0.8,
            fav_dealer[cust[pos]],
            rng.choice(self.dealer_ids.size, pos.size, p=self.dealer_weight),
        )
        # Shops are open until their closing hour.
        closing = self.dealer_closing[pick]
        hours = (ts[pos].astype("datetime64[h]") - ts[pos].astype("datetime64[D]")).astype(int)
        late = hours >= closing
        hours[late] = rng.integers(9, closing[late])
        ts[pos] = self._ts(days[pos], hours=hours)
        dealer[pos] = self.dealer_ids[pick]
        payment = np.full(n, None, dtype=object)
        electronic = channel_idx > 0
        payment[electronic] = self._payment_refs(int(electronic.sum()))
        keys = self._add(
            n,
            cust=cust,
            ts=ts,
            mtype=mtype,
            channel=np.array([RECHARGE_CHANNEL[t] for t in PAID], dtype=object)[channel_idx],
            amount=face,
            face=face,
            dealer=dealer,
            payment_ref=payment,
        )
        self.rch = pd.DataFrame({"key": keys, "cust": cust, "ts": ts, "face": face, "type": mtype})
        self._vouchers()

    def _vouchers(self) -> None:
        """Assign each voucher recharge to a batch distributed before it; build voucher_batches."""
        rng = self.rng
        n_batches = 400
        face = rng.choice(FACE_VALUES, n_batches, p=[0.15, 0.30, 0.15, 0.2, 0.1, 0.1]) * 100
        dist = rng.integers(-40, self.days - 10, n_batches)
        # T4: two batches distributed 6 days after their fraudulent redemption.
        t4 = rng.choice(np.where(face == 1000)[0], 2, replace=False)
        dist[t4] = self.t4_day + 6
        self.t4_batches = t4
        part = self.parts[-1]
        vmask = part["mtype"] == "RCH_VOUCHER"
        rows = np.where(vmask)[0]
        rday = (part["ts"][rows] - self.start).astype("timedelta64[D]").astype(int)
        rface = part["face"][rows]
        batch = np.empty(rows.size, int)
        ok = np.ones(n_batches, bool)
        ok[t4] = False
        for f in np.unique(rface):
            m = np.where(rface == f)[0]
            cand = np.where((face == f) & ok)[0]
            age = rday[m][:, None] - dist[cand][None, :]
            w = np.where(age >= 1, np.exp(-age / 14.0), 0.0)
            w[w.sum(axis=1) == 0, np.argmax(dist[cand] < 0)] = 1.0  # always some batch
            w = w / w.sum(axis=1, keepdims=True)
            batch[m] = cand[_choice_rows(rng, w, np.arange(m.size))]
        # Serial numbers in redemption order within each batch.
        order = np.lexsort((part["ts"][rows], batch))
        seq = np.empty(rows.size, int)
        sorted_batch = batch[order]
        seq[order] = np.arange(rows.size) - np.searchsorted(sorted_batch, sorted_batch)
        used = np.bincount(batch, minlength=n_batches)
        used[t4] = 300
        n_vouchers = np.maximum(100, np.ceil(used / rng.uniform(0.55, 0.9, n_batches) / 100) * 100).astype(int)
        n_vouchers[t4] = 300
        self.batch_ids = np.array([f"B{i:04d}" for i in range(1, n_batches + 1)], dtype=object)
        part["batch"][rows] = self.batch_ids[batch]
        part["serial"][rows] = [f"{b + 1:04d}{s:06d}" for b, s in zip(batch, seq, strict=True)]
        self.batch_face, self.batch_used = face, used
        # The distributing dealer: chains distribute more batches.
        p = np.where(self.dealer_kind == "CHAIN", 8.0, 1.0)
        self.batches = pd.DataFrame(
            {
                "batch_id": self.batch_ids,
                "face_value": face / 100,
                "n_vouchers": n_vouchers.astype(np.int32),
                "distribution_date": [self.cfg.start_date + timedelta(days=int(d)) for d in dist],
                "dealer_id": self.dealer_ids[rng.choice(self.dealer_ids.size, n_batches, p=p / p.sum())],
            }
        )

    def _bonus_amount(self, face: np.ndarray, promo_rows: np.ndarray) -> np.ndarray:
        pct = self.promos["bonus_pct"].to_numpy()[promo_rows]
        cap = (self.promos["max_bonus"].to_numpy()[promo_rows] * 100).astype(np.int64)
        return np.minimum(np.round(face * pct / 100).astype(np.int64), cap)

    def _normal_bonuses(self) -> None:
        """About 40% of recharges earn a bonus a few seconds later, exactly by the promotion's rule."""
        rng = self.rng
        rch = self.rch
        promos = self.promos
        rec = np.where(promos["kind"] == "recharge")[0]
        day = (rch["ts"].to_numpy() - self.start).astype("timedelta64[D]").astype(int)
        start, end = promos["start"].to_numpy()[rec], promos["end"].to_numpy()[rec]
        low = (promos["min_recharge"].to_numpy()[rec] * 100).astype(np.int64)
        eligible = (day[:, None] >= start) & (day[:, None] <= end) & (rch["face"].to_numpy()[:, None] >= low)
        # T3: no regular P017 bonus inside the misconfiguration window (all of them are the anomaly).
        p17 = int(np.where(promos["promo_id"].to_numpy()[rec] == "P017")[0][0])
        ts = rch["ts"].to_numpy()
        self.t3_start = self.start + np.timedelta64(self.t3_day * 86_400 + 12 * 3_600, "s")
        self.t3_end = self.t3_start + np.timedelta64(36 * 3_600, "s")
        in_t3 = (ts >= self.t3_start) & (ts < self.t3_end)
        eligible[in_t3, p17] = False
        # T2/T8 customers only get the bonuses of their anomaly.
        excluded = np.isin(rch["cust"].to_numpy(), np.concatenate([self.roles["T2"], self.roles["T8"]]))
        eligible[excluded] = False
        any_eligible = eligible.any(axis=1)
        p = min(1.0, BONUS_SHARE * 1.04 / max(any_eligible.mean(), 1e-9))
        gets = any_eligible & (rng.random(len(rch)) < p)
        score = np.where(eligible, rng.random(eligible.shape), -1)
        promo_row = rec[score.argmax(axis=1)]
        idx = np.where(gets)[0]
        sel = pd.DataFrame({"i": idx, "cust": rch["cust"].to_numpy()[idx], "promo": promo_row[idx], "ts": ts[idx]})
        # max_per_customer: keep the first bonuses of each customer and promotion.
        sel = sel.sort_values("ts", kind="stable")
        rank = sel.groupby(["cust", "promo"]).cumcount().to_numpy()
        sel = sel[rank < promos["max_per_customer"].to_numpy()[sel["promo"].to_numpy()]].sort_values("i")
        idx, prow = sel["i"].to_numpy(), sel["promo"].to_numpy()
        amount = self._bonus_amount(rch["face"].to_numpy()[idx], prow)
        keep = amount > 0
        idx, prow, amount = idx[keep], prow[keep], amount[keep]
        bkeys = self._add(
            idx.size,
            cust=rch["cust"].to_numpy()[idx],
            ts=ts[idx] + rng.integers(2, 31, idx.size).astype("timedelta64[s]"),
            mtype="BONUS_RCH",
            channel="SYSTEM",
            wallet="BONUS",
            amount=amount,
            promo=promos["promo_id"].to_numpy()[prow],
            ref=rch["key"].to_numpy()[idx],
        )
        self.bonus_of = dict(zip(rch["key"].to_numpy()[idx], zip(bkeys, amount, strict=True), strict=True))
        self.details["bonus_share"] = round(idx.size / len(rch), 4)

    def _t3(self) -> None:
        """T3: P017 pays 100% instead of its 10% for 36 hours (a configuration error, not fraud)."""
        rng = self.rng
        rch, promos = self.rch, self.promos
        ts = rch["ts"].to_numpy()
        p17 = int(np.where(promos["promo_id"] == "P017")[0][0])
        cand = np.where(
            (ts >= self.t3_start)
            & (ts < self.t3_end)
            & ~self.role_mask[rch["cust"].to_numpy()]
            & (rch["face"].to_numpy() >= promos["min_recharge"][p17] * 100)
        )[0]
        size = min(cand.size, round(1500 * self.volume))
        chosen = np.sort(rng.choice(cand, size, replace=False))
        keys = rch["key"].to_numpy()[chosen]
        # A recharge earns one bonus: the regular bonus of a chosen recharge is replaced.
        self._drop_keys = {self.bonus_of.pop(k)[0] for k in keys if k in self.bonus_of}
        bkeys = self._add(
            size,
            cust=rch["cust"].to_numpy()[chosen],
            ts=ts[chosen] + rng.integers(2, 31, size).astype("timedelta64[s]"),
            mtype="BONUS_RCH",
            channel="SYSTEM",
            wallet="BONUS",
            amount=rch["face"].to_numpy()[chosen],  # 100% of the recharge
            promo="P017",
            ref=keys,
            tag="T3",
        )
        self.t3_customers = np.unique(rch["cust"].to_numpy()[chosen])
        self.details["T3"] = {
            "description": "Promotion P017 pays a 100% bonus instead of its 10% for 36 hours",
            "promo_id": "P017",
            "start": str(self.t3_start),
            "end": str(self.t3_end),
            "movements": int(size),
            "customers": int(self.t3_customers.size),
        }
        self.t3_keys = bkeys

    def _expiry_and_reversals(self) -> None:
        rng = self.rng
        rch = self.rch
        normal = ~self.role_mask[rch["cust"].to_numpy()] & ~np.isin(rch["cust"].to_numpy(), self.t3_customers)
        # Reversals: 0.3% of recharges, within 24 hours; a reversed bonus is always clawed back.
        pick = rng.choice(np.where(normal)[0], round(REVERSAL_SHARE * len(rch)), replace=False)
        pick = pick[(rch["ts"].to_numpy()[pick] < self.start + np.timedelta64((self.days - 1) * 86_400, "s"))]
        rev_ts = rch["ts"].to_numpy()[pick] + rng.integers(600, 86_400, pick.size).astype("timedelta64[s]")
        rkeys = rch["key"].to_numpy()[pick]
        self.reversed = set(rkeys.tolist())
        self._add(
            pick.size,
            cust=rch["cust"].to_numpy()[pick],
            ts=rev_ts,
            mtype="REVERSAL",
            channel="BACKOFFICE",
            amount=-rch["face"].to_numpy()[pick],
            reason=rng.choice(REV_REASONS, pick.size, p=REV_REASON_P),
            ref=rkeys,
        )
        claw = [(i, *self.bonus_of[k]) for i, k in enumerate(rkeys) if k in self.bonus_of]
        if claw:
            i, bkey, amount = (np.array(v) for v in zip(*claw, strict=True))
            self._add(
                i.size,
                cust=rch["cust"].to_numpy()[pick][i],
                ts=rev_ts[i] + rng.integers(1, 6, i.size).astype("timedelta64[s]"),
                mtype="BONUS_CLAWBACK",
                channel="SYSTEM",
                wallet="BONUS",
                amount=-amount,
                ref=bkey,
            )
        clawed = {self.bonus_of[k][0] for k in rkeys if k in self.bonus_of}
        # Expiry: a quarter of the (not clawed back) bonuses partly expire 30 days later.
        bonus = [(k, b, a) for k, (b, a) in self.bonus_of.items() if b not in clawed]
        keys = np.array([k for k, _, _ in bonus])
        amounts = np.array([a for _, _, a in bonus])
        lookup = rch.set_index("key")
        exp = rng.random(keys.size) < 0.25
        exp_ts = (
            lookup.loc[keys[exp], "ts"].to_numpy()
            + np.timedelta64(30 * 86_400, "s")
            + rng.integers(0, 3_600, int(exp.sum())).astype("timedelta64[s]")
        )
        inside = exp_ts < self.start + np.timedelta64(self.days * 86_400 - 3_601, "s")
        cust = lookup.loc[keys[exp], "cust"].to_numpy()[inside]
        amount = np.maximum(1, np.round(amounts[exp][inside] * rng.uniform(0.2, 1, inside.sum()))).astype(np.int64)
        self._add(
            int(inside.sum()),
            cust=cust,
            ts=exp_ts[inside],
            mtype="EXPIRY",
            channel="SYSTEM",
            wallet="BONUS",
            amount=-amount,
        )

    # ------------------------------------------------------ other normal --
    def _bundles(self) -> None:
        rng = self.rng
        rch = self.rch
        n = round(BUNDLES_PER_RECHARGE * len(rch))
        src = rng.integers(0, len(rch), n)  # bundle buyers follow recharge activity
        cust = rch["cust"].to_numpy()[src]
        ts = rch["ts"].to_numpy()[src] + rng.integers(600, 10 * 86_400, n).astype("timedelta64[s]")
        inside = ts < self.start + np.timedelta64(self.days * 86_400, "s")
        cust, ts, n = cust[inside], ts[inside], int(inside.sum())
        ussd = rng.random(n) < USSD_BUNDLE_SHARE
        self._add(
            n,
            cust=cust,
            ts=ts,
            mtype="BUNDLE_PURCHASE",
            channel=np.where(ussd, "USSD", "APP").astype(object),
            wallet=np.where(rng.random(n) < 0.8, "MAIN", "BONUS").astype(object),
            amount=-(rng.choice([2, 3, 5, 7.5, 10, 15], n, p=[0.2, 0.25, 0.25, 0.1, 0.12, 0.08]) * 100).astype(
                np.int64
            ),
        )

    def _promo_credits(self) -> None:
        rng = self.rng
        promos = self.promos
        credit = promos[promos["kind"] == "credit"]
        per_day = PROMO_CREDITS_PER_DAY * self.n * self.cfg.scale
        total_days = sum(
            min(self.days - 1, e) - max(0, s) + 1 for s, e in zip(credit["start"], credit["end"], strict=True)
        )
        self.p009_receivers: set[int] = set()
        for _, p in credit.iterrows():
            s, e = max(0, p["start"]), min(self.days - 1, p["end"])
            n = round(per_day * self.days * (e - s + 1) / total_days)
            # Distinct customers per promotion, within max_per_customer.
            cust = rng.choice(self.normal, n, replace=p["max_per_customer"] > 1)
            if p["max_per_customer"] > 1:
                cust = cust[pd.Series(cust).groupby(cust).cumcount().to_numpy() < p["max_per_customer"]]
            days = rng.integers(s, e + 1, cust.size)
            self._add(
                cust.size,
                cust=cust,
                ts=self._ts(days, HOURLY["APP"]),
                mtype="PROMO_CREDIT",
                channel="SYSTEM",
                wallet="BONUS",
                amount=int(p["max_bonus"] * 100),
                promo=p["promo_id"],
            )
            if p["promo_id"] == "P009":
                self.p009_receivers = set(cust.tolist())
        # TR1: a legitimate campaign day, 3x the normal PROMO_CREDIT volume, within P009's rules.
        p9 = credit[credit["promo_id"] == "P009"].iloc[0]
        extra = round(2 * per_day)
        pool = np.setdiff1d(self.normal, np.array(sorted(self.p009_receivers) + self.t3_customers.tolist(), int))
        cust = rng.choice(pool, extra, replace=False)
        keys = self._add(
            extra,
            cust=cust,
            ts=self._ts(np.full(extra, self.tr1_day), hours=rng.integers(10, 21, extra)),
            mtype="PROMO_CREDIT",
            channel="SYSTEM",
            wallet="BONUS",
            amount=int(p9["max_bonus"] * 100),
            promo="P009",
            tag="TR1",
        )
        self.tr1 = {"keys": keys, "date": self.dates[self.tr1_day].isoformat(), "extra": extra, "per_day": per_day}

    def _adjustments(self, n_other: int) -> None:
        rng = self.rng
        self.agents = np.array([f"OP{i:03d}" for i in range(1, N_AGENTS + 1)], dtype=object)
        workload = rng.lognormal(0, 0.5, N_AGENTS)
        n = round(ADJUSTMENT_SHARE * n_other / (1 - ADJUSTMENT_SHARE))
        weekdays = [d for d in range(self.days) if self.dates[d].weekday() < 5]
        days = rng.choice(weekdays, n)
        credit = rng.random(n) < 0.65
        amount = np.round(rng.lognormal(np.log(500), 0.8, n)).astype(np.int64).clip(50, 5_000)
        reason = rng.choice(ADJ_REASONS, n, p=ADJ_REASON_P).astype(object)
        missing = rng.choice(n, round(0.05 * n), replace=False)
        reason[missing] = None
        self._add(
            n,
            cust=rng.choice(self.normal, n),
            ts=self._ts(days, hours=rng.integers(9, 18, n)),
            mtype=np.where(credit, "ADJ_CREDIT", "ADJ_DEBIT").astype(object),
            channel="BACKOFFICE",
            wallet=np.where(rng.random(n) < 0.85, "MAIN", "BONUS").astype(object),
            amount=np.where(credit, amount, -amount),
            agent=self.agents[rng.choice(N_AGENTS, n, p=workload / workload.sum())],
            reason=reason,
        )
        self.quality.append(
            {
                "id": "topup_null_reason_code",
                "dataset": "topup",
                "column": "reason_code",
                "kind": "nulls",
                "rows": int(missing.size),
                "note": "reason_code must be filled in adjustments; null in 5% of them",
            }
        )
        # T1: 3 operators, 240 night credits on the same 15 customers, always reason OTHER.
        agents = rng.choice(self.agents, 3, replace=False)
        m = 240
        days = self._days(m, avoid=self.anomaly_free_days)
        keys = self._add(
            m,
            cust=rng.choice(self.roles["T1"], m),
            ts=self._ts(days, hours=rng.integers(0, 5, m)),
            mtype="ADJ_CREDIT",
            channel="BACKOFFICE",
            amount=(rng.choice([10, 15, 20, 25, 30], m) * 100).astype(np.int64),
            agent=np.repeat(agents, m // 3),
            reason="OTHER",
            tag="T1",
        )
        self.t1 = {"agents": sorted(agents.tolist()), "keys": keys}
        self.details["T1"] = {
            "description": "3 operators make 240 ADJ_CREDIT on the same 15 customers between 00h and 05h, reason OTHER",
            "agent_user": self.t1["agents"],
            "customers": self.pop.msisdn[self.roles["T1"]].tolist(),
        }

    def _p2p(self) -> None:
        rng = self.rng
        senders = self.p2p_senders
        k = rng.integers(1, 3, senders.size)  # 1-2 stable receivers (family)
        pair_sender = np.repeat(senders, k)
        pair_receiver = rng.choice(np.setdiff1d(self.normal, senders), pair_sender.size)
        transfers = np.maximum(1, rng.poisson(2 * self.days / 30 / 1.5, pair_sender.size))
        s = np.repeat(pair_sender, transfers)
        r = np.repeat(pair_receiver, transfers)
        n = s.size
        ts = self._ts(self._days(n), HOURLY["APP"])
        channel = np.where(rng.random(n) < 0.6, "USSD", "APP").astype(object)
        amount = (rng.choice([1, 2, 3, 5], n, p=[0.25, 0.3, 0.25, 0.2]) * 100).astype(np.int64)
        out = self._add(n, cust=s, ts=ts, mtype="P2P_OUT", channel=channel, amount=-amount, counterparty=r)
        # R5: 60 P2P_OUT without the matching P2P_IN (outside the sign-error days, see TR2).
        day = (ts - self.start).astype("timedelta64[D]").astype(int)
        cand = np.where(~np.isin(day, self.tr2_days))[0]
        lost = np.zeros(n, bool)
        lost[rng.choice(cand, self._count(60), replace=False)] = True
        self.recon["R5"] = out[lost].tolist()
        self._add(
            int((~lost).sum()),
            cust=r[~lost],
            ts=ts[~lost] + np.timedelta64(1, "s"),
            mtype="P2P_IN",
            channel=channel[~lost],
            amount=amount[~lost],
            counterparty=s[~lost],
        )

    def _advances(self) -> None:
        """5% of customers take advances; 95% are recovered at their next recharge.

        Each advance is placed between two recharges of the customer (or before the
        first one), so the "next recharge" always exists and is the one that recovers it.
        """
        rng = self.rng
        rch = self.rch[self.rch["cust"].isin(self.adv_users)].sort_values(["cust", "ts"]).reset_index(drop=True)
        cust, rts = rch["cust"].to_numpy(), rch["ts"].to_numpy()
        first = np.r_[True, cust[1:] != cust[:-1]]
        prev = np.where(first, rts - np.timedelta64(6 * 86_400, "s"), np.r_[rts[:1], rts[:-1]])
        prev = np.maximum(prev, self.start)
        # 1-3 advances per user, each before a different recharge.
        rank = rng.random(len(rch))
        order = np.lexsort((rank, cust))
        nth = np.empty(len(rch), int)
        nth[order] = np.arange(len(rch)) - np.searchsorted(cust[order], cust[order])
        quota = dict(zip(self.adv_users, rng.integers(1, 4, self.adv_users.size), strict=True))
        pick = np.where(nth < np.vectorize(quota.get)(cust))[0]
        gap = (rts[pick] - prev[pick]).astype(np.int64)
        pick, gap = pick[gap > 120], gap[gap > 120]
        ts = prev[pick] + (rng.random(pick.size) * (gap - 60)).astype("timedelta64[s]")
        users = cust[pick]
        # Users without a recharge to place it before: one advance that is never recovered.
        lonely = np.setdiff1d(self.adv_users, users)
        users = np.concatenate([users, lonely])
        ts = np.concatenate([ts, self._ts(self._days(lonely.size), HOURLY["APP"])])
        n = users.size
        amount = (rng.choice([2, 3, 5], n) * 100).astype(np.int64)
        grants = self._add(n, cust=users, ts=ts, mtype="ADV_GRANT", channel="USSD", amount=amount)
        recovered = (rng.random(n) < 0.95) & (np.arange(n) < pick.size)
        self._add(
            int(recovered.sum()),
            cust=users[recovered],
            ts=rts[pick][recovered[: pick.size]] + rng.integers(1, 11, recovered.sum()).astype("timedelta64[s]"),
            mtype="ADV_RECOVERY",
            channel="SYSTEM",
            amount=-amount[recovered],
            ref=grants[recovered],
        )
        self.details["advance_recovery_share"] = round(float(recovered.mean()), 4)

    # ------------------------------------------------------------ anomalies --
    def _t2(self) -> None:
        """T2: recharge, bonus, reversal of the recharge, and the bonus is never clawed back."""
        rng = self.rng
        promos = self.promos
        cust = np.repeat(self.roles["T2"], rng.integers(4, 9, self.roles["T2"].size))
        n = cust.size
        days = self._days(n, 2, self.days - 2, avoid=self.anomaly_free_days)
        ts = self._ts(days, HOURLY["APP"])
        face = (rng.choice([20, 25, 50], n) * 100).astype(np.int64)
        # The evergreen P001 (10%, min 10, cap 5, 30 per customer) keeps the bonus within its rules.
        p1 = int(np.where(promos["promo_id"] == "P001")[0][0])
        bonus = self._bonus_amount(face, np.full(n, p1))
        r = self._add(
            n,
            cust=cust,
            ts=ts,
            mtype="RCH_APP",
            channel="APP",
            amount=face,
            face=face,
            payment_ref=self._payment_refs(n),
            tag="T2",
        )
        b = self._add(
            n,
            cust=cust,
            ts=ts + rng.integers(2, 31, n).astype("timedelta64[s]"),
            mtype="BONUS_RCH",
            channel="SYSTEM",
            wallet="BONUS",
            amount=bonus,
            promo="P001",
            ref=r,
            tag="T2",
        )
        v = self._add(
            n,
            cust=cust,
            ts=ts + rng.integers(3_600, 20 * 3_600, n).astype("timedelta64[s]"),
            mtype="REVERSAL",
            channel="BACKOFFICE",
            amount=-face,
            reason=rng.choice(["PAYMENT_FAILED", "CUSTOMER_REQUEST"], n),
            ref=r,
            tag="T2",
        )
        self.t2_keys = np.concatenate([r, b, v])
        self.details["T2"] = {
            "description": "4-8 cycles of recharge, bonus and reversal without BONUS_CLAWBACK",
            "cycles": int(n),
        }

    def _t4(self) -> None:
        """T4: two whole batches redeemed in 4 hours by 20 customers, 6 days before distribution."""
        rng = self.rng
        n = 600
        batch = np.repeat(self.t4_batches, 300)
        ts = self._ts(np.full(n, self.t4_day), hours=rng.integers(14, 18, n))
        order = np.lexsort((ts, batch))
        batch, ts = batch[order], ts[order]
        seq = np.tile(np.arange(300), 2)
        face = self.batch_face[batch]
        keys = self._add(
            n,
            cust=rng.choice(self.roles["T4"], n),
            ts=ts,
            mtype="RCH_VOUCHER",
            channel="VOUCHER",
            amount=face,
            face=face,
            serial=np.array([f"{b + 1:04d}{s:06d}" for b, s in zip(batch, seq, strict=True)], dtype=object),
            batch=self.batch_ids[batch],
            tag="T4",
        )
        self.t4_keys = keys
        self.details["T4"] = {
            "description": "2 batches of 300 vouchers redeemed in 4 hours by 20 customers, 6 days before distribution",
            "batches": self.batch_ids[self.t4_batches].tolist(),
            "date": self.dates[self.t4_day].isoformat(),
            "customers": self.pop.msisdn[self.roles["T4"]].tolist(),
        }

    def _t5(self) -> None:
        """T5: 4 independent dealers, 1,200 last-hour recharges, 70% of the value to 10 customers each."""
        rng = self.rng
        indep = np.where(self.dealer_kind == "INDEPENDENT")[0]
        mid = indep[np.argsort(self.dealer_weight[indep])][indep.size // 3 : 2 * indep.size // 3]
        dealers = rng.choice(mid, 4, replace=False)
        keys = []
        others_pool = np.setdiff1d(self.normal, self.t3_customers)
        for i, d in enumerate(dealers):
            targets = self.roles["T5"][i * 10 : (i + 1) * 10]
            # 96 recharges of 50 for the targets, 204 of 10 for others: 70% of the value to the targets.
            cust = np.concatenate([rng.choice(targets, 96), rng.choice(others_pool, 204)])
            face = np.concatenate([np.full(96, 5_000), np.full(204, 1_000)]).astype(np.int64)
            days = self._days(300, avoid=self.anomaly_free_days)
            ts = self._ts(days, hours=np.full(300, self.dealer_closing[d] - 1))
            keys.append(
                self._add(
                    300,
                    cust=cust,
                    ts=ts,
                    mtype="RCH_DEALER",
                    channel="POS",
                    amount=face,
                    face=face,
                    dealer=self.dealer_ids[d],
                    payment_ref=self._payment_refs(300),
                    tag="T5",
                )
            )
        self.t5_dealers, self.t5_keys = self.dealer_ids[dealers], np.concatenate(keys)
        self.details["T5"] = {
            "description": "4 INDEPENDENT dealers: 1,200 recharges in the last hour before closing, "
            "70% of the value to 10 customers each, repeated values",
            "dealer_id": sorted(self.t5_dealers.tolist()),
            "customers": self.pop.msisdn[self.roles["T5"]].tolist(),
        }

    def _t6(self) -> None:
        """T6: 120 feeders recharge by card, transfer to 3 collectors within the hour; recharge reversed later."""
        rng = self.rng
        feeders, collectors = self.roles["T6f"], self.roles["T6c"]
        n = feeders.size
        to = np.repeat(collectors, n // collectors.size)
        base_day = rng.integers(15, self.days - 25, collectors.size)
        days = np.repeat(base_day, n // collectors.size) + rng.integers(0, 10, n)
        days = np.where(days == self.tz_day, days + 1, days)
        ts = self._ts(days, HOURLY["APP"])
        face = (rng.choice([20, 25, 50], n) * 100).astype(np.int64)
        r = self._add(
            n,
            cust=feeders,
            ts=ts,
            mtype="RCH_APP",
            channel="APP",
            amount=face,
            face=face,
            payment_ref=self._payment_refs(n),
            tag="T6",
        )
        t_out = ts + rng.integers(300, 3_300, n).astype("timedelta64[s]")
        o = self._add(
            n, cust=feeders, ts=t_out, mtype="P2P_OUT", channel="APP", amount=-face, counterparty=to, tag="T6"
        )
        i = self._add(
            n,
            cust=to,
            ts=t_out + np.timedelta64(1, "s"),
            mtype="P2P_IN",
            channel="APP",
            amount=face,
            counterparty=feeders,
            tag="T6",
        )
        v = self._add(
            n,
            cust=feeders,
            ts=ts + rng.integers(3 * 86_400, 10 * 86_400, n).astype("timedelta64[s]"),
            mtype="REVERSAL",
            channel="BACKOFFICE",
            amount=-face,
            reason="CARD_CHARGEBACK",
            ref=r,
            tag="T6",
        )
        self.t6_keys = np.concatenate([r, o, i, v])
        self.details["T6"] = {
            "description": "3 collectors receive P2P_IN from 40 customers each; each feeder recharged by card "
            "less than 1 hour before, reversed 3-10 days later",
            "collectors": self.pop.msisdn[collectors].tolist(),
            "feeders": int(n),
        }

    def _t7(self, frame: pd.DataFrame) -> pd.DataFrame:
        """T7: advance, a bundle within the hour, then nothing else (never recovered)."""
        rng = self.rng
        cust = self.roles["T7"]
        n = cust.size
        days = self._days(n, 5, 60, avoid=self.anomaly_free_days)
        ts = self._ts(days, HOURLY["APP"])
        # Drop every later movement of these customers (their story ends with the bundle).
        grant_at = dict(zip(cust, ts, strict=True))
        limit = frame["cust"].map(grant_at)
        frame = frame[~(limit.notna() & (frame["ts"] >= limit))]
        amount = (rng.choice([3, 5], n) * 100).astype(np.int64)
        g = self._add(n, cust=cust, ts=ts, mtype="ADV_GRANT", channel="USSD", amount=amount, tag="T7")
        b = self._add(
            n,
            cust=cust,
            ts=ts + rng.integers(60, 3_540, n).astype("timedelta64[s]"),
            mtype="BUNDLE_PURCHASE",
            channel="APP",
            amount=-amount,
            tag="T7",
        )
        self.t7_keys = np.concatenate([g, b])
        self.details["T7"] = {
            "description": "ADV_GRANT in the first 60 days, BUNDLE_PURCHASE within the hour, "
            "no other movement after; never recovered"
        }
        return frame

    def _t8(self) -> None:
        """T8: 140 bonuses with the right percentage but not eligible (below min_recharge or out of dates)."""
        rng = self.rng
        promos = self.promos
        # Promotions with a minimum to break and dates inside the window (P001 is always active).
        cand = np.where(
            (promos["kind"] == "recharge")
            & (promos["min_recharge"] >= 10)
            & ~promos["promo_id"].isin(["P001", "P017"])
            & ((promos["end"] + 6 < self.days) | (promos["start"] - 6 >= 0))
        )[0]
        cust = np.concatenate([np.repeat(self.roles["T8"][:20], 3), np.repeat(self.roles["T8"][20:], 2)])
        n = cust.size
        prow = rng.choice(cand, n)
        start, end = promos["start"].to_numpy()[prow], promos["end"].to_numpy()[prow]
        low = (promos["min_recharge"].to_numpy()[prow] * 100).astype(np.int64)
        below = rng.random(n) < 0.5
        days = np.empty(n, int)
        face = np.empty(n, np.int64)
        # Below the minimum: inside the dates, a 5 recharge.
        days[below] = rng.integers(np.maximum(start[below], 0), np.minimum(end[below], self.days - 1) + 1)
        face[below] = 500
        # Out of dates: 2-6 days after the end (or before the start), a valid amount. Two days
        # of margin keep it out of dates even on the day written in another time zone.
        out = ~below
        after = end[out] + rng.integers(2, 7, out.sum())
        before = start[out] - rng.integers(2, 7, out.sum())
        days[out] = np.where(after < self.days, after, before)
        face[out] = np.maximum(low[out], 2_000)
        # Below the minimum stays inside the dates: not on a boundary day.
        days[below] = np.clip(days[below], np.maximum(start[below], 0) + 1, np.minimum(end[below], self.days - 1) - 1)
        ts = self._ts(days, HOURLY["ATM"])
        r = self._add(
            n,
            cust=cust,
            ts=ts,
            mtype="RCH_ATM",
            channel="ATM",
            amount=face,
            face=face,
            payment_ref=self._payment_refs(n),
            tag="T8",
        )
        b = self._add(
            n,
            cust=cust,
            ts=ts + rng.integers(2, 31, n).astype("timedelta64[s]"),
            mtype="BONUS_RCH",
            channel="SYSTEM",
            wallet="BONUS",
            amount=self._bonus_amount(face, prow),
            promo=promos["promo_id"].to_numpy()[prow],
            ref=r,
            tag="T8",
        )
        self.t8_keys = np.concatenate([r, b])
        self.details["T8"] = {
            "description": "140 BONUS_RCH with the promotion's percentage but for recharges "
            "below min_recharge or outside its dates",
            "below_min": int(below.sum()),
            "out_of_dates": int(out.sum()),
        }

    # -------------------------------------------- reconciliation & quality --
    def _reconciliation(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        normal = df["tag"].to_numpy() == ""
        day = (df["ts"].to_numpy() - self.start).astype("timedelta64[D]").astype(int)
        # TR2 (trap): 5 days of USSD debits written with a positive amount.
        debit = df["amount"].to_numpy() < 0
        tr2 = normal & debit & (df["channel"].to_numpy() == "USSD") & np.isin(day, self.tr2_days)
        df["true_amount"] = df["amount"]
        df.loc[tr2, "amount"] = -df.loc[tr2, "amount"]
        df["tag"] = np.where(tr2, "TR2", df["tag"])
        free = (
            normal
            & ~tr2
            & ~self.role_mask[df["cust"].to_numpy()]
            & ~np.isin(df["cust"].to_numpy(), self.t3_customers)
            & ~df["_key"].isin(self.recon.get("R5", [])).to_numpy()
        )
        electronic = np.isin(df["mtype"].to_numpy(), ["RCH_DEALER", "RCH_ATM", "RCH_APP"])
        plain = free & electronic & ~df["_key"].isin(list(self.bonus_of) + list(self.reversed)).to_numpy()
        # R4: 80 paid recharges credited above their face value.
        r4 = rng.choice(np.where(plain)[0], self._count(80), replace=False)
        df.loc[r4, "amount"] = df.loc[r4, "face"] + rng.choice([500, 1_000], r4.size)
        df.loc[r4, "true_amount"] = df.loc[r4, "amount"]
        plain[r4] = False
        # R2: 150 recharges credited twice with the same payment_ref.
        r2 = rng.choice(np.where(plain)[0], self._count(150), replace=False)
        dup = df.iloc[r2].copy()
        dup["_key"] = np.arange(self.next_key, self.next_key + r2.size)
        self.next_key += r2.size
        dup["ts"] = dup["ts"].to_numpy() + rng.integers(1, 121, r2.size).astype("timedelta64[s]")
        self.recon["R2"] = df["_key"].to_numpy()[r2].tolist() + dup["_key"].tolist()
        self.recon["R4"] = df["_key"].to_numpy()[r4].tolist()
        free[r2] = False
        free[r4] = False
        # R3: 100 reversals pointing to movements that do not exist.
        n3 = self._count(100)
        cust = rng.choice(self.normal, n3)
        r3 = pd.DataFrame(
            {
                "_key": np.arange(self.next_key, self.next_key + n3),
                "cust": cust,
                "ts": self._ts(self._days(n3, avoid=set(self.tr2_days)), hours=rng.integers(9, 18, n3)),
                "mtype": "REVERSAL",
                "channel": "BACKOFFICE",
                "wallet": "MAIN",
                "amount": -(rng.choice([10, 20], n3) * 100).astype(np.int64),
                "face": -1,
                "reason": rng.choice(REV_REASONS, n3, p=REV_REASON_P),
                "ref": -2,
                "counterparty": -1,
                "tag": "",
            }
        )
        r3["true_amount"] = r3["amount"]
        self.next_key += n3
        self.recon["R3"] = r3["_key"].tolist()
        df = pd.concat([df, dup, r3], ignore_index=True)
        # Balances: random but consistent with the true amount, never negative.
        n = len(df)
        true = df["true_amount"].to_numpy()
        before = np.round(rng.lognormal(np.log(800), 1.0, n)).astype(np.int64) + np.maximum(0, -true)
        df["before"] = before
        df["after"] = before + true
        # R1: 300 scattered rows whose balance_after is off.
        free = np.concatenate([free, np.zeros(n - free.size, bool)])
        r1 = rng.choice(np.where(free)[0], self._count(300), replace=False)
        delta = rng.integers(1, 500, r1.size) * rng.choice([-1, 1], r1.size)
        df.loc[r1, "after"] = np.maximum(0, df.loc[r1, "after"].to_numpy() + delta)
        same = df.loc[r1, "after"].to_numpy() == df.loc[r1, "before"].to_numpy() + df.loc[r1, "amount"].to_numpy()
        df.loc[np.asarray(r1)[same], "after"] += 1
        self.recon["R1"] = df["_key"].to_numpy()[r1].tolist()
        self.tr2_rows = int(tr2.sum())
        return df

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        df = df.sort_values(["ts", "_key"], kind="stable").reset_index(drop=True)
        n = len(df)
        # Movement ids grow with time but leave gaps (some of them used by R3's dangling references).
        n_dangling = int((df["ref"].to_numpy() == -2).sum())
        pool = np.sort(rng.choice(np.arange(10**9, 10**9 + 3 * n), n + n_dangling, replace=False))
        dangling = rng.choice(np.arange(1, n + n_dangling - 1), n_dangling, replace=False)
        used = np.delete(pool, dangling)
        ids = np.char.add("MV", used.astype(str)).astype(object)
        id_of = dict(zip(df["_key"].to_numpy(), ids, strict=True))
        df["movement_id"] = ids
        ref = df["ref"].to_numpy()
        ref_id = np.full(n, None, dtype=object)
        linked = ref >= 0
        missing = {int(k) for k in ref[linked] if k not in id_of}
        assert not missing, f"references to dropped movements: {sorted(missing)[:5]}"
        ref_id[linked] = [id_of[k] for k in ref[linked]]
        d_rows = np.where(ref == -2)[0]
        ref_id[d_rows] = np.char.add("MV", pool[dangling].astype(str)).astype(object)
        df["ref_movement_id"] = ref_id
        self.id_of = id_of
        return df

    def _quality(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        normal = np.where(df["tag"].to_numpy() == "")[0]
        recon = set().union(*(set(v) for v in self.recon.values()))
        normal = normal[~df["_key"].iloc[normal].isin(recon).to_numpy()]
        # movement_type spelling variants (1%).
        var = rng.choice(normal, round(0.01 * len(df)), replace=False)
        df["mtype_out"] = df["mtype"]
        df.loc[var, "mtype_out"] = df.loc[var, "mtype"].str.lower() + " "
        self.quality.append(
            {
                "id": "topup_movement_type_variants",
                "dataset": "topup",
                "column": "movement_type",
                "kind": "inconsistent_categories",
                "rows": int(var.size),
            }
        )
        # One day written in another time zone (UTC-5); rows stay in their date partition.
        day = (df["ts"].to_numpy() - self.start).astype("timedelta64[D]").astype(int)
        df["_day"] = day
        tz = day == self.tz_day
        df.loc[tz, "ts"] = df.loc[tz, "ts"] - pd.Timedelta(hours=5)
        tz_date = self.dates[self.tz_day].isoformat()
        self.quality.append(
            {
                "id": "topup_timezone_shift",
                "dataset": "topup",
                "column": "event_time",
                "kind": "timezone",
                "file": f"topup/date={tz_date}/",
                "rows": int(tz.sum()),
                "offset_hours": -5,
            }
        )
        # Exact duplicate rows (0.5%).
        dup = rng.choice(normal, round(0.005 * len(df)), replace=False)
        self.quality.append(
            {
                "id": "topup_duplicate_rows",
                "dataset": "topup",
                "kind": "duplicates",
                "rows": int(dup.size),
                "share": round(dup.size / len(df), 4),
            }
        )
        df = pd.concat([df, df.iloc[np.sort(dup)]], ignore_index=True)
        df = df.sort_values(["_day", "ts", "movement_id"], kind="stable").reset_index(drop=True)
        self.quality.append(
            {
                "id": "topup_structural_nulls",
                "dataset": "topup",
                "kind": "structural_nulls",
                "columns": {c: list(t) for c, t in COLUMNS_BY_TYPE.items()},
                "note": "type-specific columns are null for the other movement types; "
                "not a quality problem, never impute",
            }
        )
        return df

    # --------------------------------------------------------------- output --
    def _codes(self) -> dict[str, dict[str, str]]:
        """Readable codes, or numeric codes with ``opaque_codes`` (same mapping for a seed)."""
        if not self.cfg.opaque_codes:
            return {}
        rng = np.random.default_rng([self.seed, 99])  # own stream: the data do not depend on the flag
        mapping = {}
        for name, values in (
            ("movement_type", MOVEMENT_TYPES),
            ("channel", TOPUP_CHANNELS),
            ("reason_code", REASON_CODES),
        ):
            codes = rng.choice(np.arange(10, 100), len(values), replace=False)
            mapping[name] = {v: str(c) for v, c in zip(values, codes, strict=True)}
        return mapping

    def build(self, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        say = progress or (lambda _m: None)
        self._roles()
        dealers = self._dealers()
        promotions = self._promotions()
        say(f"topup: normal recharges, bonuses, reversals and bundles for {self.days} days")
        self._normal_recharges()
        self._normal_bonuses()
        self._t3()
        self._expiry_and_reversals()
        self._bundles()
        self._promo_credits()
        self._p2p()
        self._advances()
        say("topup: anomalies T1-T8, reconciliation problems R1-R5, traps TR1-TR3")
        self._t2()
        self._t4()
        self._t5()
        self._t6()
        self._t8()
        n_other = sum(p["_key"].size for p in self.parts)
        self._adjustments(n_other)
        frame = self._frame()
        if self._drop_keys:
            frame = frame[~frame["_key"].isin(self._drop_keys)]
        mark = len(self.parts)
        frame = self._t7(frame)
        added = pd.DataFrame({k: np.concatenate([p[k] for p in self.parts[mark:]]) for k in self.parts[0]})
        frame = pd.concat([frame, added], ignore_index=True)
        frame = self._reconciliation(frame.reset_index(drop=True))
        frame = self._finalize(frame)
        frame = self._quality(frame)
        self.frame = frame
        self.dealers, self.promotions = dealers, promotions
        self.codes = self._codes()
        return self._truth()

    def _truth(self) -> dict[str, Any]:
        msisdn = self.pop.msisdn
        ids = lambda keys: [self.id_of[k] for k in keys]  # noqa: E731
        rows: list[dict[str, str]] = []

        def add(column: str, values: Any, anomaly: str, kind: str, movements: Any) -> None:
            for v in values:
                rows.append(
                    {
                        "dataset": "topup",
                        "entity_column": column,
                        "entity_id": str(v),
                        "anomaly_id": anomaly,
                        "anomaly_type": kind,
                    }
                )
            for m in ids(movements):
                rows.append(
                    {
                        "dataset": "topup",
                        "entity_column": "movement_id",
                        "entity_id": m,
                        "anomaly_id": anomaly,
                        "anomaly_type": kind,
                    }
                )

        add("agent_user", self.t1["agents"], "T1", "entity_outlier", self.t1["keys"])
        add("msisdn", msisdn[self.roles["T2"]], "T2", "linked_events", self.t2_keys)
        add("promo_id", ["P017"], "T3", "temporal", self.t3_keys)
        add("voucher_batch_id", self.batch_ids[self.t4_batches], "T4", "burst", self.t4_keys)
        add("dealer_id", sorted(self.t5_dealers), "T5", "entity_outlier", self.t5_keys)
        add("msisdn", msisdn[np.concatenate([self.roles["T6c"], self.roles["T6f"]])], "T6", "graph", self.t6_keys)
        add("msisdn", msisdn[self.roles["T7"]], "T7", "sequence", self.t7_keys)
        add("msisdn", msisdn[self.roles["T8"]], "T8", "contextual", self.t8_keys)
        recon = [{"movement_id": self.id_of[k], "issue_id": issue} for issue, keys in self.recon.items() for k in keys]
        recon.sort(key=lambda r: (r["issue_id"], r["movement_id"]))
        chains = sorted(self.dealer_ids[self.dealer_kind == "CHAIN"].tolist())
        self.traps += [
            {
                "id": "TR1_legitimate_campaign",
                "dataset": "topup",
                "entity_column": "promo_id",
                "entity_ids": ["P009"],
                "date": self.tr1["date"],
                "description": "Promotion P009 triples the PROMO_CREDIT volume on one day, within its dates and rules",
                "movements": int(self.tr1["extra"]),
                "expected_verdict": "explained",
            },
            {
                "id": "TR2_sign_error",
                "dataset": "topup",
                "column": "amount",
                "channel": "USSD",
                "dates": [self.dates[d].isoformat() for d in self.tr2_days],
                "rows": self.tr2_rows,
                "description": "USSD debits written with a positive amount for 5 days (balance invariant violated)",
                "expected_verdict": "artifact",
            },
            {
                "id": "TR3_retail_chains",
                "dataset": "topup",
                "entity_column": "dealer_id",
                "entity_ids": chains,
                "description": "The 30 CHAIN dealers have volumes far above the median dealer",
                "expected_verdict": "explained",
            },
        ]
        self.quality.append(
            {
                "id": "topup_sign_error",
                "dataset": "topup",
                "column": "amount",
                "kind": "sign_error",
                "channel": "USSD",
                "dates": self.traps[-2]["dates"],
                "rows": self.tr2_rows,
            }
        )
        dealer_vol = self.frame[self.frame["mtype"] == "RCH_DEALER"].groupby("dealer")["true_amount"].sum()
        chain_share = float(dealer_vol[dealer_vol.index.isin(chains)].sum() / dealer_vol.sum())
        median = float(dealer_vol.median())
        self.details["TR3"] = {
            "chain_volume_share": round(chain_share, 4),
            "chain_over_median": round(float(dealer_vol[dealer_vol.index.isin(chains)].mean()) / median, 1),
        }
        self.details["opaque_codes"] = self.codes or None
        expected = {
            "datasets": [
                {"name": "topup", "role": "primary"},
                {"name": "promotions", "role": "enrichment"},
                {"name": "dealers", "role": "enrichment"},
                {"name": "voucher_batches", "role": "enrichment"},
                {"name": "cust_segmentation", "role": "enrichment"},
            ],
            "joins": [
                {
                    "left": "topup",
                    "right": "cust_segmentation",
                    "keys": [["msisdn", "customer_id"]],
                    "key_normalization": ["strip_prefix_plus351"],
                    "cardinality": "N:1",
                },
                {
                    "left": "topup",
                    "right": "promotions",
                    "keys": [["promo_id", "promo_id"]],
                    "key_normalization": ["none"],
                    "sparse": True,
                },
                {
                    "left": "topup",
                    "right": "dealers",
                    "keys": [["dealer_id", "dealer_id"]],
                    "key_normalization": ["none"],
                    "sparse": True,
                },
                {
                    "left": "topup",
                    "right": "voucher_batches",
                    "keys": [["voucher_batch_id", "batch_id"]],
                    "key_normalization": ["none"],
                    "sparse": True,
                },
                {
                    "left": "topup",
                    "right": "topup",
                    "keys": [["ref_movement_id", "movement_id"]],
                    "key_normalization": ["none"],
                    "self_reference": True,
                },
                {
                    "left": "voucher_batches",
                    "right": "dealers",
                    "keys": [["dealer_id", "dealer_id"]],
                    "key_normalization": ["none"],
                },
            ],
            "grain": {
                "dataset": "topup",
                "columns": ["movement_id"],
                "note": "unique except for the planted duplicate rows",
            },
            "time_column": "event_time",
            "entities": ["msisdn", "dealer_id", "agent_user", "voucher_batch_id", "promo_id"],
            "invariants": ["balance_after = balance_before + amount"],
            "structural_nulls": {c: list(t) for c, t in COLUMNS_BY_TYPE.items()},
            "opaque_codes": self.codes or None,
        }
        self.recon_rows = recon
        self.anomaly_rows = rows
        self.expected = expected
        return {
            "topup_rows": len(self.frame),
            "promotions_rows": len(self.promotions),
            "dealers_rows": len(self.dealers),
            "voucher_batches_rows": len(self.batches),
            "topup_anomalies": {
                a: int(sum(1 for r in rows if r["anomaly_id"] == a and r["entity_column"] != "movement_id"))
                for a in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
            },
            "reconciliation": {k: len(v) for k, v in self.recon.items()},
            "bonus_share": self.details["bonus_share"],
        }

    def write(self, out: Path, truth: Path) -> None:
        df = self.frame
        codes = self.codes
        if codes:
            mt = codes["movement_type"]
            variant = df["mtype_out"] != df["mtype"]
            df["mtype_out"] = df["mtype"].map(mt)
            df.loc[variant, "mtype_out"] = df.loc[variant, "mtype_out"] + " "
            df["channel"] = df["channel"].map(codes["channel"])
            df["reason"] = df["reason"].map(codes["reason_code"]).where(df["reason"].notna(), None)

        def money(cents: pd.Series, nullable: bool = False) -> pa.Array:
            values = cents.to_numpy()
            text = np.char.mod("%.2f", values / 100).astype(object)
            if nullable:
                text[values < 0] = None
            return pa.array(text, pa.string()).cast(pa.decimal128(12, 2))

        msisdn = self.pop.msisdn
        cust, cp = df["cust"].to_numpy(), df["counterparty"].to_numpy()
        counterparty = np.full(len(df), None, dtype=object)
        counterparty[cp >= 0] = msisdn[cp[cp >= 0]]
        table = pa.table(
            {
                "movement_id": _strings(df["movement_id"]),
                "msisdn": _strings(msisdn[cust]),
                "event_time": pa.array(df["ts"].to_numpy().astype("datetime64[us]")),
                "movement_type": _strings(df["mtype_out"]),
                "channel": _strings(df["channel"]),
                "wallet": _strings(df["wallet"]),
                "amount": money(df["amount"]),
                "face_value": money(df["face"], nullable=True),
                "balance_before": money(df["before"]),
                "balance_after": money(df["after"]),
                "voucher_serial": _strings(df["serial"]),
                "voucher_batch_id": _strings(df["batch"]),
                "dealer_id": _strings(df["dealer"]),
                "agent_user": _strings(df["agent"]),
                "promo_id": _strings(df["promo"]),
                "reason_code": _strings(df["reason"]),
                "ref_movement_id": _strings(df["ref_movement_id"]),
                "counterparty_msisdn": _strings(counterparty),
                "payment_ref": _strings(df["payment_ref"]),
            }
        )
        day = df["_day"].to_numpy()
        bounds = np.flatnonzero(np.diff(day)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(df)]])
        for s, e in zip(starts, ends, strict=True):
            folder = out / "topup" / f"date={self.dates[int(day[s])].isoformat()}"
            folder.mkdir(parents=True, exist_ok=True)
            pq.write_table(table.slice(s, e - s), folder / "part-0.parquet", compression="zstd")
        for name, frame in (
            ("promotions", self.promotions),
            ("dealers", self.dealers),
            ("voucher_batches", self.batches),
        ):
            (out / name).mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pandas(frame, preserve_index=False), out / name / "part-0.parquet", compression="zstd"
            )
        if self.cfg.with_readme:
            from jellybean.datagen.readmes import topup_readmes

            for name, text in topup_readmes(codes).items():
                (out / name / "README.md").write_text(text, encoding="utf-8")
        truth.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.recon_rows).to_parquet(truth / "reconciliation_issues.parquet", index=False)
        import yaml

        (truth / "expected_spec_topup.yaml").write_text(
            yaml.safe_dump(self.expected, sort_keys=False), encoding="utf-8"
        )


def _strings(values: Any) -> pa.Array:
    """A string column; None and NaN become null."""
    return pa.array(pd.Series(values, dtype=object), pa.string(), from_pandas=True)


def _choice_rows(rng: np.random.Generator, probs: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """One category per element, each element with its own row of ``probs`` (vectorised)."""
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(rows.shape[0])
    return (u[:, None] > cdf[rows]).sum(axis=1)


def merge_json_list(path: Path, new: list[dict[str, Any]], datasets: set[str]) -> None:
    """Replace the entries of ``datasets`` in a ground-truth JSON list, keeping the others."""
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    kept = [e for e in old if e.get("dataset") not in datasets]
    path.write_text(json.dumps(kept + new, indent=2), encoding="utf-8")
