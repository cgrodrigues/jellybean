# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Synthetic telecom/e-commerce data with planted anomalies (spec section 9).

Produces Parquet datasets under ``out`` and the ground truth under ``truth``
(which must be outside ``out`` so agents can never read it):

* ``traffic``           one row per call, Hive-partitioned by ``date=YYYY-MM-DD/hour=HH``
                        (or by ``date`` only with ``partition="day"``);
* ``cust_segmentation`` one row per customer (plus 15 duplicated keys);
* ``orders``            one row per order;
* ``topup``             prepaid wallet movements, with ``promotions``, ``dealers``
                        and ``voucher_batches`` (``requirements_2.md``; see
                        :mod:`jellybean.datagen.topup`).

``datasets`` selects the scenarios. Every scenario uses the same customers, so
datasets generated separately with the same seed and customers still join.

Normal behaviour is structured (daily/weekly seasonality, long tails, real
differences between segments). On top of it the generator plants six
anomalies (A-F), data quality problems, join difficulties and two traps for the
critic. Everything is driven by a single ``numpy.random.Generator`` seeded from
``seed``, so the output is identical for the same arguments.

All generation is vectorised with numpy; pandas is only used to write files.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from jellybean.datagen.readmes import DATASET_READMES
from jellybean.datagen.topup import TOPUP_DATASETS, Population, TopupConfig, TopupGenerator, merge_json_list

# --------------------------------------------------------------------------- #
# Static domain parameters                                                    #
# --------------------------------------------------------------------------- #
SEGMENTS = np.array(["Consumer", "Business", "Youth", "Premium"])
SEGMENT_P = np.array([0.58, 0.20, 0.14, 0.08])
CONSUMER, BUSINESS, YOUTH, PREMIUM = 0, 1, 2, 3

TARIFFS = np.array(["basic", "standard", "unlimited", "business_pro"])
# Tariff distribution per segment (rows follow SEGMENTS order).
TARIFF_P = np.array(
    [
        [0.50, 0.35, 0.15, 0.00],  # Consumer
        [0.00, 0.00, 0.30, 0.70],  # Business
        [0.40, 0.40, 0.20, 0.00],  # Youth
        [0.00, 0.20, 0.80, 0.00],  # Premium
    ]
)
BASIC = 0
# Domestic price per started minute, by tariff.
DOMESTIC_RATE = np.array([0.09, 0.05, 0.0, 0.02])

REGIONS = np.array(["Lisboa", "Porto", "Norte", "Centro", "Alentejo", "Algarve", "Acores", "Madeira"])
REGION_P = np.array([0.30, 0.20, 0.14, 0.16, 0.06, 0.06, 0.04, 0.04])
N_CELLS = 1000
CELL_BASE = 1000  # cell ids are 1000..1999
CREDIT = np.array(["A", "B", "C", "D"])
AGE_BANDS = np.array(["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])

# Destination countries for international calls: dial code, price/minute, weight.
COUNTRIES = np.array(["ES", "FR", "GB", "DE", "US", "BR", "AO", "CH", "CU", "SO", "TV", "NU"])
DIAL = np.array(["+34", "+33", "+44", "+49", "+1", "+55", "+244", "+41", "+53", "+252", "+688", "+683"])
INTL_RATE = np.array([0.25, 0.25, 0.30, 0.30, 0.35, 0.60, 0.90, 0.35, 2.50, 3.00, 3.50, 3.00])
INTL_W = np.array([0.24, 0.17, 0.15, 0.12, 0.08, 0.10, 0.08, 0.055, 0.0013, 0.0012, 0.0013, 0.0012])
EXPENSIVE = np.array([8, 9, 10, 11])  # indices of premium-rate destinations (anomaly A)

# Hourly activity profiles (relative weights, 24 values).
_H_CONSUMER = np.array(
    [
        0.6,
        0.3,
        0.15,
        0.1,
        0.1,
        0.2,
        0.6,
        1.4,
        2.4,
        2.8,
        3.0,
        3.2,
        3.4,
        3.2,
        3.0,
        3.2,
        3.6,
        4.2,
        5.2,
        5.8,
        5.6,
        4.6,
        3.0,
        1.4,
    ]
)
_H_BUSINESS = np.array(
    [
        0.15,
        0.08,
        0.05,
        0.05,
        0.05,
        0.1,
        0.4,
        1.6,
        5.0,
        7.0,
        7.4,
        7.0,
        4.6,
        5.6,
        7.0,
        7.2,
        6.6,
        5.2,
        2.6,
        1.4,
        0.9,
        0.6,
        0.4,
        0.25,
    ]
)
_H_YOUTH = np.array(
    [
        1.6,
        1.0,
        0.5,
        0.25,
        0.15,
        0.15,
        0.3,
        0.8,
        1.4,
        1.8,
        2.2,
        2.6,
        3.0,
        3.0,
        3.0,
        3.2,
        3.6,
        4.2,
        5.0,
        5.8,
        6.2,
        6.0,
        4.6,
        3.0,
    ]
)
HOURLY = np.vstack([_H_CONSUMER, _H_BUSINESS, _H_YOUTH, _H_CONSUMER])
HOURLY = HOURLY / HOURLY.sum(axis=1, keepdims=True)
# Weekday weights Mon..Sun per segment.
WEEKLY = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.05, 1.15, 1.05],  # Consumer
        [1.0, 1.0, 1.0, 1.0, 0.95, 0.30, 0.20],  # Business
        [1.0, 1.0, 1.0, 1.0, 1.10, 1.35, 1.20],  # Youth
        [1.0, 1.0, 1.0, 1.0, 1.05, 1.10, 1.00],  # Premium
    ]
)
# Mean calls per customer-day per segment before the global rescale.
BASE_RATE = np.array([2.0, 4.0, 3.0, 3.0])
# Base probability that a call is international, per segment.
INTL_P = np.array([0.025, 0.12, 0.02, 0.06])

PRODUCTS = np.array(
    [
        "phone_case",
        "charger",
        "earbuds",
        "smartwatch",
        "tablet",
        "smartphone",
        "router",
        "sim_topup",
        "laptop",
        "insurance",
    ]
)
PRODUCT_PRICE = np.array([15, 25, 60, 220, 380, 650, 90, 20, 1100, 12], dtype=float)
PRODUCT_P = np.array([0.18, 0.16, 0.13, 0.06, 0.04, 0.05, 0.06, 0.25, 0.02, 0.05])
CHANNELS = np.array(["web", "app", "store", "call_center"])
CHANNEL_P = np.array([0.38, 0.34, 0.22, 0.06])
PAYMENTS = np.array(["card", "mbway", "paypal", "invoice", "multibanco"])
PAYMENT_P = np.array([0.40, 0.25, 0.12, 0.08, 0.15])

# Target volumes with default arguments (scaled by --scale and population size).
CALLS_PER_CUSTOMER_DAY = 1_500_000 / (20_000 * 30)
ORDERS_PER_CUSTOMER = 120_000 / 20_000

TRUTH_COLUMNS = ["dataset", "entity_column", "entity_id", "anomaly_id", "anomaly_type"]
TELECOM_DATASETS = ("traffic", "cust_segmentation", "orders")
# Scenarios a user can ask for (``topup`` brings its three reference tables).
SCENARIOS = (*TELECOM_DATASETS, "topup")


@dataclass(frozen=True)
class GeneratorConfig:
    """Arguments of the generator (mirrors the CLI options)."""

    out: Path
    truth: Path
    seed: int = 42
    customers: int = 20_000
    days: int = 30
    with_readme: bool = False
    scale: float = 1.0
    start_date: date = date(2026, 10, 1)
    overwrite: bool = False
    # Partitioning of traffic: "hour" -> date=YYYY-MM-DD/hour=HH, "day" -> date=YYYY-MM-DD.
    partition: Literal["day", "hour"] = "hour"
    # Scenarios to generate. The library default is the original three datasets;
    # the CLI default (``generate-data``) is every scenario.
    datasets: tuple[str, ...] = TELECOM_DATASETS
    topup_days: int = 90
    opaque_codes: bool = False  # numeric codes for movement_type, channel and reason_code

    def validate(self) -> None:
        unknown = set(self.datasets) - set(SCENARIOS) - set(TOPUP_DATASETS)
        if unknown or not self.datasets:
            raise ValueError(f"unknown datasets {sorted(unknown)}; choose from {list(SCENARIOS)}")
        if self.wants_topup and self.topup_days < 60:
            raise ValueError("topup_days must be >= 60")
        if self.customers < 1_000:
            raise ValueError("customers must be >= 1000 (anomalies need room)")
        if self.days < 7:
            raise ValueError("days must be >= 7 (weekly pattern and traps need distinct days)")
        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        out, truth = self.out.resolve(), self.truth.resolve()
        if truth == out or truth.is_relative_to(out):
            raise ValueError("truth folder must be outside the data folder (agents must not see it)")

    @property
    def wants_topup(self) -> bool:
        return bool(set(self.datasets) & set(TOPUP_DATASETS))

    @property
    def telecom(self) -> tuple[str, ...]:
        return tuple(d for d in TELECOM_DATASETS if d in self.datasets)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Dataset folders this configuration writes."""
        return (*self.telecom, *(TOPUP_DATASETS if self.wants_topup else ()))


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _choice_rows(rng: np.random.Generator, probs: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Sample one category per element where each element has its own row of ``probs``.

    ``probs`` has shape (k, m) and ``rows`` selects a row per sample; this is the
    vectorised equivalent of calling ``rng.choice(m, p=probs[row])`` in a loop.
    """
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(rows.shape[0])
    return (u[:, None] > cdf[rows]).sum(axis=1)


def _digits(rng: np.random.Generator, n: int, width: int) -> np.ndarray:
    """``n`` random zero-padded digit strings of length ``width``."""
    return pd.Series(rng.integers(0, 10**width, n)).astype(str).str.zfill(width).to_numpy()


def _unique_ints(rng: np.random.Generator, n: int, low: int, high: int) -> np.ndarray:
    """``n`` distinct integers in ``[low, high)``, deterministic for the rng state."""
    values = np.unique(rng.integers(low, high, int(n * 1.2) + 10))
    while values.size < n:
        values = np.unique(np.concatenate([values, rng.integers(low, high, n)]))
    return rng.permutation(values)[:n]


@dataclass
class Calls:
    """Column arrays of a batch of calls (before string formatting)."""

    cust: np.ndarray  # customer index (int)
    ts: np.ndarray  # datetime64[s]
    dest: np.ndarray  # destination number (str)
    country: np.ndarray  # country index into COUNTRIES, -1 = PT
    duration: np.ndarray  # seconds (int)
    cell: np.ndarray  # cell id (int)
    imei: np.ndarray  # imei (str)
    outgoing: np.ndarray  # bool
    tag: np.ndarray  # anomaly tag: "" or "A".."F"/"E"

    @staticmethod
    def concat(parts: list[Calls]) -> Calls:
        return Calls(*(np.concatenate([getattr(p, f) for p in parts]) for f in Calls.__dataclass_fields__))

    def __len__(self) -> int:
        return self.cust.shape[0]


# --------------------------------------------------------------------------- #
# Generator                                                                   #
# --------------------------------------------------------------------------- #
class DummyDataGenerator:
    """Builds all datasets and the ground truth for one configuration."""

    def __init__(self, cfg: GeneratorConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.start = np.datetime64(cfg.start_date.isoformat(), "s")
        self.dates = [cfg.start_date + timedelta(days=d) for d in range(cfg.days)]
        self.dow = np.array([d.weekday() for d in self.dates])
        self.quality: list[dict[str, Any]] = []
        self.traps: list[dict[str, Any]] = []

    # ---------------------------------------------------------- population --
    def _build_customers(self) -> None:
        rng, n = self.rng, self.cfg.customers
        # Portuguese mobile numbers: 9 digits starting with 91/92/93/96.
        prefixes = rng.choice(np.array([91, 92, 93, 96]), n)
        national = prefixes * 10**7 + _unique_ints(rng, n, 0, 10**7)
        # Guarantee uniqueness after combining prefixes.
        _, first = np.unique(national, return_index=True)
        dup_mask = np.ones(n, bool)
        dup_mask[first] = False
        while dup_mask.any():
            national[dup_mask] = rng.choice([91, 92, 93, 96], dup_mask.sum()) * 10**7 + rng.integers(
                0, 10**7, dup_mask.sum()
            )
            _, first = np.unique(national, return_index=True)
            dup_mask = np.ones(n, bool)
            dup_mask[first] = False
        self.national = national.astype(str)
        self.msisdn = np.char.add("+351", self.national)

        self.segment = rng.choice(4, n, p=SEGMENT_P)
        self.tariff = _choice_rows(rng, TARIFF_P, self.segment)
        self.region = rng.choice(len(REGIONS), n, p=REGION_P)
        # Each region owns a contiguous block of cells.
        block = N_CELLS // len(REGIONS)
        self.home_cell = CELL_BASE + self.region * block + rng.integers(0, block, n)
        self.work_cell = CELL_BASE + self.region * block + rng.integers(0, block, n)
        self.imei = np.char.add("35", _digits(rng, n, 13))
        # Long-tailed individual activity, rescaled to the target daily mean.
        activity = BASE_RATE[self.segment] * rng.lognormal(0.0, 0.7, n)
        self.rate = activity / activity.mean() * CALLS_PER_CUSTOMER_DAY * self.cfg.scale
        # "Behaviour segment" drives call patterns; it differs from the recorded
        # segment only for anomaly F (consumer/basic customers behaving as business).
        self.behaviour = self.segment.copy()

        # Frequent contacts: mostly other customers, some external numbers.
        k = 3 + rng.poisson(12, n)
        self.contact_offset = np.concatenate([[0], np.cumsum(k)[:-1]])
        self.contact_count = k
        total = int(k.sum())
        on_net = rng.random(total) < 0.7
        contacts = np.empty(total, dtype=object)
        contacts[on_net] = self.msisdn[rng.integers(0, n, on_net.sum())]
        external = np.char.add(
            "+351",
            np.char.add(
                rng.choice(["21", "22", "91", "93", "96"], (~on_net).sum()), _digits(rng, int((~on_net).sum()), 7)
            ),
        )
        contacts[~on_net] = external
        self.contacts = contacts

    def _assign_roles(self) -> None:
        """Pick disjoint customer sets for anomalies, traps and join issues."""
        rng, n = self.rng, self.cfg.customers
        order = rng.permutation(n)
        sizes = {"missing": round(0.03 * n), "A": 25, "B": 15, "C": 40, "F": 60, "null_cell": 300, "dup_key": 15}
        self.roles: dict[str, np.ndarray] = {}
        start = 0
        for role, size in sizes.items():
            self.roles[role] = np.sort(order[start : start + size])
            start += size
        # Anomaly F: recorded as Consumer/basic, behaves like Business.
        f = self.roles["F"]
        self.segment[f] = CONSUMER
        self.tariff[f] = BASIC
        self.behaviour[f] = BUSINESS
        self.rate[f] = np.maximum(self.rate[f], np.quantile(self.rate[self.segment == BUSINESS], 0.5))
        # Anomaly C: SIM-box style SIMs only produce their anomalous traffic.
        self.rate[self.roles["C"]] = 0.0

    # --------------------------------------------------------------- calls --
    def _normal_calls(self) -> Calls:
        rng, days = self.rng, self.cfg.days
        counts = rng.poisson(self.rate * days)
        cust = np.repeat(np.arange(self.cfg.customers), counts)
        n = cust.shape[0]
        beh = self.behaviour[cust]

        # Day: weekly profile of the behaviour segment.
        day_w = WEEKLY[:, self.dow]
        day_p = day_w / day_w.sum(axis=1, keepdims=True)
        day = _choice_rows(rng, day_p, beh)
        weekend = self.dow[day] >= 5
        # Hour: business behaves like consumer at weekends.
        hour_profile = np.where((beh == BUSINESS) & weekend, CONSUMER, beh)
        hour = _choice_rows(rng, HOURLY, hour_profile)
        secs = rng.integers(0, 3600, n)
        ts = self.start + (day * 86400 + hour * 3600 + secs).astype("timedelta64[s]")

        work_hours = (~weekend) & (hour >= 9) & (hour < 18)
        p_intl = INTL_P[beh] * np.where(beh == BUSINESS, np.where(work_hours, 1.6, 0.4), 1.0)
        intl = rng.random(n) < p_intl
        country = np.full(n, -1)
        country[intl] = rng.choice(len(COUNTRIES), intl.sum(), p=INTL_W / INTL_W.sum())

        # Domestic destinations: skewed choice among the customer's contacts.
        j = np.floor(self.contact_count[cust] * rng.random(n) ** 1.8).astype(int)
        dest = self.contacts[self.contact_offset[cust] + j].astype(object)
        # International destinations: one of 3 stable numbers per customer and country.
        ic = np.where(intl)[0]
        slot = rng.integers(0, 3, ic.size)
        local = (cust[ic].astype(np.int64) * 7919 + country[ic] * 104729 + slot * 1299709) % 900_000_000 + 100_000_000
        dest[ic] = np.char.add(DIAL[country[ic]], local.astype(str))

        duration = np.clip(rng.lognormal(np.log(70), 1.0, n) * np.where(intl, 1.4, 1.0), 1, 7200).astype(int)
        # Cell: work cell during business hours, home otherwise, plus mobility.
        cell = np.where((beh == BUSINESS) & work_hours, self.work_cell[cust], self.home_cell[cust])
        move = rng.random(n)
        block = N_CELLS // len(REGIONS)
        local_move = move < 0.12
        cell[local_move] = CELL_BASE + self.region[cust[local_move]] * block + rng.integers(0, block, local_move.sum())
        far_move = move > 0.97
        cell[far_move] = CELL_BASE + rng.integers(0, N_CELLS, far_move.sum())
        outgoing = rng.random(n) < 0.55
        return Calls(
            cust,
            ts,
            dest,
            country,
            duration,
            cell,
            self.imei[cust].astype(object),
            outgoing,
            np.full(n, "", dtype=object),
        )

    def _anomaly_a(self) -> Calls:
        """25 customers: night bursts of long calls to expensive destinations."""
        rng, parts = self.rng, []
        for c in self.roles["A"]:
            burst_days = rng.choice(self.cfg.days, rng.integers(2, 4), replace=False)
            for d in burst_days:
                k = int(rng.integers(8, 21))
                secs = d * 86400 + rng.integers(1 * 3600, 5 * 3600, k)
                country = rng.choice(EXPENSIVE, k)
                dest = np.char.add(DIAL[country], _digits(rng, k, 8))
                parts.append(
                    self._batch(np.full(k, c), secs, dest, country, rng.integers(1200, 3601, k), "A", outgoing=True)
                )
        return Calls.concat(parts)

    def _anomaly_b(self) -> Calls:
        """15 numbers: hundreds of 1-3 s calls to distinct destinations (wangiri)."""
        rng, parts = self.rng, []
        for c in self.roles["B"]:
            k = int(rng.integers(250, 601))
            secs = rng.integers(0, self.cfg.days, k) * 86400 + rng.integers(8 * 3600, 22 * 3600, k)
            dest = np.char.add("+3519", _digits(rng, k, 8))
            parts.append(
                self._batch(np.full(k, c), secs, dest, np.full(k, -1), rng.integers(1, 4, k), "B", outgoing=True)
            )
        return Calls.concat(parts)

    def _anomaly_c(self) -> Calls:
        """40 SIMs on 3 IMEIs and 2 cells, outgoing only, many destinations, no mobility."""
        rng = self.rng
        sims = self.roles["C"]
        shared_imei = np.char.add("86", _digits(rng, 3, 13))
        shared_cells = CELL_BASE + rng.choice(N_CELLS, 2, replace=False)
        self.c_imeis, self.c_cells = shared_imei.tolist(), shared_cells.tolist()
        parts = []
        for i, c in enumerate(sims):
            per_day = rng.integers(25, 41, self.cfg.days)
            day = np.repeat(np.arange(self.cfg.days), per_day)
            k = day.size
            secs = day * 86400 + rng.integers(7 * 3600, 23 * 3600, k)
            dest = np.char.add("+3519", _digits(rng, k, 8))
            batch = self._batch(
                np.full(k, c),
                secs,
                dest,
                np.full(k, -1),
                np.clip(rng.lognormal(np.log(110), 0.5, k), 5, 1800).astype(int),
                "C",
                outgoing=True,
            )
            batch.imei = np.full(k, shared_imei[i % 3], dtype=object)
            batch.cell = np.full(k, shared_cells[i % 2])
            parts.append(batch)
        return Calls.concat(parts)

    def _anomaly_e(self, excluded_days: set[int]) -> Calls:
        """Volume spike in one cell during 6 hours of one day."""
        rng = self.rng
        candidates = [d for d in range(1, self.cfg.days - 1) if d not in excluded_days]
        self.e_day = int(rng.choice(candidates))
        self.e_cell = int(CELL_BASE + rng.integers(0, N_CELLS))
        self.e_hour = 10
        k = int(900 * self.cfg.scale * self.cfg.customers / 20_000) + 150
        normal = np.setdiff1d(np.arange(self.cfg.customers), np.concatenate(list(self.roles.values())))
        cust = rng.choice(normal, k)
        secs = self.e_day * 86400 + rng.integers(self.e_hour * 3600, (self.e_hour + 6) * 3600, k)
        dest = self.contacts[self.contact_offset[cust]].astype(object)
        batch = self._batch(
            cust,
            secs,
            dest,
            np.full(k, -1),
            np.clip(rng.lognormal(np.log(60), 0.9, k), 1, 3600).astype(int),
            "E",
            outgoing=rng.random(k) < 0.6,
        )
        batch.cell = np.full(k, self.e_cell)
        return batch

    def _batch(self, cust, secs, dest, country, duration, tag, outgoing) -> Calls:
        """Build a :class:`Calls` batch with default cell/imei for the customers."""
        k = cust.shape[0]
        out = np.full(k, outgoing) if isinstance(outgoing, bool) else outgoing
        return Calls(
            cust.astype(int),
            self.start + np.asarray(secs).astype("timedelta64[s]"),
            np.asarray(dest, dtype=object),
            np.asarray(country),
            np.asarray(duration).astype(int),
            self.home_cell[cust],
            self.imei[cust].astype(object),
            out,
            np.full(k, tag, dtype=object),
        )

    # ---------------------------------------------------------- traffic df --
    def _traffic_frame(self) -> pd.DataFrame:
        rng = self.rng
        # Trap/quality days chosen up-front so the temporal spike avoids them.
        self.tz_day, self.units_day = (int(d) for d in rng.choice(np.arange(2, self.cfg.days - 1), 2, replace=False))
        calls = Calls.concat(
            [
                self._normal_calls(),
                self._anomaly_a(),
                self._anomaly_b(),
                self._anomaly_c(),
                self._anomaly_e({self.tz_day, self.units_day}),
            ]
        )
        # Chronological order, then sequential call ids.
        order = np.argsort(calls.ts, kind="stable")
        calls = Calls(*(getattr(calls, f)[order] for f in Calls.__dataclass_fields__))
        n = len(calls)

        country_code = np.where(calls.country >= 0, COUNTRIES[np.maximum(calls.country, 0)], "PT")
        minutes = np.ceil(calls.duration / 60.0)
        rate = np.where(
            calls.country >= 0, INTL_RATE[np.maximum(calls.country, 0)], DOMESTIC_RATE[self.tariff[calls.cust]]
        )
        charge = np.round(np.where(calls.outgoing, rate * minutes, 0.0), 2)
        df = pd.DataFrame(
            {
                "call_id": np.char.add("C", pd.Series(np.arange(1, n + 1)).astype(str).str.zfill(10).to_numpy()),
                "msisdn": self.msisdn[calls.cust],
                "destination_number": calls.dest.astype(str),
                "destination_country": country_code,
                "start_time": calls.ts.astype("datetime64[us]"),
                "duration_sec": calls.duration.astype("int64"),
                "charge": charge,
                "cell_id": pd.array(calls.cell, dtype="Int32"),
                "imei": calls.imei.astype(str),
                "call_type": np.where(calls.outgoing, "outgoing", "incoming"),
            }
        )
        df["_day"] = ((calls.ts - self.start) // np.timedelta64(1, "D")).astype(int)
        df["_hour"] = (((calls.ts - self.start) // np.timedelta64(1, "h")) % 24).astype(int)
        df["_tag"] = calls.tag
        df["_cust"] = calls.cust
        return self._plant_traffic_quality(df)

    def _plant_traffic_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        n = len(df)
        normal_rows = np.where(df["_tag"].to_numpy() == "")[0]

        # 1) cell_id nulls (2%), concentrated in 300 customers (imputation trap).
        trap_cust = self.roles["null_cell"]
        trap_rows = np.where(np.isin(df["_cust"].to_numpy(), trap_cust))[0]
        target = int(0.02 * n)
        if trap_rows.size > target:
            # Small populations: null only part of the trap customers' rows, so the
            # global share stays at 2% while remaining concentrated in them.
            trap_rows = np.sort(rng.choice(trap_rows, target, replace=False))
        extra = max(0, target - trap_rows.size)
        pool = np.setdiff1d(normal_rows, trap_rows)
        extra_rows = rng.choice(pool, extra, replace=False) if extra else np.array([], int)
        null_cell = np.concatenate([trap_rows, extra_rows])
        df.loc[null_cell, "cell_id"] = pd.NA
        self.quality.append(
            {
                "id": "null_cell_id",
                "dataset": "traffic",
                "column": "cell_id",
                "kind": "nulls",
                "rows": int(null_cell.size),
                "share": round(null_cell.size / n, 4),
                "concentrated_in_customers": int(trap_cust.size),
            }
        )
        self.traps.append(
            {
                "id": "imputation_artifact",
                "dataset": "traffic",
                "column": "cell_id",
                "description": "Rows with null cell_id are concentrated in 300 customers. Imputing a "
                "constant creates a spurious cluster of these customers.",
                "entity_column": "msisdn",
                "entity_ids": self.msisdn[trap_cust].tolist(),
                "expected_verdict": "artifact",
            }
        )

        # 2) imei nulls (1%).
        imei_null = rng.choice(n, int(0.01 * n), replace=False)
        df.loc[imei_null, "imei"] = None
        self.quality.append(
            {
                "id": "null_imei",
                "dataset": "traffic",
                "column": "imei",
                "kind": "nulls",
                "rows": int(imei_null.size),
                "share": round(imei_null.size / n, 4),
            }
        )

        # 3) Negative durations (200 rows, outside the units-trap day).
        days = df["_day"].to_numpy()
        eligible = normal_rows[days[normal_rows] != self.units_day]
        neg = rng.choice(eligible, min(200, eligible.size), replace=False)
        df.loc[neg, "duration_sec"] = -df.loc[neg, "duration_sec"]
        self.quality.append(
            {
                "id": "negative_duration",
                "dataset": "traffic",
                "column": "duration_sec",
                "kind": "invalid_values",
                "rows": int(neg.size),
                "call_ids": df.loc[np.sort(neg), "call_id"].tolist(),
            }
        )

        # 4) One day stored in another time zone (UTC-5 instead of local time).
        tz_rows = days == self.tz_day
        df.loc[tz_rows, "start_time"] = df.loc[tz_rows, "start_time"] - pd.Timedelta(hours=5)
        tz_date = self.dates[self.tz_day].isoformat()
        self.quality.append(
            {
                "id": "timezone_shift",
                "dataset": "traffic",
                "column": "start_time",
                "kind": "timezone",
                "file": f"traffic/date={tz_date}/",
                "rows": int(tz_rows.sum()),
                "offset_hours": -5,
            }
        )

        # 5) Units trap: one daily file has duration in milliseconds.
        ms_rows = days == self.units_day
        df.loc[ms_rows, "duration_sec"] = df.loc[ms_rows, "duration_sec"] * 1000
        # The unit error affects one whole day: one file, or its 24 hourly files.
        units_file = f"traffic/date={self.dates[self.units_day].isoformat()}/"
        self.quality.append(
            {
                "id": "duration_in_ms",
                "dataset": "traffic",
                "column": "duration_sec",
                "kind": "unit_error",
                "file": units_file,
                "rows": int(ms_rows.sum()),
                "factor": 1000,
            }
        )
        self.traps.append(
            {
                "id": "units_error",
                "dataset": "traffic",
                "column": "duration_sec",
                "file": units_file,
                "description": "duration_sec is in milliseconds in one day's partition; the affected calls "
                "look anomalous but correlate 100% with the source file.",
                "expected_verdict": "artifact",
            }
        )

        # 6) Exact duplicate rows (0.5%), kept in the same daily file.
        dup = rng.choice(n, int(0.005 * n), replace=False)
        df = pd.concat([df, df.iloc[np.sort(dup)]], ignore_index=True)
        df = df.sort_values(["start_time", "call_id"], kind="stable").reset_index(drop=True)
        self.quality.append(
            {
                "id": "duplicate_rows",
                "dataset": "traffic",
                "kind": "duplicates",
                "rows": int(dup.size),
                "share": round(dup.size / n, 4),
            }
        )
        return df

    # ------------------------------------------------- other datasets ------
    def _segmentation_frame(self) -> pd.DataFrame:
        rng, n = self.rng, self.cfg.customers
        present = np.setdiff1d(np.arange(n), self.roles["missing"])
        seg = self.segment[present]
        age_p = np.array(
            [
                [0.05, 0.22, 0.25, 0.22, 0.16, 0.10],
                [0.02, 0.25, 0.32, 0.25, 0.14, 0.02],
                [0.80, 0.20, 0, 0, 0, 0],
                [0.02, 0.18, 0.30, 0.28, 0.16, 0.06],
            ]
        )
        span = (self.cfg.start_date - date(2012, 1, 1)).days - 30
        block = N_CELLS // len(REGIONS)
        # region_code: integer per region, partially overlapping the cell_id range
        # (a deliberate false join candidate). Odd regions use codes outside it.
        region_code = np.where(
            self.region[present] % 2 == 0,
            CELL_BASE + self.region[present] * block + rng.integers(0, 12, present.size),
            5000 + self.region[present] * 20 + rng.integers(0, 12, present.size),
        )
        df = pd.DataFrame(
            {
                "customer_id": self.national[present],
                "segment": SEGMENTS[seg],
                "tariff_plan": TARIFFS[self.tariff[present]],
                "activation_date": pd.to_datetime(
                    np.datetime64("2012-01-01") + rng.integers(0, span, present.size).astype("timedelta64[D]")
                ).date,
                "region": REGIONS[self.region[present]],
                "credit_class": rng.choice(CREDIT, present.size, p=[0.35, 0.35, 0.2, 0.1]),
                "age_band": AGE_BANDS[_choice_rows(rng, age_p / age_p.sum(axis=1, keepdims=True), seg)],
                "region_code": region_code.astype("int32"),
            }
        )
        # Spelling variants of `segment` (e.g. Business / business / " BUSINESS  ").
        variant_rows = rng.choice(len(df), int(0.08 * len(df)), replace=False)
        kind = rng.integers(0, 2, variant_rows.size)
        values = df.loc[variant_rows, "segment"].to_numpy()
        df.loc[variant_rows, "segment"] = np.where(
            kind == 0,
            np.char.lower(values.astype(str)),
            np.char.add(np.char.add(" ", np.char.upper(values.astype(str))), "  "),
        )
        self.quality.append(
            {
                "id": "segment_spelling",
                "dataset": "cust_segmentation",
                "column": "segment",
                "kind": "inconsistent_categories",
                "rows": int(variant_rows.size),
            }
        )

        # 15 duplicated keys with conflicting attributes (tests fan-out).
        dup_ids = self.national[self.roles["dup_key"]]
        dups = df[df["customer_id"].isin(dup_ids)].copy()
        dups["tariff_plan"] = rng.choice(TARIFFS[:3], len(dups))
        dups["activation_date"] = pd.to_datetime(dups["activation_date"]) + pd.Timedelta(days=200)
        dups["activation_date"] = dups["activation_date"].dt.date
        df = pd.concat([df, dups], ignore_index=True)
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
        self.quality.append(
            {
                "id": "duplicate_keys",
                "dataset": "cust_segmentation",
                "column": "customer_id",
                "kind": "duplicate_keys",
                "keys": sorted(dup_ids.tolist()),
            }
        )
        return df

    def _orders_frame(self) -> pd.DataFrame:
        rng, n = self.rng, self.cfg.customers
        present = np.setdiff1d(np.arange(n), self.roles["missing"])
        reserved = np.concatenate([self.roles[r] for r in ("A", "B", "C", "F")])
        candidates = np.setdiff1d(present, reserved)
        d_cust = np.sort(rng.choice(candidates, 30, replace=False))
        self.d_customers = d_cust
        # Normal customers: per-customer address/device, long-tailed order counts.
        address = np.char.add("ADDR", pd.Series(rng.permutation(n) + 100000).astype(str).to_numpy())
        device = np.char.add("DEV", pd.Series(rng.permutation(n) + 500000).astype(str).to_numpy())
        lam = rng.lognormal(0, 0.9, n)
        lam = lam / lam.mean() * ORDERS_PER_CUSTOMER * self.cfg.scale
        lam[np.setdiff1d(np.arange(n), present)] = 0  # unknown customers have no orders
        lam[d_cust] = 0  # anomaly D customers get only their anomalous orders
        counts = rng.poisson(lam)
        cust = np.repeat(np.arange(n), counts)
        k = cust.size
        product = rng.choice(len(PRODUCTS), k, p=PRODUCT_P)
        qty = 1 + (rng.random(k) < 0.12).astype(int)
        amount = np.round(PRODUCT_PRICE[product] * qty * rng.lognormal(0, 0.15, k), 2)
        normal = pd.DataFrame(
            {
                "customer_id": self.national[cust],
                "order_ts": self.start + rng.integers(0, self.cfg.days * 86400, k).astype("timedelta64[s]"),
                "product": PRODUCTS[product],
                "amount": amount,
                "channel": rng.choice(CHANNELS, k, p=CHANNEL_P),
                "payment_method": rng.choice(PAYMENTS, k, p=PAYMENT_P),
                "shipping_address_id": address[cust],
                "device_id": device[cust],
            }
        )
        # Anomaly D: 30 customers sharing 4 devices and 3 addresses, high amounts.
        shared_dev = np.char.add("DEV", rng.integers(900000, 999999, 4).astype(str))
        shared_addr = np.char.add("ADDR", rng.integers(900000, 999999, 3).astype(str))
        per = rng.integers(3, 9, 30)
        dc = np.repeat(d_cust, per)
        m = dc.size
        high = np.array([5, 4, 8])  # smartphone, tablet, laptop
        prod_d = rng.choice(high, m)
        anomalous = pd.DataFrame(
            {
                "customer_id": self.national[dc],
                "order_ts": self.start + rng.integers(0, self.cfg.days * 86400, m).astype("timedelta64[s]"),
                "product": PRODUCTS[prod_d],
                "amount": np.round(rng.uniform(900, 3000, m), 2),
                "channel": rng.choice(np.array(["web", "app"]), m),
                "payment_method": rng.choice(np.array(["card", "paypal"]), m),
                "shipping_address_id": rng.choice(shared_addr, m),
                "device_id": rng.choice(shared_dev, m),
            }
        )
        self.d_devices, self.d_addresses = shared_dev.tolist(), shared_addr.tolist()
        df = pd.concat([normal, anomalous], ignore_index=True)
        df = df.sort_values("order_ts", kind="stable").reset_index(drop=True)
        df.insert(
            0, "order_id", np.char.add("O", pd.Series(np.arange(1, len(df) + 1)).astype(str).str.zfill(9).to_numpy())
        )
        df["order_ts"] = df["order_ts"].astype("datetime64[us]")
        return df

    # ------------------------------------------------------------- output --
    def _prepare_out(self) -> None:
        for name in self.cfg.outputs:
            target = self.cfg.out / name
            if target.exists():
                if not self.cfg.overwrite:
                    raise FileExistsError(f"{target} exists; pass overwrite=True (--overwrite) to replace it")
                shutil.rmtree(target)
        self.cfg.out.mkdir(parents=True, exist_ok=True)
        self.cfg.truth.mkdir(parents=True, exist_ok=True)

    def _write_traffic(self, df: pd.DataFrame) -> None:
        columns = [
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
        ]
        schema = pa.schema(
            [
                ("call_id", pa.string()),
                ("msisdn", pa.string()),
                ("destination_number", pa.string()),
                ("destination_country", pa.string()),
                ("start_time", pa.timestamp("us")),
                ("duration_sec", pa.int64()),
                ("charge", pa.float64()),
                ("cell_id", pa.int32()),
                ("imei", pa.string()),
                ("call_type", pa.string()),
            ]
        )
        keys = ["_day", "_hour"] if self.cfg.partition == "hour" else ["_day"]
        for key, part in df.groupby(keys, sort=True):  # key is a tuple: (day,) or (day, hour)
            day, hour = key[0], (key[1] if len(key) > 1 else None)
            folder = self.cfg.out / "traffic" / f"date={self.dates[int(day)].isoformat()}"
            if hour is not None:
                folder = folder / f"hour={int(hour):02d}"
            folder.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(part[columns], schema=schema, preserve_index=False)
            pq.write_table(table, folder / "part-0.parquet", compression="zstd")

    def _write_single(self, name: str, df: pd.DataFrame) -> None:
        folder = self.cfg.out / name
        folder.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), folder / "part-0.parquet", compression="zstd")

    def _write_truth(self, traffic: pd.DataFrame) -> None:
        rows: list[dict[str, str]] = []

        def add(dataset: str, column: str, ids: Any, anomaly: str, kind: str) -> None:
            rows.extend(
                {
                    "dataset": dataset,
                    "entity_column": column,
                    "entity_id": str(i),
                    "anomaly_id": anomaly,
                    "anomaly_type": kind,
                }
                for i in ids
            )

        add("traffic", "msisdn", self.msisdn[self.roles["A"]], "A", "point_outlier")
        add("traffic", "msisdn", self.msisdn[self.roles["B"]], "B", "volume_outlier")
        add("traffic", "msisdn", self.msisdn[self.roles["C"]], "C", "coordinated_group")
        add("orders", "customer_id", self.national[self.d_customers], "D", "community")
        add("traffic", "cell_id", [self.e_cell], "E", "temporal")
        add("traffic", "msisdn", self.msisdn[self.roles["F"]], "F", "contextual")
        rows = [r for r in rows if r["dataset"] in self.cfg.telecom]
        self._merge_anomalies(pd.DataFrame(rows, columns=TRUTH_COLUMNS), set(TELECOM_DATASETS))

        details = {
            "A": {
                "description": "Night bursts of long calls to premium-rate countries on 2-3 days",
                "countries": COUNTRIES[EXPENSIVE].tolist(),
            },
            "B": {"description": "Hundreds of 1-3 s outgoing calls to distinct numbers"},
            "C": {
                "description": "40 SIMs sharing 3 IMEIs and 2 cells, outgoing only",
                "imeis": self.c_imeis,
                "cells": self.c_cells,
            },
            "D": {
                "description": "30 customers sharing 4 device_id and 3 shipping_address_id, high amounts",
                "device_ids": self.d_devices,
                "shipping_address_ids": self.d_addresses,
            },
            "E": {
                "description": "Volume spike in one cell for 6 hours",
                "cell_id": self.e_cell,
                "date": self.dates[self.e_day].isoformat(),
                "start_hour": self.e_hour,
                "hours": 6,
            },
            "F": {"description": "Consumer/basic customers with a business profile (only visible per segment)"},
        }
        self._merge_details(details, keep=("topup",))
        telecom = set(self.cfg.telecom)
        merge_json_list(
            self.cfg.truth / "quality_issues.json",
            [q for q in self.quality if q["dataset"] in telecom],
            set(TELECOM_DATASETS),
        )
        merge_json_list(
            self.cfg.truth / "traps.json", [t for t in self.traps if t["dataset"] in telecom], set(TELECOM_DATASETS)
        )

        expected = {
            "datasets": [{"name": "traffic", "role": "primary"}, {"name": "cust_segmentation", "role": "enrichment"}],
            "joins": [
                {
                    "left": "traffic",
                    "right": "cust_segmentation",
                    "keys": [["msisdn", "customer_id"]],
                    "key_normalization": ["strip_prefix_plus351"],
                    "type": "left",
                    "cardinality": "N:1",
                    "expected_key_coverage": round(1 - self.roles["missing"].size / self.cfg.customers, 4),
                    "duplicate_keys_right": int(self.roles["dup_key"].size),
                }
            ],
            "rejected_join_candidates": [
                {
                    "left": "traffic.cell_id",
                    "right": "cust_segmentation.region_code",
                    "reason": "partial value overlap only, different meaning",
                }
            ],
            "grain": {
                "dataset": "traffic",
                "columns": ["call_id"],
                "note": "call_id is unique except for the planted duplicate rows",
            },
            "entity": {"column": "msisdn"},
            "time_column": "start_time",
            "orders": {
                "join_key": [["customer_id", "customer_id"]],
                "entity": "customer_id",
                "time_column": "order_ts",
            },
        }
        (self.cfg.truth / "expected_spec.yaml").write_text(yaml.safe_dump(expected, sort_keys=False), encoding="utf-8")

    def _write_readmes(self) -> None:
        for name, text in DATASET_READMES.items():
            if name in self.cfg.telecom:
                (self.cfg.out / name / "README.md").write_text(text, encoding="utf-8")

    # ------------------------------------------------------ ground truth --
    def _merge_anomalies(self, rows: pd.DataFrame, replaced: set[str]) -> None:
        """Replace the ground-truth rows of ``replaced`` datasets, keeping the others."""
        path = self.cfg.truth / "anomalies.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            rows = pd.concat([old[~old["dataset"].isin(replaced)], rows], ignore_index=True)
        rows.to_parquet(path, index=False)

    def _merge_details(self, details: dict[str, Any], keep: tuple[str, ...]) -> None:
        """Write anomaly_details.json; ``keep`` are the keys of the other scenario."""
        path = self.cfg.truth / "anomaly_details.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        merged = {k: v for k, v in old.items() if k in keep} | details
        path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    def _topup(self, say: Callable[[str], None]) -> dict[str, Any]:
        cfg = self.cfg
        population = Population(
            msisdn=self.msisdn, segment=self.segment, tariff=self.tariff, region=self.region, regions=REGIONS
        )
        gen = TopupGenerator(
            population,
            TopupConfig(
                days=cfg.topup_days,
                start_date=cfg.start_date,
                scale=cfg.scale,
                opaque_codes=cfg.opaque_codes,
                with_readme=cfg.with_readme,
            ),
            seed=cfg.seed,
        )
        summary = gen.build(say)
        say(f"writing topup ({summary['topup_rows']:,} movements, {cfg.topup_days} daily files) and reference tables")
        gen.write(cfg.out, cfg.truth)
        self._merge_anomalies(pd.DataFrame(gen.anomaly_rows, columns=TRUTH_COLUMNS), {"topup"})
        self._merge_details({"topup": gen.details}, keep=("A", "B", "C", "D", "E", "F"))
        merge_json_list(cfg.truth / "quality_issues.json", gen.quality, {"topup"})
        merge_json_list(cfg.truth / "traps.json", gen.traps, {"topup"})
        return summary

    # ---------------------------------------------------------------- run --
    def run(self, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Generate everything and return a short summary.

        Args:
            progress: Optional callback receiving a short message at each step.
        """
        say = progress or (lambda message: None)
        started = datetime.now()
        self._prepare_out()
        say(f"customers: {self.cfg.customers:,} with segments, cells, devices and contacts")
        self._build_customers()
        self._assign_roles()
        summary: dict[str, Any] = {}
        if self.cfg.telecom:
            summary |= self._telecom(say)
        if self.cfg.wants_topup:
            summary |= self._topup(say)
        return summary | {
            "datasets": list(self.cfg.outputs),
            "seconds": round((datetime.now() - started).total_seconds(), 1),
            "out": str(self.cfg.out),
            "truth": str(self.cfg.truth),
        }

    def _telecom(self, say: Callable[[str], None]) -> dict[str, Any]:
        """traffic, cust_segmentation and orders (always built together: one random stream)."""
        say(f"traffic: normal calls for {self.cfg.days} days + anomalies A, B, C, E + quality problems")
        traffic = self._traffic_frame()
        say(f"traffic: {len(traffic):,} calls generated")
        say("cust_segmentation: master data, spelling variants, duplicated keys (anomaly F planted)")
        segmentation = self._segmentation_frame()
        say("orders: normal orders + shared devices/addresses (anomaly D)")
        orders = self._orders_frame()
        layout = "date/hour" if self.cfg.partition == "hour" else "date"
        wanted = self.cfg.telecom
        summary: dict[str, Any] = {}
        if "traffic" in wanted:
            say(f"writing traffic ({self.cfg.days} days, partitioned by {layout})")
            self._write_traffic(traffic)
            summary |= {
                "traffic_rows": len(traffic),
                "traffic_files": len(list((self.cfg.out / "traffic").rglob("*.parquet"))),
                "traffic_partitioning": "date/hour" if self.cfg.partition == "hour" else "date",
            }
        if "cust_segmentation" in wanted:
            self._write_single("cust_segmentation", segmentation)
            summary["cust_segmentation_rows"] = len(segmentation)
        if "orders" in wanted:
            self._write_single("orders", orders)
            summary["orders_rows"] = len(orders)
        say(f"writing ground truth to {self.cfg.truth}")
        self._write_truth(traffic)
        if self.cfg.with_readme:
            self._write_readmes()
        sizes = {"A": 25, "B": 15, "C": 40, "D": 30, "E": 1, "F": 60}
        where = {"A": "traffic", "B": "traffic", "C": "traffic", "D": "orders", "E": "traffic", "F": "traffic"}
        return summary | {"anomalies": {k: v for k, v in sizes.items() if where[k] in wanted}}


def generate(cfg: GeneratorConfig, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Convenience entry point used by the CLI and the script."""
    return DummyDataGenerator(cfg).run(progress)
