"""Industrial machine telemetry: the "unknown domain" dataset.

Generated with ``jellybean generate-data --type telemetry`` (or
``scripts/generate_unknown_domain.py``). It has nothing in common with the
telecom simulator, and no tool, skill or threshold knows anything about
machines, sensors or firmware: it checks that the system is domain-agnostic.

Datasets (under ``out``):

* ``telemetry``  one hourly reading per machine, partitioned by ``date=``;
  columns: reading_id, machine_id ("MCH-00123"), ts, temp_c, vibration_mm_s,
  pressure_bar, power_kw, status, fw_version, error_count.
* ``machines``   reference table keyed by asset_id ("00123": no "MCH-" prefix);
  columns: asset_id, model, site_code, install_year, rated_kw, criticality.

Planted (ground truth in ``truth/telemetry_truth.json``, never under ``out``):

* S1  12 machines whose bearing degrades: vibration up to x4, temperature +15 in the last 5 days.
* S2  10 machines with the temperature sensor stuck at 21.5 for the last 7 days.
* S3  one site with a power surge (x2.2) for 4 hours on day 9.
* S4  15 machines of model A consuming like model C (x3.5): visible only within the model.
* S5  8 machines on a dev firmware with many errors.
* decoy  20 machines with 2 days of legitimate maintenance (power 0, status MAINT).
* quality: 0.3% duplicated rows, 1% null vibration, one day with pressure in kPa
  (x100), 1% status spelling variants.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATASETS = ("telemetry", "machines")
TRUTH_FILE = "telemetry_truth.json"

# Column meanings only (no hints about planted anomalies), as for the telecom READMEs.
TELEMETRY_READMES: dict[str, str] = {
    "telemetry": """# telemetry

Sensor readings of industrial machines, one row per machine and hour, partitioned by day.

| column | meaning |
| --- | --- |
| reading_id | unique identifier of the reading |
| machine_id | machine that sent the reading (MCH-<asset number>) |
| ts | time of the reading |
| temp_c | motor temperature in degrees Celsius |
| vibration_mm_s | vibration velocity in mm/s |
| pressure_bar | hydraulic pressure in bar |
| power_kw | power drawn in kW |
| status | RUN, IDLE or MAINT (planned maintenance) |
| fw_version | firmware version of the controller |
| error_count | controller errors during the hour |
""",
    "machines": """# machines

Asset register, one row per machine.

| column | meaning |
| --- | --- |
| asset_id | asset number (machine_id in telemetry without the MCH- prefix) |
| model | machine model (A, B, C) |
| site_code | plant where the machine is installed |
| install_year | year of installation |
| rated_kw | nominal power in kW |
| criticality | business criticality (low, medium, high) |
| register_synced_at | when the register entry was last synchronised (UTC, timezone-aware) |
""",
}

START = np.datetime64("2026-09-01T00:00:00")


def generate_telemetry(
    out: Path,
    truth: Path,
    seed: int = 7,
    machines: int = 1500,
    days: int = 14,
    overwrite: bool = False,
    with_readme: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Write the datasets under ``out`` and the ground truth under ``truth``; return the truth.

    Raises:
        ValueError: ``truth`` inside ``out``, or too few machines/days.
        FileExistsError: A dataset exists and ``overwrite`` is false.
    """
    say = progress or (lambda _message: None)
    if truth.resolve() == out.resolve() or truth.resolve().is_relative_to(out.resolve()):
        raise ValueError("truth folder must be outside the data folder (agents must not see it)")
    if machines < 100 or days < 8:
        raise ValueError("telemetry needs at least 100 machines and 8 days (the planted anomalies need them)")
    for name in DATASETS:
        if (out / name).exists() and not overwrite:
            raise FileExistsError(f"{out / name} exists; pass overwrite=True (--overwrite) to replace it")
    rng = np.random.default_rng(seed)
    n = machines

    # ------------------------------------------------------------ machines (reference)
    ids = np.arange(1, n + 1)
    model = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    rated = np.select([model == "A", model == "B"], [15.0, 30.0], 55.0) * rng.normal(1, 0.05, n)
    site = rng.choice([f"S{i:02d}" for i in range(1, 13)], n)
    machines_df = pd.DataFrame(
        {
            "asset_id": [f"{i:05d}" for i in ids],
            "model": model,
            "site_code": site,
            "install_year": rng.integers(2012, 2026, n),
            "rated_kw": rated.round(1),
            "criticality": rng.choice(["low", "medium", "high"], n, p=[0.5, 0.35, 0.15]),
        }
    )
    # Timezone-aware timestamp (TIMESTAMP WITH TIME ZONE in Parquet): keeps that type
    # covered end to end. Derived from the asset number, so it draws nothing from the RNG.
    last_sync = START.astype("datetime64[us]") - (ids % 97).astype("timedelta64[h]") - np.timedelta64(1, "h")
    machines_df["register_synced_at"] = pd.Series(last_sync).dt.tz_localize("UTC")

    # ------------------------------------------------------------ roles (disjoint)
    perm = rng.permutation(n)
    roles = {"S1": perm[:12], "S2": perm[12:22], "S5": perm[22:30], "decoy": perm[30:50]}
    roles["S4"] = rng.choice(np.setdiff1d(np.where(model == "A")[0], perm[:50]), 15, replace=False)
    s3_site, s3_day = "S07", 8
    fw = rng.choice(["4.2.1", "4.3.0", "4.1.9"], n, p=[0.6, 0.35, 0.05]).astype(object)
    fw[roles["S5"]] = "4.4.0-dev"

    # ------------------------------------------------------------ hourly readings
    say(f"telemetry: {n:,} machines x {days * 24} hourly readings")
    hours = days * 24
    m = np.repeat(np.arange(n), hours)
    h = np.tile(np.arange(hours), n)
    ts = START + (h * 3600 + rng.integers(0, 3600, len(h))).astype("timedelta64[s]")
    hod, dow = h % 24, (h // 24 + 1) % 7  # 2026-09-01 is a Tuesday
    load = np.where((hod >= 6) & (hod < 22), 0.8, 0.35) * np.where(dow >= 5, 0.5, 1.0) * rng.normal(1, 0.08, len(h))
    load = np.clip(load, 0.05, 1.2)
    power = rated[m] * load
    temp = np.select([model[m] == "A", model[m] == "B"], [55.0, 60.0], 65.0) + 10 * load + rng.normal(0, 1.5, len(h))
    vib = rng.lognormal(np.log(2.0), 0.25, len(h)) * (1 + 0.3 * load)
    pressure = rng.normal(6.0, 0.3, len(h))
    status = np.where(load < 0.25, "IDLE", "RUN").astype(object)
    errors = rng.poisson(0.05, len(h))

    s1 = np.isin(m, roles["S1"]) & (h >= (days - 5) * 24)
    progress = (h[s1] - (days - 5) * 24) / (5 * 24)
    vib[s1] *= 1 + 3 * progress
    temp[s1] += 15 * progress
    s2 = np.isin(m, roles["S2"]) & (h >= (days - 7) * 24)
    temp[s2] = 21.5
    s3 = (site[m] == s3_site) & (h >= s3_day * 24 + 10) & (h < s3_day * 24 + 14)
    power[s3] *= 2.2
    power[np.isin(m, roles["S4"])] *= 3.5
    s5 = np.isin(m, roles["S5"])
    errors[s5] = rng.poisson(2.0, s5.sum())
    decoy = np.isin(m, roles["decoy"]) & (h >= 4 * 24) & (h < 6 * 24)
    power[decoy], status[decoy] = 0.0, "MAINT"

    df = pd.DataFrame(
        {
            "reading_id": [f"R{i:08d}" for i in range(len(h))],
            "machine_id": [f"MCH-{i:05d}" for i in ids[m]],
            "ts": ts.astype("datetime64[us]"),
            "temp_c": temp.round(2),
            "vibration_mm_s": vib.round(3),
            "pressure_bar": pressure.round(3),
            "power_kw": power.round(2),
            "status": status,
            "fw_version": fw[m],
            "error_count": errors,
        }
    )
    day = h // 24

    # ------------------------------------------------------------ quality problems
    say("planting quality problems (nulls, kPa day, status variants, duplicates)")
    vib_null = rng.choice(len(df), int(0.01 * len(df)), replace=False)
    df.loc[vib_null, "vibration_mm_s"] = np.nan
    kpa_day = 5
    df.loc[day == kpa_day, "pressure_bar"] *= 100
    variants = rng.choice(len(df), int(0.01 * len(df)), replace=False)
    df.loc[variants, "status"] = df.loc[variants, "status"].str.lower() + " "
    dups = rng.choice(len(df), int(0.003 * len(df)), replace=False)
    df["_day"] = day
    df = pd.concat([df, df.iloc[np.sort(dups)]], ignore_index=True)

    # ------------------------------------------------------------ write
    say(f"writing telemetry ({len(df):,} rows, {days} daily files) and machines to {out}")
    for name in DATASETS:
        if (out / name).exists():
            shutil.rmtree(out / name)
    for d, part in df.groupby("_day"):
        date = str(START.astype("datetime64[D]") + np.timedelta64(int(d), "D"))
        folder = out / "telemetry" / f"date={date}"
        folder.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(part.drop(columns="_day"), preserve_index=False), folder / "part-0.parquet")
    (out / "machines").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(machines_df, preserve_index=False), out / "machines" / "part-0.parquet")
    if with_readme:
        for name, text in TELEMETRY_READMES.items():
            (out / name / "README.md").write_text(text, encoding="utf-8")

    date_of = lambda d: str(START.astype("datetime64[D]") + np.timedelta64(d, "D"))  # noqa: E731
    result = {
        "entity": "machine_id",
        "anomalies": {k: [f"MCH-{ids[i]:05d}" for i in v] for k, v in roles.items()},
        "S3": {"site_code": s3_site, "date": date_of(s3_day), "hours": [10, 14]},
        "join": {"left": ["telemetry", "machine_id"], "right": ["machines", "asset_id"], "prefix": "MCH-"},
        "grain": ["reading_id"],
        "time_column": "ts",
        "quality": {
            "kpa_day": date_of(kpa_day),
            "vibration_nulls": len(vib_null),
            "status_variants": len(variants),
            "duplicates": len(dups),
        },
        "rows": len(df),
    }
    say(f"writing ground truth to {truth / TRUTH_FILE}")
    truth.mkdir(parents=True, exist_ok=True)
    (truth / TRUTH_FILE).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
