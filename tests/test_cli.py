# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""CLI smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from jellybean.cli import app

runner = CliRunner()


def test_generate_data_and_list_datasets(tmp_path: Path, monkeypatch) -> None:
    data, truth = tmp_path / "data", tmp_path / "truth"
    args = [
        "generate-data",
        "--out",
        str(data),
        "--truth",
        str(truth),
        "--customers",
        "1000",
        "--days",
        "7",
        "--scale",
        "0.2",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert (data / "traffic").is_dir() and (truth / "anomalies.parquet").exists()
    assert "] traffic:" in result.output and "] writing ground truth" in result.output  # step-by-step progress

    # Second run without --overwrite fails cleanly; with it, succeeds.
    assert runner.invoke(app, args).exit_code == 1
    assert runner.invoke(app, [*args, "--overwrite"]).exit_code == 0

    monkeypatch.setenv("JELLYBEAN_DATA_ROOT", str(data))
    from jellybean.settings import reset_settings

    reset_settings()
    listed = runner.invoke(app, ["datasets"])
    assert listed.exit_code == 0, listed.output
    names = {d["name"] for d in json.loads(listed.output)["summary"]["datasets"]}
    # The CLI default generates every scenario, the top-up ledger included.
    assert names == {"traffic", "cust_segmentation", "orders", "topup", "promotions", "dealers", "voucher_batches"}


def test_generate_selected_datasets(tmp_path: Path) -> None:
    data, truth = tmp_path / "data", tmp_path / "truth"
    base = ["generate-data", "--out", str(data), "--truth", str(truth), "--customers", "1000", "--scale", "0.2"]
    result = runner.invoke(app, [*base, "--datasets", "topup", "--topup-days", "60", "--opaque-codes"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in data.iterdir()) == ["dealers", "promotions", "topup", "voucher_batches"]
    assert len(list((data / "topup").glob("date=*"))) == 60
    assert (truth / "expected_spec_topup.yaml").exists() and not (truth / "expected_spec.yaml").exists()
    bad = runner.invoke(app, [*base, "--datasets", "traffic,weather"])
    assert bad.exit_code == 2 and "weather" in bad.output
    assert runner.invoke(app, [*base, "--type", "telemetry", "--datasets", "topup"]).exit_code == 2


def test_generate_telemetry_data(tmp_path: Path) -> None:
    data, truth = tmp_path / "data", tmp_path / "truth"
    args = ["generate-data", "--type", "telemetry", "--out", str(data), "--truth", str(truth)]
    args += ["--machines", "200", "--days", "8", "--with-readme"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(list((data / "telemetry").glob("date=*/part-0.parquet"))) == 8
    assert (data / "machines" / "README.md").exists() and (truth / "telemetry_truth.json").exists()
    assert "] writing ground truth" in result.output
    summary = json.loads(result.output[result.output.index("{") :])
    assert summary["type"] == "telemetry" and summary["anomalies"]["S1"] == 12
    # Existing datasets are protected, as for telecom.
    assert runner.invoke(app, args).exit_code == 1
    assert runner.invoke(app, [*args, "--overwrite"]).exit_code == 0


def test_generate_data_rejects_bad_type_and_telecom_options(tmp_path: Path) -> None:
    base = ["generate-data", "--out", str(tmp_path / "d"), "--truth", str(tmp_path / "t")]
    assert runner.invoke(app, [*base, "--type", "weather"]).exit_code == 2
    bad = runner.invoke(app, [*base, "--type", "telemetry", "--scale", "0.5"])
    assert bad.exit_code == 2 and "telecom only" in bad.output
