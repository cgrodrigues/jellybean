"""Phase 2 verification: profiler tools find the right spec on the dummy data, without an LLM.

Expected values come from the generator's ground truth (expected_spec.yaml,
quality_issues.json), never from hard-coded guesses.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from jellybean.runtime.context import RunContext, use_run
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills.profiler import tools as pt
from tests.conftest import dummy_settings


@pytest.fixture(scope="module")
def dummy(dummy_base: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[RunContext, Path]]:
    settings = dummy_settings(dummy_base, tmp_path_factory.mktemp("ws"))
    ctx = RunContext.create(settings, seed=42, run_id="dummy_run")
    with use_run(ctx):
        yield ctx, dummy_base / "truth"


def test_join_is_discovered_with_prefix_normalization(dummy) -> None:
    _ctx, truth = dummy
    expected = yaml.safe_load((truth / "expected_spec.yaml").read_text())["joins"][0]
    cand = pt.discover_join_candidates("traffic", "cust_segmentation")
    best = cand.summary["best"]
    assert [[best["left_column"], best["right_column"]]] == expected["keys"]
    assert [best["normalization"]] == expected["key_normalization"]
    assert cand.summary["score_margin"] >= 0.15
    # The false candidate region_code is ranked far below.
    false = [c for c in cand.summary["candidates"] if c["right_column"] == "region_code"]
    assert all(c["score"] <= best["score"] - 0.15 for c in false)

    joined = pt.test_join(
        "traffic",
        "cust_segmentation",
        expected["keys"],
        expected["key_normalization"],
        candidates_artifact=cand.artifact_id,
    )
    metrics, assessment = joined.summary["metrics"], joined.summary["assessment"]
    assert metrics["key_coverage"] == pytest.approx(expected["expected_key_coverage"], abs=0.001)
    assert metrics["coverage"] >= 0.95
    assert metrics["duplicate_keys"] == expected["duplicate_keys_right"]  # flagged before materialisation
    assert metrics["cardinality"] == expected["cardinality"]
    assert assessment["auto_confirm"] is True
    assert {i["kind"] for i in joined.summary["issues"]} == {"duplicate_keys", "unmatched_rows"}


def test_region_code_join_is_rejected(dummy) -> None:
    result = pt.test_join("traffic", "cust_segmentation", [["cell_id", "region_code"]])
    assert result.summary["assessment"]["auto_confirm"] is False
    assert result.summary["metrics"]["fanout"] > 1.01


def test_grain_entity_and_time_column(dummy) -> None:
    _, truth = dummy
    expected = yaml.safe_load((truth / "expected_spec.yaml").read_text())
    grain = pt.infer_grain("traffic")
    assert grain.summary["recommended"] == expected["grain"]["columns"]
    assert grain.summary["exact_duplicate_rows"] > 0
    assert pt.suggest_entities("traffic").summary["recommended"] == expected["entity"]["column"]
    types = pt.infer_semantic_types("traffic").summary
    assert expected["time_column"] in types["time_column_candidates"]
    seg_types = {k: v[0] for k, v in pt.infer_semantic_types("cust_segmentation").summary["columns"].items()}
    assert seg_types["customer_id"] == "identifier" and seg_types["region_code"] == "category"


def test_planted_quality_problems_are_detected(dummy) -> None:
    ctx, truth = dummy
    planted = {i["id"]: i for i in json.loads((truth / "quality_issues.json").read_text())}
    profile = pt.profile_table("traffic")
    seg_profile = pt.profile_table("cust_segmentation")
    temporal = pt.check_temporal_coverage("traffic", "start_time")
    cand = pt.discover_join_candidates("traffic", "cust_segmentation")
    joined = pt.test_join(
        "traffic",
        "cust_segmentation",
        [["msisdn", "customer_id"]],
        ["strip_prefix_plus351"],
        candidates_artifact=cand.artifact_id,
    )
    merged = pt.compile_quality_issues(
        [profile.artifact_id, seg_profile.artifact_id, temporal.artifact_id, joined.artifact_id]
    )
    issues = ArtifactStore(ctx).load_json(merged.artifact_id)["issues"]
    found = {(i["column"], i["kind"]) for i in issues}

    assert (None, "duplicate_rows") in found
    assert ("cell_id", "nulls") in found and ("imei", "nulls") in found
    assert ("duration_sec", "negative_values") in found
    assert ("segment", "inconsistent_categories") in found
    assert ("customer_id", "duplicate_keys") in found
    # Units error: the flagged file is exactly the planted one.
    shift = next(i for i in issues if i["kind"] == "scale_shift_by_file")
    assert planted["duration_in_ms"]["file"].endswith(shift["detail"]["file"])
    # Time-zone shift: the flagged day is exactly the planted one, with no false alarms.
    tz_date = planted["timezone_shift"]["file"].split("date=")[1].split("/")[0]
    tz = [i for i in issues if i["kind"] == "possible_timezone_shift"]
    assert [i["detail"]["date"] for i in tz] == [tz_date]
    assert not [i for i in issues if i["kind"] == "hour_profile_shift"]
