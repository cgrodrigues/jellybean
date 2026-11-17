# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for preparer tools over the small fixtures (see conftest.py)."""

from __future__ import annotations

import duckdb
import pytest
import yaml

from jellybean.runtime.context import RunContext
from jellybean.skills._shared import data_tools as dt
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills.preparer import tools as pr
from jellybean.skills.profiler import tools as pt


def _rows(ctx: RunContext, artifact_id: str, sql: str = "SELECT count(*) FROM t") -> list:
    path = ArtifactStore(ctx).path_of(artifact_id)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{path}')")
    return con.execute(sql).fetchall()


@pytest.fixture
def messy(run_ctx: RunContext) -> str:
    """A small table with duplicates, nulls, variants, negatives and a unit error."""
    result = dt.run_sql(
        """SELECT * FROM (VALUES
             ('a', 'Business', 10.0, 1), ('a', 'Business', 10.0, 1), ('b', ' BUSINESS  ', NULL, 2),
             ('c', 'business', -5.0, 3), ('d', 'Consumer', 3000.0, 4), ('e', 'Consumer', 7.0, 5))
           t(key, segment, amount, n)""",
        save_as="messy",
    )
    return result.artifact_id


@pytest.mark.parametrize(
    ("operation", "expected_rows", "affected"),
    [
        ({"op": "dedupe"}, 5, 1),
        ({"op": "dedupe", "keys": ["segment"], "keep": "latest", "order_by": "n"}, 4, 2),
        ({"op": "drop_rows", "where": "amount < 0"}, 5, 1),
        ({"op": "impute", "column": "amount", "method": "median"}, 6, 1),
        ({"op": "normalize", "column": "segment", "method": "canonical"}, 6, 2),
        ({"op": "winsorize", "column": "amount", "lower_q": 0.0, "upper_q": 0.8}, 6, 1),
        ({"op": "fix_units", "column": "amount", "factor": 1000, "where": "amount > 1000"}, 6, 1),
        ({"op": "cast", "column": "n", "to": "VARCHAR"}, 6, 0),
    ],
)
def test_apply_transform_operations(
    run_ctx: RunContext, messy: str, operation: dict, expected_rows: int, affected: int
) -> None:
    result = pr.apply_transform(messy, [operation])
    assert result.status == "ok", result.summary
    assert result.summary["operations"][0]["affected_rows"] == affected
    assert _rows(run_ctx, result.artifact_id)[0][0] == expected_rows
    assert ArtifactStore(run_ctx).get(result.artifact_id).parents == [messy]


def test_canonical_normalisation_and_unit_fix_values(run_ctx: RunContext, messy: str) -> None:
    result = pr.apply_transform(
        messy,
        [
            {"op": "normalize", "column": "segment"},
            {"op": "fix_units", "column": "amount", "factor": 1000, "where": "amount > 1000"},
        ],
    )
    segments = {r[0] for r in _rows(run_ctx, result.artifact_id, "SELECT DISTINCT segment FROM t")}
    assert segments == {"Business", "Consumer"}
    assert _rows(run_ctx, result.artifact_id, "SELECT max(amount) FROM t")[0][0] == 10.0


@pytest.mark.parametrize(
    "operations",
    [
        [{"op": "explode"}],
        [{"op": "drop_rows", "where": "1=1; DROP TABLE x"}],
        [{"op": "impute", "column": "nope"}],
        [{"op": "cast", "column": "n", "to": "BLOB"}],
        [{"op": "impute", "column": "amount", "method": "constant"}],
        [{"op": "fix_units", "column": "amount", "factor": 10, "file": "x.parquet"}],  # artifact: no provenance
    ],
)
def test_apply_transform_errors(run_ctx: RunContext, messy: str, operations: list) -> None:
    assert pr.apply_transform(messy, operations).status == "error"


def test_fix_units_by_file_uses_internal_provenance(run_ctx: RunContext) -> None:
    result = pr.apply_transform(
        "events", [{"op": "fix_units", "column": "amount", "factor": 10, "file": "date=2026-10-02/part-0.parquet"}]
    )
    assert result.status == "ok"
    assert result.summary["operations"][0]["affected_rows"] == 500
    columns = [r[0] for r in _rows(run_ctx, result.artifact_id, "DESCRIBE t")]
    assert "__file" not in columns


def _confirmed_spec(ctx: RunContext) -> None:
    spec = {
        "run_id": ctx.run_id,
        "instruction": "x",
        "seed": 7,
        "datasets": [
            {"name": "events", "path": "p", "role": "primary"},
            {"name": "customers", "path": "p", "role": "enrichment"},
        ],
        "joins": [
            {
                "left": "events",
                "right": "customers",
                "keys": [["phone", "customer_id"]],
                "key_normalization": ["strip_prefix_plus351"],
                "status": "confirmed",
            }
        ],
        "entity": {"column": "phone"},
        "grain": {"dataset": "events", "columns": ["event_id"]},
        "columns": {"segment": {"dataset": "customers", "semantic_type": "category"}},
    }
    ctx.spec_path.write_text(yaml.safe_dump(spec))


def test_materialize_join_flags_and_keeps_rows(run_ctx: RunContext) -> None:
    _confirmed_spec(run_ctx)
    result = pr.materialize_join(0)
    assert result.status == "ok", result.summary
    assert result.summary["rows"] == 1000 and result.summary["unmatched_rows"] == 0
    assert set(result.summary["added_columns"]) == {"contact", "segment"}
    columns = [r[0] for r in _rows(run_ctx, result.artifact_id, "DESCRIBE t")]
    assert "matched_customers" in columns and "customer_id" not in columns
    from jellybean.spec import load_spec

    assert load_spec(run_ctx.spec_path).enriched_artifact == result.artifact_id


def test_materialize_join_refuses_duplicates_and_unconfirmed(run_ctx: RunContext) -> None:
    _confirmed_spec(run_ctx)
    dup = dt.run_sql("SELECT * FROM customers UNION ALL SELECT * FROM customers LIMIT 60", save_as="dup_customers")
    spec = yaml.safe_load(run_ctx.spec_path.read_text())
    spec["prepared"] = {"customers": dup.artifact_id}
    run_ctx.spec_path.write_text(yaml.safe_dump(spec))
    assert "duplicated keys" in pr.materialize_join(0).summary["error"]
    spec["joins"][0]["status"] = "proposed"
    run_ctx.spec_path.write_text(yaml.safe_dump(spec))
    assert "not confirmed" in pr.materialize_join(0).summary["error"]
    assert pr.materialize_join(5).status == "error"


def test_unmatched_rows_are_kept(run_ctx: RunContext) -> None:
    _confirmed_spec(run_ctx)
    few = dt.run_sql("SELECT * FROM customers WHERE customer_id < '910000025'", save_as="few")
    spec = yaml.safe_load(run_ctx.spec_path.read_text())
    spec["prepared"] = {"customers": few.artifact_id}
    run_ctx.spec_path.write_text(yaml.safe_dump(spec))
    result = pr.materialize_join(0)
    assert result.summary["rows"] == 1000 and 0 < result.summary["unmatched_rows"] < 1000


def test_diff_and_validate(run_ctx: RunContext, messy: str) -> None:
    clean = pr.apply_transform(messy, [{"op": "dedupe"}, {"op": "drop_rows", "where": "amount < 0"}])
    diff = pr.diff_versions(messy, clean.artifact_id)
    assert diff.summary["row_change"] == -2
    rules = [
        {"column": "key", "check": "unique"},
        {"column": "amount", "check": "min", "value": 0},
        {"column": "segment", "check": "in_set", "value": ["Business", "Consumer"]},
        {"check": "no_duplicate_rows"},
        {"column": "ghost", "check": "not_null"},
    ]
    result = pr.validate_constraints(clean.artifact_id, rules)
    by = {(r.get("column"), r["check"]): r for r in result.summary["rules"]}
    assert by[("key", "unique")]["passed"] and by[("amount", "min")]["passed"]
    assert not by[("segment", "in_set")]["passed"]  # spelling variants still present
    assert by[(None, "no_duplicate_rows")]["passed"] and not by[("ghost", "not_null")]["passed"]


def test_propose_plan_policies(run_ctx: RunContext) -> None:
    _confirmed_spec(run_ctx)
    issues = {
        "issues": [
            {
                "id": "Q1",
                "dataset": "events",
                "column": None,
                "kind": "duplicate_rows",
                "severity": "low",
                "detail": {"rows": 3},
            },
            {
                "id": "Q2",
                "dataset": "events",
                "column": "phone",
                "kind": "nulls",
                "severity": "medium",
                "detail": {"share": 0.02},
            },
            {
                "id": "Q3",
                "dataset": "events",
                "column": "amount",
                "kind": "scale_shift_by_file",
                "severity": "high",
                "detail": {"file": "date=2026-10-02/part-0.parquet", "factor": 986.1},
            },
            {
                "id": "Q4",
                "dataset": "events",
                "column": "amount",
                "kind": "scale_shift_by_file",
                "severity": "high",
                "detail": {"file": "x", "factor": 37.0},
            },
            {
                "id": "Q5",
                "dataset": "customers",
                "column": "customer_id",
                "kind": "duplicate_keys",
                "severity": "high",
                "detail": {"keys": 2},
            },
            {
                "id": "Q6",
                "dataset": "events",
                "column": "ts",
                "kind": "possible_timezone_shift",
                "severity": "high",
                "detail": {},
            },
        ]
    }
    issues_id = ArtifactStore(run_ctx).save_json("quality_issues", issues).id
    plan = pr.propose_cleaning_plan(issues_id)
    ops = {o["id"]: o for o in plan.summary["ops"]}
    assert [o["op"] for o in ops.values()] == ["dedupe", "fix_units", "dedupe"]
    fix = next(o for o in ops.values() if o["op"] == "fix_units")
    assert fix["explicit_approval"] is True
    reasons = {n["issue_id"]: n["reason"] for n in plan.summary["not_addressed"]}
    assert "artificial group" in reasons["Q2"] and "power of ten" in reasons["Q4"] and "Q6" in reasons
    full = ArtifactStore(run_ctx).load_json(plan.artifact_id)
    assert next(o for o in full["ops"] if o["operation"]["op"] == "fix_units")["operation"]["factor"] == 1000
    assert pr.propose_cleaning_plan("nope_v1").status == "error"


def test_apply_cleaning_plan_requires_approval(run_ctx: RunContext) -> None:
    _confirmed_spec(run_ctx)
    assert "not approved" in pr.apply_cleaning_plan().summary["error"]


def test_profiler_issues_to_plan_on_fixture(run_ctx: RunContext) -> None:
    _confirmed_spec(run_ctx)
    profile = pt.profile_table("events")
    issues = pt.compile_quality_issues([profile.artifact_id])
    assert pr.propose_cleaning_plan(issues.artifact_id).status == "ok"


def test_plan_ignores_issues_of_joins_not_in_spec(run_ctx: RunContext) -> None:
    """Regression: a rejected join candidate's duplicate keys must not become a dedupe."""
    _confirmed_spec(run_ctx)
    good = pt.test_join("events", "customers", [["phone", "customer_id"]], ["strip_prefix_plus351"])
    bad = pt.test_join("events", "customers", [["event_id", "segment"]])  # a nonsense candidate
    issues = pt.compile_quality_issues([good.artifact_id, bad.artifact_id])
    plan = pr.propose_cleaning_plan(issues.artifact_id)
    ops = plan.summary["ops"]
    assert not any(o["column"] == ["segment"] for o in ops), ops
    reasons = " ".join(n["reason"] for n in plan.summary["not_addressed"])
    assert "not in the spec" in reasons
    assert all("issue_ids" in o and o["justification"] for o in ops)


def test_skip_accepts_operation_ids(run_ctx: RunContext) -> None:
    _confirmed_spec(run_ctx)
    issues = (
        ArtifactStore(run_ctx)
        .save_json(
            "quality_issues",
            {
                "issues": [
                    {
                        "id": "Q1",
                        "dataset": "events",
                        "column": None,
                        "kind": "duplicate_rows",
                        "severity": "low",
                        "detail": {"rows": 3},
                    },
                    {
                        "id": "Q2",
                        "dataset": "events",
                        "column": "amount",
                        "kind": "negative_values",
                        "severity": "medium",
                        "detail": {"rows": 1, "share": 0.001},
                    },
                ]
            },
        )
        .id
    )
    first = pr.propose_cleaning_plan(issues)
    drop_id = next(o["id"] for o in first.summary["ops"] if o["op"] == "drop_rows")
    second = pr.propose_cleaning_plan(issues, skip_issues=[drop_id])
    assert [o["op"] for o in second.summary["ops"]] == ["dedupe"]


def test_read_json_artifact_allowlist(run_ctx: RunContext) -> None:
    _confirmed_spec(run_ctx)
    joined = pt.test_join("events", "customers", [["phone", "customer_id"]])
    issues = pt.compile_quality_issues([joined.artifact_id])
    assert dt.read_json_artifact(issues.artifact_id).summary["content"]["issues"]
    assert "cannot be read" in dt.read_json_artifact(joined.artifact_id).summary["error"]  # raw keys stay hidden
    assert dt.read_json_artifact("nope_v1").status == "error"


def test_fix_units_on_a_whole_partition(run_ctx: RunContext) -> None:
    result = pr.apply_transform(
        "events", [{"op": "fix_units", "column": "amount", "factor": 10, "file": "date=2026-10-02/"}]
    )
    assert result.status == "ok" and result.summary["operations"][0]["affected_rows"] == 500


def test_flip_sign_inverts_one_segment(run_ctx) -> None:
    from jellybean.skills.preparer import tools as prt

    result = prt.apply_transform(
        "events",
        [
            {
                "op": "flip_sign",
                "column": "amount",
                "date_column": "ts",
                "date_from": "2026-10-02",
                "date_to": "2026-10-02",
                "only_if": "amount > 0",
            }
        ],
    )
    assert result.status == "ok", result.summary
    assert result.summary["operations"][0]["affected_rows"] == 494  # 500 rows of that day, 6 with amount 0
    bad = prt.apply_transform("events", [{"op": "flip_sign", "column": "amount"}])
    assert bad.status == "error" and "needs a filter" in str(bad.summary)
