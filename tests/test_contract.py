"""Contract tests: every shared tool returns a valid ToolResult within limits.

Each tool is called with ordinary and with deliberately oversized requests.
"""

from __future__ import annotations

import json

import pytest

from jellybean.runtime import agent_runner
from jellybean.runtime.agent_runner import Runtime
from jellybean.runtime.context import RunContext
from jellybean.runtime.llm_client import FakeLLM, call
from jellybean.runtime.tool_registry import ToolResult, get_tool_spec, iter_module_tools
from jellybean.skills._shared import data_tools as dt
from jellybean.skills._shared import findings as fd
from jellybean.skills._shared import spec_tools as st
from jellybean.skills.analista import tools as an
from jellybean.skills.critico import tools as cr
from jellybean.skills.detector import tools as de
from jellybean.skills.feature_engineer import tools as fe
from jellybean.skills.orquestrador import tools as ot
from jellybean.skills.preparer import tools as pr
from jellybean.skills.profiler import tools as pt

MODULES = [dt, st, fd, pt, ot, pr, fe, de, cr, an, agent_runner]

MAX_CHARS = 4000
MAX_ROWS = 20

CALLS = [
    # Orchestrator (init_run first so later calls in the sequential test have a spec)
    ("init_run", {"instruction": "analyse events enriched with customers"}),
    (
        "update_spec",
        {
            "patch": {
                "datasets": [
                    {"name": "events", "path": "p", "role": "primary"},
                    {"name": "customers", "path": "p", "role": "enrichment"},
                ]
            }
        },
    ),
    ("get_spec", {}),
    ("ask_human", {"question": "q?", "options": ["a", "b"], "evidence": {"x": "y" * 5000}, "recommended": "a"}),
    ("delegate", {"agent": "profiler", "task": "just finish"}),
    ("prepare_spec_confirmation", {}),
    ("confirm_spec", {}),
    (
        "propose_spec",
        {"join_tests": [], "grain": "missing_v1", "entities": "missing_v1", "semantic_types": ["missing_v1"]},
    ),
    ("finish", {"status": "done", "summary": "s" * 10_000}),
    # Shared data tools
    ("list_datasets", {}),
    ("resolve_dataset", {"name": "events"}),
    ("resolve_dataset", {"name": "unknown_thing"}),
    ("get_schema", {"dataset_or_artifact": "wide"}),
    ("get_schema", {"dataset_or_artifact": "missing"}),
    ("run_sql", {"query": "SELECT * FROM wide"}),  # 300 rows x 31 wide columns
    ("run_sql", {"query": "SELECT * FROM events", "save_as": "all_events"}),
    ("run_sql", {"query": "DROP TABLE events"}),
    ("sample_rows", {"dataset_or_artifact": "wide", "n": 20}),
    ("sample_rows", {"dataset_or_artifact": "events"}),
    ("read_dataset_notes", {"dataset": "customers"}),
    ("get_artifact_info", {"artifact_id": "missing_v1"}),
    ("read_json_artifact", {"artifact_id": "missing_v1"}),
    # Profiler tools
    ("profile_table", {"dataset": "wide"}),
    ("profile_table", {"dataset": "events"}),
    ("infer_semantic_types", {"dataset": "wide"}),
    ("infer_grain", {"dataset": "wide"}),
    ("suggest_entities", {"dataset": "events"}),
    ("check_temporal_coverage", {"dataset": "events", "time_column": "ts"}),
    ("discover_join_candidates", {"left": "events", "right": "customers"}),
    (
        "test_join",
        {
            "left": "events",
            "right": "customers",
            "keys": [["phone", "customer_id"]],
            "normalization": ["strip_prefix_plus351"],
        },
    ),
    ("normalize_key", {"dataset": "events", "column": "phone", "match_against": "customers.customer_id"}),
    ("correlation_summary", {"dataset": "wide"}),
    ("compile_quality_issues", {"artifact_ids": ["missing_v1"]}),
    # Integrity checks (requirements_2.md 10.2)
    ("discover_arithmetic_invariants", {"dataset": "wide"}),
    ("discover_arithmetic_invariants", {"dataset": "events", "columns": ["amount", "nope"]}),
    (
        "check_referential_integrity",
        {
            "dataset": "events",
            "column": "phone",
            "ref_dataset": "customers",
            "ref_column": "customer_id",
            "normalization": "strip_prefix_plus351",
        },
    ),
    (
        "check_referential_integrity",
        {"dataset": "events", "column": "nope", "ref_dataset": "events", "ref_column": "event_id"},
    ),
    ("find_duplicate_keys", {"dataset": "wide"}),
    ("find_duplicate_keys", {"dataset": "events", "columns": ["phone"]}),
    ("classify_nulls", {"dataset": "wide"}),
    ("infer_sign_convention", {"dataset": "events", "measure": "amount", "category": "phone"}),
    ("infer_sign_convention", {"dataset": "events", "measure": "phone", "category": "amount"}),
    ("discover_self_references", {"dataset": "wide"}),
    (
        "check_reciprocity",
        {"dataset": "events", "column": "phone", "counterpart_column": "event_id", "time_column": "ts"},
    ),
    # Control 2
    ("prepare_cleaning_approval", {}),
    ("approve_cleaning_plan", {}),
    # Preparer
    ("propose_cleaning_plan", {"issues_artifact": "missing_v1"}),
    ("apply_transform", {"input": "wide", "operations": [{"op": "dedupe"}, {"op": "normalize", "column": "col_1"}]}),
    ("apply_cleaning_plan", {}),
    ("materialize_join", {"join_index": 0}),
    ("diff_versions", {"a": "wide", "b": "events"}),
    ("validate_constraints", {"artifact": "wide", "rules": [{"column": "id", "check": "unique"}]}),
    # Feature engineer
    ("build_entity_features", {"input": "events", "entity": "phone", "time_column": "ts"}),
    ("build_graph", {"input": "events", "entity": "phone", "link_columns": ["amount"]}),
    ("scale_encode", {"features": "missing_v1"}),
    ("reduce_dimensions", {"features": "missing_v1"}),
    ("feature_summary", {"features": "missing_v1"}),
    # Typed-event feature models (requirements_2.md 10.4); good paths in test_topup_features
    (
        "build_category_features",
        {"input": "events", "entity": "phone", "category": "event_id", "measure": "amount", "features": "missing_v1"},
    ),
    (
        "build_linked_event_features",
        {"input": "events", "entity": "phone", "category": "event_id", "time_column": "ts", "features": "missing_v1"},
    ),
    (
        "build_sequence_features",
        {"input": "events", "entity": "phone", "category": "nope", "time_column": "ts", "features": "missing_v1"},
    ),
    (
        "build_conformity_features",
        {
            "input": "events",
            "entity": "phone",
            "name": "rule",
            "observed": "amount",
            "expected": "(SELECT 1)",
            "features": "missing_v1",
        },
    ),
    (
        "build_interaction_graph",
        {"input": "events", "entity": "phone", "counterpart_column": "event_id", "features": "missing_v1"},
    ),
    # Findings, detector, critic, analyst, control 3 (error paths here; good paths in test_phase5_dummy)
    ("list_findings", {}),
    ("get_finding", {"finding": "F001"}),
    ("run_outlier_detector", {"features": "missing_v1", "method": "ecod"}),
    ("run_clustering", {"features": "missing_v1", "method": "kmeans"}),
    ("plot_cluster_run", {"run_id": "missing_v1"}),
    ("run_graph_analysis", {"graph": "missing_v1"}),
    ("run_timeseries_anomaly", {"input": "events", "time_column": "ts", "measure": "amount", "granularity": "hour"}),
    ("run_contextual_outliers", {"features": "missing_v1", "context_column": "attr_x"}),
    ("consolidate_findings", {"run_ids": ["missing_v1"]}),
    ("stability_check", {"finding": "F001"}),
    ("ensemble_agreement", {"finding": "F001"}),
    ("artifact_check", {"finding": "F001"}),
    ("rerun_on_version", {"finding": "F001", "artifact_version": "missing_v1"}),
    ("inject_synthetic_anomalies", {"features": "missing_v1"}),
    ("compare_to_known_labels", {"finding": "F001"}),
    ("match_pattern_registry", {"finding": "F001"}),
    ("explain_by_reference", {"finding": "F001"}),
    ("set_verdict", {"finding": "F001", "verdict": "weak", "evidence": {}}),
    ("describe_group", {"finding": "F001"}),
    ("fit_surrogate_rules", {"finding": "F001"}),
    ("get_exemplars", {"finding": "F001"}),
    ("make_plot", {"finding": "F001", "kind": "embedding"}),
    ("write_finding", {"finding": "F001", "description": "x" * 30}),
    ("register_pattern", {"finding": "F001"}),
    ("review_findings", {}),
    ("write_report", {}),
]


def _tool(name: str):
    for module in MODULES:
        if hasattr(module, name) and get_tool_spec(getattr(module, name)):
            return getattr(module, name)
    raise KeyError(name)


def test_all_shared_tools_are_covered() -> None:
    specs = [spec for module in MODULES for spec in iter_module_tools(module)]
    assert {spec.name for spec in specs} == {name for name, _ in CALLS}
    for spec in specs:
        assert spec.description, f"{spec.name} needs a docstring description"
        assert spec.input_schema["type"] == "object"


@pytest.fixture
def runtime(run_ctx: RunContext):
    """Active runtime so orchestrator tools take their real code paths."""
    rt = Runtime(
        ctx=run_ctx, llm=FakeLLM({"profiler": [call("finish", status="done", summary="ok") for _ in range(5)]})
    )
    with rt.active():
        yield rt


@pytest.mark.parametrize(("name", "kwargs"), CALLS)
def test_tool_respects_contract(runtime: Runtime, name: str, kwargs: dict) -> None:
    func = _tool(name)
    result = func(**kwargs)
    assert isinstance(result, ToolResult)
    ToolResult.model_validate(result.model_dump())  # schema round-trip
    serialised = result.to_json()
    json.loads(serialised)  # valid JSON
    assert len(serialised) <= MAX_CHARS
    assert result.raw_row_count <= MAX_ROWS
    if result.status == "error":
        assert result.summary.get("error"), "errors must carry a helpful message"


def test_trace_never_exceeds_limits(runtime: Runtime, run_ctx: RunContext) -> None:
    for name, kwargs in CALLS:
        _tool(name)(**kwargs)
    events = [json.loads(line) for line in run_ctx.trace_path.read_text().splitlines()]
    tool_events = [e for e in events if e["kind"] == "tool_call"]
    assert len([e for e in tool_events if e["agent"] == "system"]) == len(CALLS)
    assert max(e["output_chars"] for e in tool_events) <= MAX_CHARS
    assert max(e["raw_rows"] for e in tool_events) <= MAX_ROWS
