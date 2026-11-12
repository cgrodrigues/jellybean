"""Shared tools to read and update ``spec.yaml`` (written by orchestrator and profiler).

Every update is a validated patch (see :func:`jellybean.spec.apply_patch`);
an invalid patch is rejected and the stored spec is left untouched.
"""

from __future__ import annotations

import threading
from typing import Any

from pydantic import ValidationError

from jellybean.runtime.context import RunContext, current_run
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.runtime.tracing import trace_event
from jellybean.spec import AnalysisSpec, Decision, apply_patch, load_spec, record_decision, save_spec

# Serialise read-modify-write cycles of spec.yaml inside the process.
_SPEC_LOCK = threading.Lock()


class SpecMissingError(FileNotFoundError):
    """Raised when the run has no spec.yaml yet."""


def read_spec(ctx: RunContext | None = None) -> AnalysisSpec:
    ctx = ctx or current_run()
    if not ctx.spec_path.exists():
        raise SpecMissingError("No spec.yaml yet: the orchestrator must call init_run first.")
    return load_spec(ctx.spec_path)


def patch_spec(patch: dict[str, Any], ctx: RunContext | None = None, allow_protected: bool = False) -> AnalysisSpec:
    """Apply a patch atomically and return the new spec."""
    ctx = ctx or current_run()
    with _SPEC_LOCK:
        spec = apply_patch(read_spec(ctx), patch, allow_protected=allow_protected)
        save_spec(ctx.spec_path, spec)
    trace_event("spec_update", fields=sorted(patch), phase=patch.get("phase"))
    return spec


def add_decision(decision: Decision, ctx: RunContext | None = None) -> AnalysisSpec:
    """Append a decision to the spec (the only way decisions are written)."""
    ctx = ctx or current_run()
    with _SPEC_LOCK:
        spec = record_decision(read_spec(ctx), decision)
        save_spec(ctx.spec_path, spec)
    trace_event("decision", topic=decision.topic, choice=decision.choice, by=decision.by)
    return spec


@tool
def get_spec() -> ToolResult:
    """Return a compact view of the current spec.yaml (datasets, joins, grain, entity, time column, phase)."""
    try:
        return ToolResult.ok(read_spec().compact_summary())
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))


def read_spec_safe() -> AnalysisSpec | None:
    try:
        return read_spec()
    except SpecMissingError:
        return None


def _join_edit_problem(spec: AnalysisSpec | None, patch: dict[str, Any]) -> str | None:
    """Joins measured by test_join cannot be dropped or have their keys/normalisation changed by hand."""
    if spec is None or "joins" not in patch or not isinstance(patch["joins"], list):
        return None

    def ident(j: Any) -> tuple:
        d = j if isinstance(j, dict) else j.model_dump()
        return (d.get("left"), d.get("right"))

    new = {ident(j): j for j in patch["joins"] if isinstance(j, dict)}
    for join in spec.joins:
        if not join.evidence_artifact:
            continue
        other = new.get(ident(join))
        if other is None:
            return (
                f"The join {join.left} -> {join.right} comes from test_join ({join.evidence_artifact}) and cannot be "
                "removed with update_spec. Patches replace the whole list: send every join, or reject it at control 1."
            )
        keys = [list(k) for k in other.get("keys", join.keys)]
        norm = other.get("key_normalization", join.key_normalization)
        if keys != [list(k) for k in join.keys] or list(norm) != list(join.key_normalization):
            return (
                f"The keys and normalisation of {join.left} -> {join.right} come from test_join "
                f"({join.evidence_artifact}): keys {join.keys}, normalisation {join.key_normalization or ['none']}. "
                "To change them, run test_join with the new keys and propose_spec."
            )
    return None


@tool
def update_spec(patch: dict[str, Any]) -> ToolResult:
    """Merge a partial update into spec.yaml, validating the result against the schema.

    Nested objects are merged; lists and scalars are replaced. `run_id`, `seed`,
    `instruction`, `labels_dataset` and `decisions` cannot be patched.

    Args:
        patch: Fields to change, e.g. {"time_column": "start_time", "phase": "cleaning"}.
    """
    problem = _join_edit_problem(read_spec_safe(), patch)
    if problem:
        return ToolResult.error(problem)
    try:
        spec = patch_spec(patch)
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    except (ValidationError, ValueError) as exc:
        return ToolResult.error(f"Patch rejected: {exc}"[:1500])
    return ToolResult.ok({"updated": sorted(patch), "spec": spec.compact_summary()})
