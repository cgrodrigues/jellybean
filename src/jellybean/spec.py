"""Schema of ``spec.yaml``: the analysis specification and source of truth of a run.

The spec is written by the orchestrator and the profiler and read by every
agent (spec section 6). All writes go through :func:`apply_patch`, which merges
a partial update and validates the result, so an invalid spec can never be
stored.

This module also holds the deterministic join-confidence rules of section 8,
because they decide whether a join can be confirmed automatically.
"""

from __future__ import annotations

import copy
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jellybean.settings import JoinConfidenceConfig

SemanticType = Literal["identifier", "timestamp", "measure", "category", "boolean", "free_text", "geo", "unknown"]
SEMANTIC_TYPES: tuple[str, ...] = SemanticType.__args__  # type: ignore[attr-defined]

Phase = Literal["discovery", "cleaning", "features", "detection", "critique", "analysis", "report", "done"]
JoinStatus = Literal["proposed", "confirmed", "rejected"]


_JOIN_SUMMARY_FIELDS = {
    "left",
    "right",
    "keys",
    "key_normalization",
    "status",
    "coverage",
    "null_key_share",
    "cardinality",
    "evidence_artifact",
}


class _Model(BaseModel):
    """Strict base: unknown keys are errors (catches typos in patches)."""

    model_config = ConfigDict(extra="forbid")


class DatasetRef(_Model):
    name: str
    path: str
    role: Literal["primary", "enrichment"]


class JoinSpec(_Model):
    left: str
    right: str
    # One [left_column, right_column] pair per key component.
    keys: list[tuple[str, str]] = Field(min_length=1)
    # Normalisation strategies (see skills/_shared/keys.py): one per key pair,
    # or a single one applied to every pair.
    key_normalization: list[str] = Field(default_factory=list)
    type: Literal["left", "inner"] = "left"
    cardinality: Literal["1:1", "N:1", "1:N", "N:M"] | None = None
    # Share of primary rows with a non-null key that find a match (sparse keys:
    # rows with a null key are not counted, see null_key_share).
    coverage: float | None = Field(None, ge=0, le=1)
    null_key_share: float | None = Field(None, ge=0, le=1)
    # True when the null keys are determined by another column (e.g. a type column).
    nulls_structural: bool | None = None
    fanout: float | None = Field(None, ge=0)
    right_uniqueness: float | None = Field(None, ge=0, le=1)
    temporal_validity: str | None = None
    confidence: float | None = Field(None, ge=0, le=1)
    status: JoinStatus = "proposed"
    evidence_artifact: str | None = None

    @model_validator(mode="after")
    def _check_normalization_length(self) -> JoinSpec:
        if self.key_normalization and len(self.key_normalization) not in (1, len(self.keys)):
            raise ValueError("key_normalization must have one entry or one per key pair")
        return self


class ReferenceSpec(_Model):
    """A column that refers to another column (same dataset or another), checked for orphans.

    ``kind`` is ``reference`` when the target is unique (a row id: ``ref -> id``)
    and ``counterpart`` when both columns identify the same kind of entity
    (``counterparty -> customer``): their rows are expected to come in pairs.
    """

    dataset: str
    column: str
    ref_dataset: str
    ref_column: str
    kind: Literal["reference", "counterpart"] = "reference"
    orphan_rows: int = 0
    orphan_share: float = Field(0.0, ge=0, le=1)
    null_share: float = Field(0.0, ge=0, le=1)
    nulls_structural: bool | None = None
    evidence_artifact: str | None = None


class InvariantSpec(_Model):
    """An arithmetic relation between columns that holds on (almost) every row."""

    dataset: str
    expression: str
    holds_share: float = Field(ge=0, le=1)
    violations: int = 0
    evidence_artifact: str | None = None


class GrainSpec(_Model):
    dataset: str
    columns: list[str] = Field(min_length=1)


class EntitySpec(_Model):
    column: str
    alternatives: list[str] = Field(default_factory=list)
    # "user": fixed with `jellybean run --entity <dataset>.<column>` (agents cannot change it).
    source: Literal["auto", "user"] = "auto"
    dataset: str | None = None


class ColumnSpec(_Model):
    dataset: str
    semantic_type: SemanticType


class CleaningPlanRef(_Model):
    artifact_id: str | None = None
    status: Literal["pending", "proposed", "approved", "rejected", "applied"] = "pending"
    # Operation ids approved at control 2 (a subset of the plan).
    approved_ops: list[str] = Field(default_factory=list)
    # Signatures (dataset + operation) of the operations actually applied, kept
    # across re-proposals: an approval identical to them changes nothing.
    applied_signature: list[str] = Field(default_factory=list)


class FeatureSet(_Model):
    """Artifacts of the per-entity feature table and its derivatives."""

    table: str | None = None
    dictionary: str | None = None
    scaled: str | None = None
    embedding: str | None = None
    graph: str | None = None


class Decision(_Model):
    """A human or automatic decision, with the evidence it was based on."""

    topic: str
    choice: str
    by: Literal["human", "auto", "agent"]
    evidence: dict[str, Any] = Field(default_factory=dict)
    options: list[str] = Field(default_factory=list)
    at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class AnalysisSpec(_Model):
    """Root of ``spec.yaml``."""

    run_id: str
    instruction: str
    seed: int
    report_language: str = "en"
    datasets: list[DatasetRef] = Field(default_factory=list)
    joins: list[JoinSpec] = Field(default_factory=list)
    grain: GrainSpec | None = None
    entity: EntitySpec | None = None
    time_column: str | None = None
    columns: dict[str, ColumnSpec] = Field(default_factory=dict)
    references: list[ReferenceSpec] = Field(default_factory=list)
    invariants: list[InvariantSpec] = Field(default_factory=list)
    # Columns whose nulls are structural: {"dataset.column": determining column}.
    # They are never imputed: the value does not exist for those rows.
    structural_nulls: dict[str, str] = Field(default_factory=dict)
    # Evidence behind the proposal: {"grain": id, "entity": id, "semantic_types": id, ...}.
    evidence_artifacts: dict[str, str] = Field(default_factory=dict)
    # Optional dataset of confirmed cases (`jellybean run --labels <dataset>`).
    labels_dataset: str | None = None
    # Row filters per dataset, fixed at the start of the run (`jellybean run --filter`).
    filters: dict[str, str] = Field(default_factory=dict)
    context_notes: str = ""
    cleaning_plan: CleaningPlanRef = Field(default_factory=CleaningPlanRef)
    # Cleaned version of each dataset: {dataset: artifact_id}.
    prepared: dict[str, str] = Field(default_factory=dict)
    # Primary dataset enriched with the confirmed joins.
    enriched_artifact: str | None = None
    features: FeatureSet = Field(default_factory=FeatureSet)
    phase: Phase = "discovery"
    decisions: list[Decision] = Field(default_factory=list)

    @field_validator("datasets")
    @classmethod
    def _one_primary(cls, value: list[DatasetRef]) -> list[DatasetRef]:
        if sum(1 for d in value if d.role == "primary") > 1:
            raise ValueError("only one dataset can have role 'primary'")
        names = [d.name for d in value]
        if len(names) != len(set(names)):
            raise ValueError("dataset names must be unique")
        return value

    @model_validator(mode="after")
    def _references_exist(self) -> AnalysisSpec:
        """Joins, grain and columns must refer to datasets declared in the spec."""
        known = {d.name for d in self.datasets}
        if not known:
            return self
        for join in self.joins:
            for side in (join.left, join.right):
                if side not in known:
                    raise ValueError(f"join refers to unknown dataset '{side}'")
        for ref in self.references:
            for side in (ref.dataset, ref.ref_dataset):
                if side not in known:
                    raise ValueError(f"reference refers to unknown dataset '{side}'")
        if self.grain and self.grain.dataset not in known:
            raise ValueError(f"grain refers to unknown dataset '{self.grain.dataset}'")
        for name, col in self.columns.items():
            if col.dataset not in known:
                raise ValueError(f"column '{name}' refers to unknown dataset '{col.dataset}'")
        return self

    def primary(self) -> DatasetRef | None:
        return next((d for d in self.datasets if d.role == "primary"), None)

    def compact_summary(self) -> dict[str, Any]:
        """Short view of the spec given to every agent at start-up."""
        return {
            "run_id": self.run_id,
            "instruction": self.instruction,
            "phase": self.phase,
            "report_language": self.report_language,
            "datasets": [{"name": d.name, "role": d.role} for d in self.datasets],  # paths: list_datasets
            "filters": self.filters,
            # Human context from the datasets' README.md files (column meanings, units...).
            "context_notes": self.context_notes[:700] + ("…" if len(self.context_notes) > 700 else ""),
            # Essential join fields only; the full test is in each join's evidence artifact.
            "joins": [
                {k: v for k, v in j.model_dump().items() if k in _JOIN_SUMMARY_FIELDS and v not in (None, [], "")}
                for j in self.joins
            ],
            "grain": self.grain.model_dump() if self.grain else None,
            "entity": self.entity.model_dump() if self.entity else None,
            "time_column": self.time_column,
            # Counts only: the per-column types are in the semantic_types artifacts.
            "columns": dict(Counter(v.semantic_type for v in self.columns.values())),
            "references": [
                f"{r.dataset}.{r.column} -> {r.ref_dataset}.{r.ref_column} ({r.kind}, {r.orphan_rows} orphans)"
                for r in self.references
            ],
            "invariants": [f"{i.dataset}: {i.expression} ({i.violations} violations)" for i in self.invariants],
            "structural_nulls": {
                det: sorted(c.split(".", 1)[-1] for c, d in self.structural_nulls.items() if d == det)
                for det in sorted(set(self.structural_nulls.values()))
            },
            "evidence_artifacts": {
                k: v for k, v in self.evidence_artifacts.items() if not k.startswith("semantic_types:")
            },
            "cleaning_plan": self.cleaning_plan.model_dump(),
            "prepared": self.prepared,
            "enriched_artifact": self.enriched_artifact,
            "features": {k: v for k, v in self.features.model_dump().items() if v},
            "decisions": len(self.decisions),
        }


# --------------------------------------------------------------------------- #
# Persistence and patching                                                    #
# --------------------------------------------------------------------------- #
def load_spec(path: Path) -> AnalysisSpec:
    """Read and validate ``spec.yaml``."""
    return AnalysisSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def save_spec(path: Path, spec: AnalysisSpec) -> None:
    """Write ``spec.yaml`` atomically."""
    data = spec.model_dump(mode="json")
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, path)


def _merge(base: Any, patch: Any) -> Any:
    """Deep-merge dictionaries; lists and scalars in ``patch`` replace ``base``."""
    if isinstance(base, dict) and isinstance(patch, dict):
        merged = dict(base)
        for key, value in patch.items():
            merged[key] = _merge(base.get(key), value) if key in base else copy.deepcopy(value)
        return merged
    return copy.deepcopy(patch)


# Fields that only the runtime may change (agents cannot rewrite history).
_PROTECTED = {"run_id", "seed", "instruction", "decisions", "labels_dataset", "filters"}


def apply_patch(spec: AnalysisSpec, patch: dict[str, Any], allow_protected: bool = False) -> AnalysisSpec:
    """Return a new spec with ``patch`` merged in and validated.

    Raises:
        ValueError: If the patch touches protected fields or the result is invalid.
    """
    touched = _PROTECTED & set(patch)
    if touched and not allow_protected:
        raise ValueError(f"fields {sorted(touched)} cannot be patched; decisions are recorded via record_decision")
    if (
        not allow_protected
        and "entity" in patch
        and spec.entity is not None
        and spec.entity.source == "user"
        and (patch["entity"] or {}).get("column", spec.entity.column) != spec.entity.column
    ):
        raise ValueError(
            f"the entity '{spec.entity.dataset}.{spec.entity.column}' was set by the user (--entity) "
            "and cannot be changed"
        )
    merged = _merge(spec.model_dump(mode="json"), patch)
    return AnalysisSpec.model_validate(merged)


def record_decision(spec: AnalysisSpec, decision: Decision) -> AnalysisSpec:
    """Append a decision (the only way to grow ``decisions``)."""
    return spec.model_copy(update={"decisions": [*spec.decisions, decision]})


# --------------------------------------------------------------------------- #
# Join confidence (spec section 8)                                            #
# --------------------------------------------------------------------------- #
class JoinAssessment(_Model):
    """Outcome of the join-confidence rules."""

    confidence: float
    auto_confirm: bool
    criteria: dict[str, dict[str, Any]]
    requires_confirmation_reasons: list[str]


def assess_join(
    coverage: float,
    right_uniqueness: float,
    fanout: float,
    score_margin: float | None,
    n_keys: int,
    cardinality: str | None,
    temporal_validity: str | None,
    thresholds: JoinConfidenceConfig,
    null_key_share: float = 0.0,
    nulls_structural: bool | None = None,
) -> JoinAssessment:
    """Apply the section-8 rules.

    A join is auto-confirmed only when **all** threshold criteria pass and none
    of the "always confirm" conditions apply (composite keys, N:M relations,
    dimensions with temporal validity).

    Confidence is ``coverage * right_uniqueness / fanout``, multiplied by 0.8
    when the score margin over the runner-up is missing or below the threshold.
    It is a ranking aid; the decision itself uses the criteria.

    Coverage is measured on rows with a non-null key (section 10.3 of
    ``requirements_2.md``). A sparse key, with more null keys than the coverage
    tolerance, can still be auto-confirmed when its nulls are structural
    (decided by another column); otherwise it needs a human.
    """
    criteria = {
        "coverage": {
            "value": round(coverage, 4),
            "threshold": f">= {thresholds.min_coverage}",
            "ok": coverage >= thresholds.min_coverage,
        },
        "right_uniqueness": {
            "value": round(right_uniqueness, 5),
            "threshold": f">= {thresholds.min_right_uniqueness}",
            "ok": right_uniqueness >= thresholds.min_right_uniqueness,
        },
        "fanout": {
            "value": round(fanout, 5),
            "threshold": f"<= {thresholds.max_fanout}",
            "ok": fanout <= thresholds.max_fanout,
        },
        "score_margin": {
            "value": None if score_margin is None else round(score_margin, 4),
            "threshold": f">= {thresholds.min_score_margin}",
            "ok": score_margin is not None and score_margin >= thresholds.min_score_margin,
        },
    }
    reasons = []
    if n_keys > 1:
        reasons.append("composite key")
    if cardinality == "N:M":
        reasons.append("N:M relation")
    if temporal_validity:
        reasons.append("dimension with temporal validity")
    if null_key_share > 1 - thresholds.min_coverage and not nulls_structural:
        reasons.append(f"sparse key ({null_key_share:.0%} null) whose nulls are not structural")
    margin_factor = 1.0 if criteria["score_margin"]["ok"] else 0.8
    confidence = max(0.0, min(1.0, coverage * right_uniqueness / max(fanout, 1e-9) * margin_factor))
    auto = all(c["ok"] for c in criteria.values()) and not reasons
    return JoinAssessment(
        confidence=round(confidence, 4), auto_confirm=auto, criteria=criteria, requires_confirmation_reasons=reasons
    )
