"""Findings: candidate anomalies/patterns and their lifecycle (spec section 6).

A finding is a JSON record in ``workspace/<run_id>/findings/<id>.json`` with a
fixed schema. It is created by the detector, judged by the critic, described by
the analyst and reviewed by a human. The entity ids of a finding are stored in a
table artifact (``entity_ids_artifact``), never in the JSON shown to agents.

This module also holds helpers shared by detector, critic and analyst: loading
the feature tables recorded in the spec and computing effect sizes and
signatures.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from jellybean.runtime.context import RunContext, current_run
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.spec_tools import SpecMissingError, read_spec

FindingType = Literal["outlier_group", "cluster", "community", "temporal"]
Verdict = Literal["confirmed", "explained", "artifact", "unstable", "weak"]
Novelty = Literal["known", "variant", "new"]
_ID_RE = re.compile(r"^F\d{3,}$")
_LOCK = threading.Lock()


class Finding(BaseModel):
    """One finding. Field names follow spec section 6, plus bookkeeping fields."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: FindingType
    entity_column: str
    entity_ids_artifact: str
    size: int
    methods: list[str]
    scores: dict[str, Any] = Field(default_factory=dict)
    critic_verdict: Verdict | None = None
    critic_evidence: dict[str, Any] | None = None
    novelty: Novelty | None = None
    description: str = ""
    top_features: list[dict[str, Any]] = Field(default_factory=list)
    rules: list[dict[str, Any]] = Field(default_factory=list)
    exemplars: list[dict[str, Any]] = Field(default_factory=list)
    plots: list[str] = Field(default_factory=list)
    human_feedback: Literal["confirmed", "false_alarm"] | None = None
    # Bookkeeping (not in the spec table, but needed by the pipeline).
    source_runs: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    checks: dict[str, Any] = Field(default_factory=dict)
    pattern_match: dict[str, Any] | None = None
    registered_pattern: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def summary(self) -> dict[str, Any]:
        """Compact view for agents (no entity ids)."""
        return {
            "id": self.id,
            "type": self.type,
            "size": self.size,
            "entity_column": self.entity_column,
            "methods": self.methods,
            "scores": self.scores,
            "context": self.context,
            "top_features": self.top_features[:8],
            "verdict": self.critic_verdict,
            "novelty": self.novelty,
            "checks_done": sorted(self.checks),
            "human_feedback": self.human_feedback,
            "has_description": bool(self.description),
            "rules": self.rules[:3],
        }


class FindingNotFoundError(LookupError):
    pass


def _path(ctx: RunContext, finding_id: str):
    if not _ID_RE.match(finding_id):
        raise FindingNotFoundError(f"Invalid finding id '{finding_id}' (expected F001, F002...)")
    return ctx.findings_dir / f"{finding_id}.json"


def load_finding(finding_id: str, ctx: RunContext | None = None) -> Finding:
    ctx = ctx or current_run()
    path = _path(ctx, finding_id)
    if not path.exists():
        raise FindingNotFoundError(f"Unknown finding '{finding_id}'")
    return Finding.model_validate_json(path.read_text(encoding="utf-8"))


def save_finding(finding: Finding, ctx: RunContext | None = None) -> None:
    """Write a finding atomically (findings are mutable records, unlike artifacts)."""
    ctx = ctx or current_run()
    ctx.findings_dir.mkdir(parents=True, exist_ok=True)
    path = _path(ctx, finding.id)
    with _LOCK:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(finding.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)


def all_findings(ctx: RunContext | None = None) -> list[Finding]:
    ctx = ctx or current_run()
    if not ctx.findings_dir.exists():
        return []
    return [
        Finding.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(ctx.findings_dir.glob("F*.json"))
    ]


def finding_entities(finding: Finding, ctx: RunContext | None = None) -> pd.DataFrame:
    """Entity ids (and, for temporal findings, the windows) of a finding."""
    ctx = ctx or current_run()
    return pd.read_parquet(ArtifactStore(ctx).path_of(finding.entity_ids_artifact))


# --------------------------------------------------------------------------- #
# Feature tables                                                              #
# --------------------------------------------------------------------------- #
def feature_tables(ctx: RunContext | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(raw feature table, scaled feature table, entity column) recorded in the spec."""
    ctx = ctx or current_run()
    spec = read_spec(ctx)
    if not spec.features.table or not spec.features.scaled or not spec.entity:
        raise SpecMissingError("Feature tables are missing in the spec: run the feature engineer first.")
    store = ArtifactStore(ctx)
    raw = pd.read_parquet(store.path_of(spec.features.table))
    scaled = pd.read_parquet(store.path_of(spec.features.scaled))
    entity = store.get(spec.features.scaled).params.get("entity") or spec.entity.column
    return raw, scaled, entity


def family_columns(columns: list[str], families: list[str] | None, ctx: RunContext | None = None) -> list[str]:
    """Columns of a scaled matrix whose feature belongs to one of ``families`` (dictionary templates).

    ``None`` keeps every column. One-hot columns of an attribute ``attr_x`` belong
    to the attribute's template.
    """
    if not families:
        return list(columns)
    ctx = ctx or current_run()
    spec = read_spec(ctx)
    if not spec.features.dictionary:
        return []
    templates = {d["name"]: d["template"] for d in ArtifactStore(ctx).load_json(spec.features.dictionary)["features"]}
    wanted = set(families)
    out = []
    for c in columns:
        template = templates.get(c)
        if template is None:  # one-hot column of an attribute (attr_<col>_<value>)
            template = next((t for n, t in templates.items() if n.startswith("attr_") and c.startswith(n + "_")), None)
        if template in wanted:
            out.append(c)
    return out


def numeric_matrix(scaled: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Numeric columns of a scaled table, indexed by entity."""
    return scaled.set_index(entity).select_dtypes(include=["number", "bool"]).astype(float)


def effect_sizes(raw: pd.DataFrame, entity: str, members: set[Any], top: int = 8) -> list[dict[str, Any]]:
    """Features that differ most between a group and the population.

    Numeric features: robust effect = (median_group - median_population) / scale,
    with scale = 1.4826 * MAD of the population (std when MAD is 0). Categorical
    attributes: the largest share difference of a value. Sorted by |effect|.
    """
    mask = raw[entity].isin(members)
    if not mask.any():
        return []
    group, rest = raw[mask], raw[~mask] if (~mask).any() else raw
    out: list[dict[str, Any]] = []
    for col in raw.columns:
        if col == entity:
            continue
        series = raw[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            pop = rest[col].dropna().astype(float)
            grp = group[col].dropna().astype(float)
            if pop.empty or grp.empty:
                continue
            med_p = float(pop.median())
            scale = 1.4826 * float((pop - med_p).abs().median())
            if scale == 0:
                scale = float(pop.std()) or 0.0
            if scale == 0:
                continue
            effect = (float(grp.median()) - med_p) / scale
            out.append(
                {
                    "feature": col,
                    "kind": "numeric",
                    "group_median": round(float(grp.median()), 4),
                    "population_median": round(med_p, 4),
                    "effect": round(float(np.clip(effect, -1e3, 1e3)), 3),
                    "direction": "higher" if effect > 0 else "lower",
                }
            )
        else:
            g_share = group[col].astype("string").fillna("__missing__").value_counts(normalize=True)
            p_share = rest[col].astype("string").fillna("__missing__").value_counts(normalize=True)
            diff = (g_share - p_share.reindex(g_share.index).fillna(0)).sort_values(key=abs, ascending=False)
            if diff.empty:
                continue
            value = diff.index[0]
            out.append(
                {
                    "feature": col,
                    "kind": "categorical",
                    "value": str(value),
                    "group_share": round(float(g_share[value]), 4),
                    "population_share": round(float(p_share.get(value, 0.0)), 4),
                    "effect": round(float(diff.iloc[0]) * 10, 3),  # share difference x10, comparable scale
                    "direction": "higher" if diff.iloc[0] > 0 else "lower",
                }
            )
    out.sort(key=lambda d: -abs(d["effect"]))
    return out[:top]


def signature(
    raw: pd.DataFrame,
    scaled: pd.DataFrame,
    entity: str,
    members: set[Any],
    top_features: list[dict[str, Any]] | None = None,
    rules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pattern signature: dominant features (direction, effect), rules and centroid (scaled space)."""
    top_features = top_features or effect_sizes(raw, entity, members)
    matrix = numeric_matrix(scaled, entity)
    centroid = matrix[matrix.index.isin(members)].median()
    return {
        "dominant_features": [
            {"feature": f["feature"], "direction": f["direction"], "effect": f["effect"]} for f in top_features[:8]
        ],
        "rules": [r.get("rule") for r in (rules or []) if r.get("rule")],
        "centroid": {k: round(float(v), 4) for k, v in centroid.items() if np.isfinite(v)},
    }


def signature_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Average of cosine similarity of centroids (common features) and Jaccard of dominant features."""
    ca, cb = a.get("centroid", {}), b.get("centroid", {})
    common = sorted(set(ca) & set(cb))
    cosine = 0.0
    if common:
        va, vb = np.array([ca[k] for k in common]), np.array([cb[k] for k in common])
        norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
        cosine = float(va @ vb / norm) if norm else 0.0
    fa = {(f["feature"], f["direction"]) for f in a.get("dominant_features", [])}
    fb = {(f["feature"], f["direction"]) for f in b.get("dominant_features", [])}
    jaccard = len(fa & fb) / len(fa | fb) if fa | fb else 0.0
    return round(0.5 * max(cosine, 0.0) + 0.5 * jaccard, 4)


# --------------------------------------------------------------------------- #
# Shared tools                                                                #
# --------------------------------------------------------------------------- #
@tool
def list_findings() -> ToolResult:
    """List the findings of this run: id, type, size, methods, verdict, novelty, feedback."""
    findings = all_findings()
    return ToolResult.ok(
        {
            "count": len(findings),
            "findings": [
                {
                    "id": f.id,
                    "type": f.type,
                    "size": f.size,
                    "methods": f.methods[:6],
                    "verdict": f.critic_verdict,
                    "novelty": f.novelty,
                    "human_feedback": f.human_feedback,
                    "described": bool(f.description),
                }
                for f in findings
            ],
        }
    )


@tool
def get_finding(finding: str) -> ToolResult:
    """Show one finding: type, size, methods, scores, top features, checks done, verdict (never entity ids).

    Args:
        finding: Finding id, e.g. "F001".
    """
    try:
        f = load_finding(finding)
    except FindingNotFoundError as exc:
        return ToolResult.error(str(exc))
    return ToolResult.ok(f.summary(), artifact_id=f.entity_ids_artifact)


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=str)
