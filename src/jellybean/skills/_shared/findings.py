# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

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
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from jellybean.runtime.cancellation import check_cancelled
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
    check_cancelled()  # a call that timed out changes nothing
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
@dataclass(frozen=True)
class FeatureSource:
    """Where the feature tables of the run are, without loading them."""

    raw_path: Path
    scaled_path: Path
    entity: str

    def columns(self, which: Literal["raw", "scaled"] = "raw") -> list[str]:
        return list(pq.read_schema(self.raw_path if which == "raw" else self.scaled_path).names)

    def n_rows(self) -> int:
        return int(pq.read_metadata(self.raw_path).num_rows)


def feature_source(ctx: RunContext | None = None) -> FeatureSource:
    """Paths of the raw and scaled feature tables recorded in the spec, and the entity column."""
    ctx = ctx or current_run()
    spec = read_spec(ctx)
    if not spec.features.table or not spec.features.scaled or not spec.entity:
        raise SpecMissingError("Feature tables are missing in the spec: run the feature engineer first.")
    store = ArtifactStore(ctx)
    entity = store.get(spec.features.scaled).params.get("entity") or spec.entity.column
    return FeatureSource(store.path_of(spec.features.table), store.path_of(spec.features.scaled), entity)


def read_features(
    path: Path, entity: str, columns: list[str] | None = None, members: Iterable[Any] | None = None
) -> pd.DataFrame:
    """Rows of a feature table, in file order: only ``columns`` and, with ``members``, only those entities.

    Both filters are applied while reading, so a group of a few entities costs
    little memory whatever the size of the table.
    """
    if columns is not None:
        columns = list(dict.fromkeys(columns))
    if members is None:
        return pd.read_parquet(path, columns=columns)
    wanted = pd.Index(list(members))
    file = pq.ParquetFile(path)

    def matching(as_text: bool) -> list[pa.Table]:
        # One row group at a time: its ids first, then the wanted rows of the groups that have any.
        chosen = pd.Index([str(m) for m in wanted]) if as_text else wanted
        pieces = []
        for group in range(file.num_row_groups):
            check_cancelled()
            ids = file.read_row_group(group, columns=[entity]).column(entity).to_pandas()
            keep = np.flatnonzero((ids.astype(str) if as_text else ids).isin(chosen).to_numpy())
            if len(keep):
                pieces.append(file.read_row_group(group, columns=columns).take(keep))
        return pieces

    pieces = matching(as_text=False) if len(wanted) else []
    if not pieces and len(wanted):
        pieces = matching(as_text=True)  # ids whose type does not match the stored column
    if not pieces:
        empty = file.schema_arrow.empty_table()
        pieces = [empty.select(columns) if columns is not None else empty]
    return pa.concat_tables(pieces).to_pandas().reset_index(drop=True)


def feature_tables(ctx: RunContext | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(raw feature table, scaled feature table, entity column), both loaded in full.

    Costly on large populations: prefer :func:`feature_source` with
    :func:`read_features`, :func:`group_effect_sizes` or :func:`group_signature`.
    """
    source = feature_source(ctx)
    return pd.read_parquet(source.raw_path), pd.read_parquet(source.scaled_path), source.entity


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


_COLUMN_CHUNK = 16  # columns of a feature table read at a time


def numeric_columns(path: Path, entity: str) -> list[str]:
    """Numeric and boolean columns of a feature table file (the columns of :func:`numeric_matrix`)."""
    types = pa.types
    return [
        f.name
        for f in pq.read_schema(path)
        if f.name != entity and (types.is_integer(f.type) or types.is_floating(f.type) or types.is_boolean(f.type))
    ]


def read_numeric_matrix(
    path: Path, entity: str, columns: list[str] | None = None, members: Iterable[Any] | None = None
) -> pd.DataFrame:
    """:func:`numeric_matrix` of a feature table file, reading only the numeric columns (or ``columns`` of them).

    The whole table is filled a few columns at a time into one array, so its
    peak memory is one copy of the matrix (not the several a DataFrame load makes).
    """
    numeric = numeric_columns(path, entity)
    if columns is not None:
        wanted = set(columns)
        numeric = [c for c in numeric if c in wanted]
    if members is not None:
        return numeric_matrix(read_features(path, entity, columns=[entity, *numeric], members=members), entity)
    ids = pd.read_parquet(path, columns=[entity])[entity]
    matrix = np.empty((len(ids), len(numeric)), dtype=float)
    for start in range(0, len(numeric), _COLUMN_CHUNK):
        check_cancelled()
        names = numeric[start : start + _COLUMN_CHUNK]
        block = pd.read_parquet(path, columns=names)
        for offset, name in enumerate(names):
            matrix[:, start + offset] = block[name].to_numpy(dtype=float, na_value=np.nan)
    return pd.DataFrame(matrix, index=pd.Index(ids, name=entity), columns=numeric, copy=False)


def _is_measure(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _numeric_stats(pop: pd.Series) -> tuple[float, float] | None:
    """(median, robust scale) of a population column; None when it is empty."""
    pop = pop.dropna().astype(float)
    if pop.empty:
        return None
    med = float(pop.median())
    scale = 1.4826 * float((pop - med).abs().median())
    if scale == 0:
        scale = float(pop.std()) or 0.0
    return med, scale


def _shares(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__missing__").value_counts(normalize=True)


def _numeric_effect(col: str, grp: pd.Series, stats: tuple[float, float] | None) -> dict[str, Any] | None:
    grp = grp.dropna().astype(float)
    if stats is None or grp.empty or stats[1] == 0:
        return None
    med_p, scale = stats
    effect = (float(grp.median()) - med_p) / scale
    return {
        "feature": col,
        "kind": "numeric",
        "group_median": round(float(grp.median()), 4),
        "population_median": round(med_p, 4),
        "effect": round(float(np.clip(effect, -1e3, 1e3)), 3),
        "direction": "higher" if effect > 0 else "lower",
    }


def _categorical_effect(col: str, g_share: pd.Series, p_share: pd.Series) -> dict[str, Any] | None:
    diff = (g_share - p_share.reindex(g_share.index).fillna(0)).sort_values(key=abs, ascending=False)
    if diff.empty:
        return None
    value = diff.index[0]
    return {
        "feature": col,
        "kind": "categorical",
        "value": str(value),
        "group_share": round(float(g_share[value]), 4),
        "population_share": round(float(p_share.get(value, 0.0)), 4),
        "effect": round(float(diff.iloc[0]) * 10, 3),  # share difference x10, comparable scale
        "direction": "higher" if diff.iloc[0] > 0 else "lower",
    }


def effect_sizes(raw: pd.DataFrame, entity: str, members: set[Any], top: int = 8) -> list[dict[str, Any]]:
    """Features that differ most between a group and the population.

    Numeric features: robust effect = (median_group - median_population) / scale,
    with scale = 1.4826 * MAD of the population (std when MAD is 0). Categorical
    attributes: the largest share difference of a value. Sorted by |effect|.
    The population is every entity outside the group.
    """
    mask = raw[entity].isin(members)
    if not mask.any():
        return []
    group, rest = raw[mask], raw[~mask] if (~mask).any() else raw
    out: list[dict[str, Any]] = []
    for col in raw.columns:
        if col == entity:
            continue
        if _is_measure(raw[col]):
            found = _numeric_effect(col, group[col], _numeric_stats(rest[col]))
        else:
            found = _categorical_effect(col, _shares(group[col]), _shares(rest[col]))
        if found:
            out.append(found)
    out.sort(key=lambda d: -abs(d["effect"]))
    return out[:top]


# Population statistics of large feature tables, per file: {column: (median, scale) | shares}.
_POPULATION_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}
_POPULATION_CACHE_MAX = 4


def clear_population_cache() -> None:
    _POPULATION_CACHE.clear()


def _population_stats(path: Path, entity: str) -> dict[str, Any]:
    """Median/scale (numeric) or value shares (categorical) of every column, read a few columns at a time."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _POPULATION_CACHE.get(key)
    if cached is not None:
        return cached
    columns = [c for c in pq.read_schema(path).names if c != entity]
    stats: dict[str, Any] = {}
    for start in range(0, len(columns), _COLUMN_CHUNK):
        check_cancelled()
        block = pd.read_parquet(path, columns=columns[start : start + _COLUMN_CHUNK])
        for col in block.columns:
            stats[col] = _numeric_stats(block[col]) if _is_measure(block[col]) else _shares(block[col])
    if len(_POPULATION_CACHE) >= _POPULATION_CACHE_MAX:
        _POPULATION_CACHE.clear()
    _POPULATION_CACHE[key] = stats
    return stats


def group_effect_sizes(members: set[Any], top: int = 8, ctx: RunContext | None = None) -> list[dict[str, Any]]:
    """:func:`effect_sizes` of a group against the run's raw feature table, without loading it whole.

    Up to ``detection.exact_population_max_rows`` entities the result is the
    same as :func:`effect_sizes` (population = every entity outside the group).
    Above it, the population statistics are computed once per feature table over
    **all** entities (the group included) and reused for every group; only the
    group's rows are read.
    """
    ctx = ctx or current_run()
    source = feature_source(ctx)
    if source.n_rows() <= ctx.settings.config.detection.exact_population_max_rows:
        return effect_sizes(pd.read_parquet(source.raw_path), source.entity, members, top=top)
    group = read_features(source.raw_path, source.entity, members=members)
    if group.empty:
        return []
    population = _population_stats(source.raw_path, source.entity)
    out: list[dict[str, Any]] = []
    for col in group.columns:
        if col == source.entity:
            continue
        if _is_measure(group[col]):
            stats = population[col] if isinstance(population[col], tuple) else None
            found = _numeric_effect(col, group[col], stats)
        else:
            found = _categorical_effect(col, _shares(group[col]), population[col])
        if found:
            out.append(found)
    out.sort(key=lambda d: -abs(d["effect"]))
    return out[:top]


def _signature(
    centroid: pd.Series, top_features: list[dict[str, Any]], rules: list[dict[str, Any]] | None
) -> dict[str, Any]:
    return {
        "dominant_features": [
            {"feature": f["feature"], "direction": f["direction"], "effect": f["effect"]} for f in top_features[:8]
        ],
        "rules": [r.get("rule") for r in (rules or []) if r.get("rule")],
        "centroid": {k: round(float(v), 4) for k, v in centroid.items() if np.isfinite(v)},
    }


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
    return _signature(matrix[matrix.index.isin(members)].median(), top_features, rules)


def group_signature(
    members: set[Any],
    top_features: list[dict[str, Any]] | None = None,
    rules: list[dict[str, Any]] | None = None,
    ctx: RunContext | None = None,
) -> dict[str, Any]:
    """:func:`signature` of a group of the run's feature tables, reading only the group's rows."""
    ctx = ctx or current_run()
    source = feature_source(ctx)
    top_features = top_features or group_effect_sizes(members, ctx=ctx)
    matrix = numeric_matrix(read_features(source.scaled_path, source.entity, members=members), source.entity)
    return _signature(matrix.median(), top_features, rules)


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
