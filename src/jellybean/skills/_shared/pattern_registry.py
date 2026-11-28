# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Persistent pattern registry, shared across runs.

Stored in ``workspace/pattern_registry/registry.json`` and indexed by the
combination of datasets and entity (e.g. ``cust_segmentation+traffic|msisdn``).
A pattern's signature holds its dominant features (direction and effect size),
surrogate rules and centroid in the normalised feature space. Matching combines
cosine similarity of centroids with the Jaccard of dominant features:
>= known_threshold -> known, >= variant_threshold -> variant, else new.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jellybean.runtime.context import RunContext
from jellybean.skills._shared.findings import signature_similarity
from jellybean.spec import AnalysisSpec

_LOCK = threading.Lock()


def registry_path(ctx: RunContext) -> Path:
    return ctx.settings.workspace_root / "pattern_registry" / "registry.json"


def registry_key(spec: AnalysisSpec) -> str:
    """Index key: sorted dataset names + entity column."""
    datasets = "+".join(sorted(d.name for d in spec.datasets))
    entity = spec.entity.column if spec.entity else "?"
    return f"{datasets}|{entity}"


def load_registry(ctx: RunContext) -> dict[str, Any]:
    path = registry_path(ctx)
    if not path.exists():
        return {"patterns": []}
    return json.loads(path.read_text(encoding="utf-8"))


def add_pattern(ctx: RunContext, key: str, entry: dict[str, Any]) -> str:
    """Append a pattern; returns its id (P-0001...)."""
    path = registry_path(ctx)
    with _LOCK:
        registry = load_registry(ctx)
        pattern_id = f"PAT-{len(registry['patterns']) + 1:04d}"
        registry["patterns"].append(
            {"id": pattern_id, "key": key, "registered_at": datetime.now(UTC).isoformat(timespec="seconds"), **entry}
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    return pattern_id


def best_match(ctx: RunContext, key: str, sig: dict[str, Any], finding_type: str) -> tuple[dict | None, float]:
    """Most similar registered pattern with the same key and finding type."""
    best, score = None, 0.0
    for pattern in load_registry(ctx)["patterns"]:
        if pattern["key"] != key or pattern.get("type") != finding_type:
            continue
        similarity = signature_similarity(sig, pattern["signature"])
        if similarity > score:
            best, score = pattern, similarity
    return best, score


def classify_novelty(score: float, known: float, variant: float) -> str:
    if score >= known:
        return "known"
    if score >= variant:
        return "variant"
    return "new"
