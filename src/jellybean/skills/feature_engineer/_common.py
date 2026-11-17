# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the feature engineer's tool modules (``tools.py`` and ``models.py``)."""

from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd

from jellybean.runtime.context import current_run
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.column_stats import classify_column, compute_column_stats


def _store() -> ArtifactStore:
    return ArtifactStore(current_run())


def _feature_name(*parts: Any) -> str:
    """Safe, readable feature name: lowercase letters, digits and underscores."""
    text = "_".join(str(p) for p in parts if p not in (None, ""))
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


def _semantic_types(spec: Any, con: duckdb.DuckDBPyConnection, view: str, columns: list[str]) -> dict[str, str]:
    """Semantic type per column: from the spec when known, else inferred on the fly."""
    known: dict[str, str] = {}
    for col in columns:
        for key, info in spec.columns.items():
            if key == col or key.endswith(f".{col}"):
                known[col] = info.semantic_type
                break
    missing = [c for c in columns if c not in known]
    if missing:
        _, stats = compute_column_stats(con, view, columns=missing, with_top=True, with_samples=True)
        known.update({s.name: classify_column(s).semantic_type for s in stats})
    return known


def _load_table(artifact_id: str) -> pd.DataFrame:
    """Load a per-entity table artifact into pandas (entity tables are small)."""
    path = _store().path_of(artifact_id)
    return pd.read_parquet(path)


def _entity_of(artifact_id: str) -> str:
    """Entity column recorded when a feature table was created."""
    store = _store()
    record = store.get(artifact_id)
    entity = record.params.get("entity")
    for parent in store.lineage(artifact_id) if entity is None else []:
        entity = parent.params.get("entity")
        if entity:
            break
    if not entity:
        raise ValueError(f"'{artifact_id}' is not a feature table (no entity recorded).")
    return entity


def latest_features(features: str) -> tuple[str, bool]:
    """The feature table to extend: the spec's latest version when ``features`` is one of its ancestors.

    An agent that asks for several extensions in one turn passes the same
    version to all of them; extending each from that version would fork the
    table and lose the columns of the others. Returns (table, rebased).
    """
    from jellybean.skills._shared.spec_tools import SpecMissingError, read_spec

    try:
        current = read_spec().features.table
    except SpecMissingError:
        return features, False
    if not current or current == features:
        return features, False
    store = _store()
    try:
        ancestors = {r.id for r in store.lineage(current)}
    except Exception:
        return features, False
    return (current, True) if features in ancestors else (features, False)
