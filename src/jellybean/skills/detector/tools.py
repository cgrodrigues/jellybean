# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Detector tools (spec section 7.5).

Runs several unsupervised methods on the per-entity features (and the event
table for time series) and consolidates their results into candidate findings.
Each detection run is stored as a table artifact (entity, score, flagged [,
label]) whose params record the kind, method and parameters, so the critic can
re-run exactly the same method.

The detector must use at least two outlier methods and one clustering method
per run (enforced in :func:`consolidate_findings`).
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import Field

from jellybean.runtime.context import current_run
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.skills._shared.artifacts import ArtifactNotFoundError, ArtifactStore, prefix_part
from jellybean.skills._shared.catalog import SourceNotFoundError, connect, quote_ident, resolve_source
from jellybean.skills._shared.column_stats import describe, type_family
from jellybean.skills._shared.detection import (
    DetectionError,
    cluster_labels,
    component_table,
    contextual_scores,
    event_unit,
    flag_top,
    flag_within_groups,
    grouped_series_anomalies,
    low_support,
    merge_windows,
    outlier_scores,
    seasonal_residual_anomalies,
)
from jellybean.skills._shared.findings import (
    Finding,
    all_findings,
    family_columns,
    feature_source,
    group_effect_sizes,
    numeric_columns,
    read_features,
    read_numeric_matrix,
    save_finding,
)
from jellybean.skills._shared.masking import Masker
from jellybean.skills._shared.spec_tools import SpecMissingError, patch_spec

OutlierMethod = Literal["isolation_forest", "lof", "ecod", "hbos", "autoencoder"]
ClusterMethod = Literal["hdbscan", "gmm", "kmeans"]
GraphMethod = Literal["components", "communities", "centrality"]
_MAX_SERIES_POINTS = 3_000_000
_MIN_SERIES_BUCKETS = 8  # fewer time buckets than this cannot show what is usual


def _store() -> ArtifactStore:
    return ArtifactStore(current_run())


def _cfg():
    return current_run().settings.config.detection


def _load_matrix(features: str, families: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Numeric scaled matrix (indexed by entity) and the entity column.

    With ``families`` only the columns of those feature templates are read (the
    matrix has no column when the table has no feature of these families).
    """
    store = _store()
    record = store.get(features)
    entity = record.params.get("entity")
    if not entity:
        raise ValueError(f"'{features}' is not a feature table.")
    path = store.path_of(features)
    columns = family_columns(numeric_columns(path, entity), families) if families else None
    matrix = read_numeric_matrix(path, entity, columns=columns)
    if matrix.isna().any().any():
        raise ValueError(f"'{features}' has nulls: use the scaled table (scale_encode output).")
    return matrix, entity


def _save_run(kind: str, method: str, table: pd.DataFrame, parents: list[str], params: dict[str, Any]) -> str:
    record = _store().save_dataframe(
        f"detection_{prefix_part(method)}", table, parents=parents, params={"kind": kind, "method": method, **params}
    )
    return record.id


@tool
def run_outlier_detector(
    features: str,
    method: OutlierMethod,
    params: dict[str, Any] | None = None,
    contamination: Annotated[float | None, Field(gt=0, lt=0.5)] = None,
    seed: int | None = None,
    families: list[str] | None = None,
) -> ToolResult:
    """Score every entity with an outlier method and flag the most anomalous share.

    Methods: isolation_forest, lof (sampled above a size limit), ecod, hbos,
    autoencoder (optional extra "deep"). With `families`, only the features of
    those templates of the feature dictionary are used (e.g. ["linked"]): an
    anomaly that lives in a few features is not diluted by hundreds of others.
    In small populations each method flags at least twice `min_group_size` entities.

    Args:
        features: Scaled feature table (spec.features.scaled).
        method: Outlier method.
        params: Method parameters, e.g. {"n_neighbors": 35} for lof.
        contamination: Share of entities to flag (default from config).
        seed: Random seed (defaults to the run seed).
        families: Feature templates to use (e.g. ["linked"], ["conformity"]); default all features.
    """
    cfg = _cfg()
    contamination = contamination or cfg.contamination
    try:
        matrix, entity = _load_matrix(features, families)
        if families and not matrix.shape[1]:
            # Not an error: the feature table simply has no feature of this family.
            return ToolResult.ok(
                {"skipped": True, "families": families, "reason": "no feature of these families"},
                warnings=[f"No feature of the families {families} in '{features}': skipped (no run to consolidate)."],
            )
        # Small populations: room for one group on which two methods can agree.
        floor = 2 * cfg.min_group_size
        if len(matrix) * contamination < floor <= len(matrix) / 2:
            contamination = floor / len(matrix)
        scores = outlier_scores(
            matrix.to_numpy(), method, params or {}, int(seed), cfg.lof_max_rows, fit_max_rows=cfg.fit_max_rows
        )
    except (ArtifactNotFoundError, ValueError, DetectionError) as exc:
        return ToolResult.error(str(exc))
    flagged = flag_top(scores, contamination)
    table = pd.DataFrame({entity: matrix.index, "score": scores, "flagged": flagged})
    run_id = _save_run(
        "outlier",
        method,
        table,
        [features],
        {
            "entity": entity,
            "features": features,
            "method_params": params or {},
            "contamination": contamination,
            "seed": seed,
            "families": sorted(families) if families else None,
        },
    )
    q = np.quantile(scores, [0.5, 0.99])
    return ToolResult.ok(
        {
            "method": method,
            "families": sorted(families) if families else None,
            "features_used": matrix.shape[1],
            "entities": len(table),
            "flagged": int(flagged.sum()),
            "score_p50": round(float(q[0]), 4),
            "score_p99": round(float(q[1]), 4),
        },
        artifact_id=run_id,
    )


@tool
def run_clustering(
    features: str,
    method: ClusterMethod,
    params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> ToolResult:
    """Cluster entities; small clusters are candidate anomalous groups (HDBSCAN noise is label -1).

    Prefer the PCA embedding (spec.features.embedding) as input for speed.

    Args:
        features: Embedding or scaled feature table.
        method: hdbscan (default parameters: min_cluster_size 10), gmm or kmeans.
        params: Method parameters, e.g. {"min_cluster_size": 10} or {"n_clusters": 12}.
        seed: Random seed (defaults to the run seed).
    """
    cfg = _cfg()
    try:
        matrix, entity = _load_matrix(features)
        labels = cluster_labels(matrix.to_numpy(), method, params or {}, int(seed), max_rows=cfg.cluster_max_rows)
    except (ArtifactNotFoundError, ValueError, DetectionError) as exc:
        return ToolResult.error(str(exc))
    sizes = pd.Series(labels).value_counts()
    limit = cfg.max_cluster_share * len(labels)
    small = sizes[(sizes.index != -1) & (sizes >= cfg.min_group_size) & (sizes <= limit)]
    flagged = np.isin(labels, small.index.to_numpy())
    table = pd.DataFrame(
        {
            entity: matrix.index,
            "label": labels,
            "flagged": flagged,
            "score": pd.Series(labels).map(lambda lab: 1.0 / sizes[lab] if lab != -1 else 0.0),
        }
    )
    run_id = _save_run(
        "cluster",
        method,
        table,
        [features],
        {"entity": entity, "features": features, "method_params": params or {}, "seed": seed},
    )
    plot_id = _cluster_plot(run_id)
    return ToolResult.ok(
        {
            "method": method,
            "plot": plot_id,
            "entities": len(labels),
            "clusters": int((sizes.index != -1).sum()),
            "noise": int(sizes.get(-1, 0)),
            "largest_sizes": sizes[sizes.index != -1].head(5).tolist(),
            "small_clusters": len(small),
            "small_cluster_sizes": small.tolist()[:10],
        },
        artifact_id=run_id,
    )


def _cluster_plot(run_id: str) -> str | None:
    """Cluster map of a clustering run (best effort: a plotting problem never fails detection)."""
    from jellybean.skills._shared.plots import cluster_run_plot

    cfg = _cfg()
    try:
        return cluster_run_plot(current_run(), run_id, cfg.max_cluster_share, cfg.min_group_size).id
    except Exception:
        return None


@tool
def plot_cluster_run(run_id: str) -> ToolResult:
    """Draw the cluster map of a clustering run: entities on the PCA plane coloured by cluster.

    Noise is grey; small clusters (candidate anomalous groups) are highlighted
    and labelled with their size. run_clustering already does this; use it for
    older runs.

    Args:
        run_id: Detection artifact id returned by run_clustering.
    """
    from jellybean.skills._shared.plots import cluster_run_plot

    cfg = _cfg()
    try:
        record = cluster_run_plot(current_run(), run_id, cfg.max_cluster_share, cfg.min_group_size)
    except (ArtifactNotFoundError, ValueError, KeyError) as exc:
        return ToolResult.error(f"Cannot plot '{run_id}': {exc}")
    return ToolResult.ok({"run": run_id, "plot": record.id}, artifact_id=record.id)


@tool
def run_graph_analysis(graph: str, method: GraphMethod = "components", seed: int | None = None) -> ToolResult:
    """Analyse the entity graph: connected components, communities (Louvain) or centrality.

    Groups (components/communities) with at least `min_group_size` members are flagged.

    Args:
        graph: Graph edges artifact (spec.features.graph).
        method: components, communities or centrality.
        seed: Seed for community detection.
    """
    store = _store()
    try:
        record = store.get(graph)
        path = store.path_of(graph)
        columns = set(pq.read_schema(path).names)
    except (ArtifactNotFoundError, FileNotFoundError):
        return ToolResult.error(f"Unknown graph artifact '{graph}'.")
    entity = record.params.get("entity")
    if not entity or not {"src", "dst"} <= columns:
        return ToolResult.error(f"'{graph}' is not a graph edges artifact.")
    min_size = _cfg().min_group_size
    if method == "components":
        # Components need no graph library: a sparse matrix of integer codes handles millions of edges.
        table = component_table(pd.read_parquet(path, columns=["src", "dst"]), entity)
        table["flagged"] = table["group_size"] >= min_size
        table["score"] = table["group_size"].astype(float)
        run_id = _save_run("graph", method, table, [graph], {"entity": entity, "graph": graph, "seed": seed})
        sizes = table.drop_duplicates("group")["group_size"].tolist()
        return ToolResult.ok(
            {
                "method": method,
                "nodes": len(table),
                "groups": len(sizes),
                "flagged_groups": sum(1 for s in sizes if s >= min_size),
                "largest_sizes": sizes[:8],
            },
            artifact_id=run_id,
        )
    import networkx as nx

    n_edges = int(pq.read_metadata(path).num_rows)
    if n_edges > _cfg().graph_max_edges:
        return ToolResult.error(
            f"The graph has {n_edges} edges, more than detection.graph_max_edges ({_cfg().graph_max_edges}): "
            f"'{method}' would not finish. Use method 'components', or build the graph with a higher minimum weight."
        )
    edges = pd.read_parquet(path)
    g = nx.Graph()
    g.add_weighted_edges_from(edges[["src", "dst", "weight"]].itertuples(index=False, name=None))
    if method == "communities":
        groups = [sorted(c) for c in nx.community.louvain_communities(g, weight="weight", seed=seed)]
    else:
        centrality = nx.degree_centrality(g)
        table = pd.DataFrame({entity: list(centrality), "score": list(centrality.values())})
        table["flagged"] = flag_top(table["score"].to_numpy(), 0.05) if len(table) else []
        table["group"] = -1
        run_id = _save_run("graph", method, table, [graph], {"entity": entity, "graph": graph, "seed": seed})
        return ToolResult.ok(
            {"method": method, "nodes": len(table), "flagged": int(table["flagged"].sum())}, artifact_id=run_id
        )
    groups.sort(key=lambda c: (-len(c), str(c[0])))
    rows = [(node, gid, len(members)) for gid, members in enumerate(groups, 1) for node in members]
    table = pd.DataFrame(rows, columns=[entity, "group", "group_size"])
    table["flagged"] = table["group_size"] >= min_size
    table["score"] = table["group_size"].astype(float)
    run_id = _save_run("graph", method, table, [graph], {"entity": entity, "graph": graph, "seed": seed})
    sizes = [len(c) for c in groups]
    return ToolResult.ok(
        {
            "method": method,
            "nodes": g.number_of_nodes(),
            "groups": len(groups),
            "flagged_groups": sum(1 for s in sizes if s >= min_size),
            "largest_sizes": sizes[:8],
        },
        artifact_id=run_id,
    )


@tool
def run_timeseries_anomaly(
    input: str,
    time_column: str,
    measure: str = "count",
    group_by: str | None = None,
    granularity: Literal["hour", "day"] = "hour",
    threshold: float | None = None,
) -> ToolResult:
    """Find anomalous points in time series: residuals against a seasonal (hour-of-day) baseline.

    Aggregates the event table per time bucket (and per group), compares each
    point with the median of the same hour and weekday type, and flags robust
    z-scores above the threshold. Consecutive anomalous points form windows.

    Args:
        input: Event table (enriched artifact or dataset).
        time_column: Event time column.
        measure: "count" (events) or a numeric column to sum.
        group_by: Optional column with one series per value (e.g. a location); None = one global series.
        granularity: "hour" or "day".
        threshold: Robust z threshold (default from config).
    """
    threshold = threshold or _cfg().ts_threshold
    try:
        source = resolve_source(input)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        schema = dict(describe(con, source.view))
        for col in [time_column, group_by] + ([] if measure == "count" else [measure]):
            if col and col not in schema:
                return ToolResult.error(f"Unknown column '{col}'.")
        if measure != "count" and type_family(schema[measure]) not in ("integer", "float"):
            return ToolResult.error(f"Measure '{measure}' is not numeric.")
        t, g = quote_ident(time_column), quote_ident(group_by) if group_by else "'__all__'"
        value = "count(*)" if measure == "count" else f"sum({quote_ident(measure)})"
        n_groups = con.execute(f"SELECT count(DISTINCT {g}) FROM {source.view}").fetchall()[0][0]
        n_buckets = con.execute(
            f"SELECT count(DISTINCT date_trunc('{granularity}', {t})) FROM {source.view}"
        ).fetchall()[0][0]
        if n_groups * n_buckets > _MAX_SERIES_POINTS:
            return ToolResult.error(
                f"{n_groups} groups x {n_buckets} buckets is too many series points; "
                "group by a lower-cardinality column."
            )
        series = con.execute(
            f"SELECT CAST({g} AS VARCHAR) AS \"group\", date_trunc('{granularity}', {t}) AS ts, {value} AS value, "
            f"count(*) AS n "
            f"FROM {source.view} WHERE {t} IS NOT NULL {f'AND {g} IS NOT NULL' if group_by else ''} GROUP BY 1, 2"
        ).df()
    finally:
        con.close()
    step = pd.Timedelta(hours=1) if granularity == "hour" else pd.Timedelta(days=1)
    series["value"] = series["value"].astype(float)
    if group_by:
        points = grouped_series_anomalies(series, step, threshold, sums=measure != "count")
        n_points = int(series.groupby("group")["ts"].agg(lambda t: (t.max() - t.min()) / step + 1).sum())
    else:
        # One series: complete the grid with zeros so that silent buckets count as observations.
        full_ts = pd.date_range(series["ts"].min(), series["ts"].max(), freq=step)
        grid = pd.MultiIndex.from_product([series["group"].unique(), full_ts], names=["group", "ts"])
        series = series.set_index(["group", "ts"]).reindex(grid, fill_value=0).reset_index()
        unit = event_unit(series) if measure != "count" else None
        points = seasonal_residual_anomalies(series, threshold, unit=unit)
        n_points = len(series)
    windows = merge_windows(points, step)
    windows.sort(key=lambda w: -w["max_z"])
    store = _store()
    points_table = points.rename(columns={"group": "group"}).reset_index(drop=True)
    masker = Masker(current_run().settings.config.masking, salt=current_run().run_id)
    sensitive = bool(group_by) and masker.is_sensitive_name(group_by)
    window_json = [
        {
            **w,
            "start": str(w["start"]),
            "end": str(w["end"]),
            "max_z": round(float(w["max_z"]), 2),
            "value": float(w["value"]),
            "expected": round(float(w["expected"]), 2),
        }
        for w in windows
    ]
    run_id = store.save_dataframe(
        f"detection_ts_{prefix_part(measure)}",
        points_table,
        parents=[source.name],
        params={
            "kind": "timeseries",
            "method": "seasonal_residual",
            "input": source.name,
            "time_column": time_column,
            "measure": measure,
            "group_by": group_by,
            "granularity": granularity,
            "threshold": threshold,
            "windows": window_json[:200],
            "entity": group_by or "__window__",
            "model": "grouped" if group_by else "single",
        },
    ).id
    shown = [
        {
            "group": masker.mask_value(w["group"]) if sensitive else w["group"],
            "start": w["start"],
            "end": w["end"],
            "max_z": w["max_z"],
            "value": w["value"],
            "expected": w["expected"],
        }
        for w in window_json[:5]
    ]
    # An empty table of anomalous points is a result, not a failure: say which.
    warnings = []
    if n_buckets < _MIN_SERIES_BUCKETS:
        warnings.append(
            f"Only {n_buckets} {granularity} buckets: too few to tell what is usual, so nothing can stand out. "
            + ("Use granularity 'hour'." if granularity == "day" else "The period is too short for this test.")
        )
    if len(points):
        outcome = f"{len(points)} anomalous {granularity} buckets in {len(windows)} windows"
    elif warnings:
        outcome = "no anomalous buckets could be found: too few time buckets (see warnings)"
    else:
        outcome = (
            f"no anomalous buckets: none of the {n_points} {granularity} buckets is {threshold:g} robust "
            "deviations above its usual value. This is a valid result (the run has no rows)."
        )
    return ToolResult.ok(
        {
            "measure": measure,
            "result": outcome,
            "series": int(series["group"].nunique()),
            "points": n_points,
            "anomalous_points": len(points),
            "windows": len(windows),
            "bursts": int((points["test"] == "burst").sum()) if "test" in points.columns else 0,
            "top_windows": shown,
        },
        artifact_id=run_id,
        warnings=warnings,
    )


@tool
def run_contextual_outliers(
    features: str,
    context_column: str,
    method: Literal["context_defining", "all_features"] = "context_defining",
    n_features: Annotated[int, Field(ge=2, le=30)] = 8,
    contamination: Annotated[float | None, Field(gt=0, lt=0.5)] = None,
    min_support_quantile: Annotated[float, Field(ge=0, lt=0.5)] = 0.10,
) -> ToolResult:
    """Flag entities that are unusual **within their context group** (e.g. within their segment).

    Uses the raw feature table, robustly standardised inside each value of
    `context_column` (an attribute such as attr_segment). The default method
    scores entities on the features that best separate the contexts, finding
    entities that behave like another context (e.g. a basic plan used like a
    business plan).

    Args:
        features: Raw feature table (spec.features.table), which keeps the attributes.
        context_column: Attribute column defining the groups, e.g. "attr_segment".
        method: "context_defining" (default) or "all_features".
        n_features: Number of context-defining features (method context_defining).
        contamination: Share flagged inside each group (default detection.contextual_contamination).
        min_support_quantile: Low-support entities (bottom quantile AND below half the median n_events) are
            never flagged (0 = off).
    """
    contamination = contamination or _cfg().contextual_contamination
    store = _store()
    try:
        entity = store.get(features).params.get("entity")
        path = store.path_of(features)
        available = list(pq.read_schema(path).names)
    except (ArtifactNotFoundError, FileNotFoundError):
        return ToolResult.error(f"Unknown feature table '{features}'.")
    if not entity:
        return ToolResult.error(f"'{features}' is not a feature table.")
    if context_column not in available:
        attrs = [c for c in available if c.startswith("attr_")]
        return ToolResult.error(f"Unknown context column '{context_column}'. Attributes: {attrs}")
    # The other attributes are not used: only the context and the numeric features are read.
    wanted = {entity, context_column, *numeric_columns(path, entity)}
    raw = read_features(path, entity, columns=[c for c in available if c in wanted])
    groups = raw[context_column].astype("string").fillna("__missing__")
    try:
        scores, z = contextual_scores(raw, entity, context_column, method=method, n_features=n_features)
    except DetectionError as exc:
        return ToolResult.error(str(exc))
    # Shares and rates computed from very few events are noisy: low-support
    # entities (see detection.low_support) are not flagged as contextual outliers.
    if min_support_quantile and "n_events" in raw.columns:
        scores = scores.where(~low_support(raw["n_events"], min_support_quantile), 0.0)
    flagged = flag_within_groups(scores, groups, contamination)
    table = pd.DataFrame(
        {entity: raw[entity], "score": scores.to_numpy(), "flagged": flagged.to_numpy(), "context": groups.to_numpy()}
    )
    # Keep the per-feature z-scores: consolidation groups contextual outliers on them.
    table = pd.concat([table, z.add_prefix("z__").reset_index(drop=True)], axis=1)
    run_id = _save_run(
        "contextual",
        f"contextual_{method}",
        table,
        [features],
        {
            "entity": entity,
            "features": features,
            "context_column": context_column,
            "method_name": method,
            "n_features": n_features,
            "contamination": contamination,
            "min_support_quantile": min_support_quantile,
            "context_features": list(z.columns),
        },
    )
    per_group = table.groupby("context")["flagged"].agg(["size", "sum"]).reset_index()
    return ToolResult.ok(
        {
            "context_column": context_column,
            "method": method,
            "context_features": list(z.columns),
            "groups": len(per_group),
            "flagged": int(flagged.sum()),
            "per_group": [[r.context, int(r.size), int(r.sum)] for r in per_group.itertuples()][:12],
        },
        artifact_id=run_id,
    )


# --------------------------------------------------------------------------- #
# consolidate_findings                                                        #
# --------------------------------------------------------------------------- #
def _group_entities(matrix: pd.DataFrame, pool: list[Any], distance: float, min_size: int) -> list[list[Any]]:
    """Group pooled entities by the direction of their deviation (cosine, average linkage).

    The grouping keeps a distance for every pair of entities: above
    ``detection.group_max_pool`` only the most deviating entities are grouped.
    """
    if len(pool) < min_size:
        return []
    from sklearn.cluster import AgglomerativeClustering

    X = matrix.loc[pool].clip(-20, 20).to_numpy()
    norms = np.linalg.norm(X, axis=1)
    keep = norms > 0
    cap = _cfg().group_max_pool
    if int(keep.sum()) > cap:
        # Stable choice: the `cap` largest deviations (ties broken by position).
        keep &= np.isin(np.arange(len(pool)), np.argsort(-norms, kind="stable")[:cap])
    X, members = X[keep] / norms[keep, None], [p for p, k in zip(pool, keep, strict=True) if k]
    if len(members) < 2:
        return []
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=distance
    ).fit_predict(X)
    groups = pd.Series(members).groupby(labels).apply(list).tolist()
    return [g for g in groups if len(g) >= min_size]


_TYPE_PRIORITY = {"community": 0, "cluster": 1, "outlier_group": 2, "temporal": 3}


@tool
def consolidate_findings(run_ids: Annotated[list[str], Field(min_length=1)]) -> ToolResult:
    """Combine detection runs into candidate findings, written to findings/ (one JSON per finding).

    Outlier runs: entities flagged by at least `min_votes` methods are pooled
    and grouped by the direction of their deviation. Contextual runs: entities
    unusual only within their context. Clusters: small clusters. Graph: groups.
    Time series: anomalous windows. Overlapping findings are merged (their
    methods combined). Requires at least two outlier runs and one clustering run.

    Args:
        run_ids: Detection artifact ids returned by the run_* tools.
    """
    cfg = _cfg()
    store = _store()
    try:
        runs = [(rid, store.get(rid)) for rid in run_ids]
    except ArtifactNotFoundError as exc:
        return ToolResult.error(f"Unknown run: {exc}")
    kinds: dict[str, list[tuple[str, Any]]] = {}
    for rid, record in runs:
        kind = record.params.get("kind")
        if kind is None:
            return ToolResult.error(f"'{rid}' is not a detection run.")
        kinds.setdefault(kind, []).append((rid, record))
    if len(kinds.get("outlier", [])) < 2 or not kinds.get("cluster"):
        return ToolResult.error("Use at least two outlier methods and one clustering method before consolidating.")
    try:
        source = feature_source()
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    entity = source.entity
    scaled_columns = numeric_columns(source.scaled_path, entity)
    n_entities = source.n_rows()
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    def space_of(pool: list[Any], columns: list[str] | None = None) -> pd.DataFrame:
        """Scaled features of the pooled entities only (the grouping reads no other row)."""
        return read_numeric_matrix(source.scaled_path, entity, columns=columns, members=pool)

    def largest(found: list[dict[str, Any]], run: str) -> list[dict[str, Any]]:
        """The largest groups of one run: merging compares every pair of candidates."""
        if len(found) <= cfg.max_candidates_per_run:
            return found
        warnings.append(
            f"Run '{run}' has {len(found)} candidate groups: only the {cfg.max_candidates_per_run} largest are used."
        )
        return sorted(found, key=lambda c: -len(c["members"]))[: cfg.max_candidates_per_run]

    # Outliers: votes across the methods run on the same features (all features,
    # or one family), then grouped by deviation direction in that space.
    by_family: dict[tuple[str, ...], list[tuple[str, Any]]] = {}
    for rid, rec in kinds["outlier"]:
        by_family.setdefault(tuple(rec.params.get("families") or ()), []).append((rid, rec))
    global_pool: set[Any] = set()
    for family, family_runs in sorted(by_family.items(), key=lambda kv: kv[0] != ()):
        outlier_tables = {rid: pd.read_parquet(store.path_of(rid)).set_index(entity) for rid, _ in family_runs}
        votes = sum(t["flagged"].astype(int) for t in outlier_tables.values())
        ranks = sum(t["score"].rank(pct=True) for t in outlier_tables.values()) / len(outlier_tables)
        min_votes = min(cfg.min_votes, len(outlier_tables))
        # Families keep entities pooled elsewhere: overlapping groups are merged below.
        pool = votes[votes >= min_votes].index.tolist()
        space = space_of(pool, family_columns(scaled_columns, list(family)) if family else None)
        suffix = f"[{','.join(family)}]" if family else ""
        methods = [rec.params["method"] + suffix for _, rec in family_runs]
        for group in _group_entities(space, pool, cfg.group_distance, cfg.min_group_size):
            candidates.append(
                {
                    "type": "outlier_group",
                    "members": set(group),
                    "methods": list(methods),
                    "source_runs": list(outlier_tables),
                    "scores": {
                        "mean_votes": round(float(votes.loc[group].mean()), 2),
                        "mean_score_percentile": round(float(ranks.loc[group].mean()), 4),
                    },
                    "context": {"families": list(family)} if family else {},
                }
            )
        if not family:  # only global outliers are excluded from the contextual groups below
            global_pool |= set(pool)

    # Contextual outliers: unusual within their group but not globally. They are
    # grouped on their z-scores over the context-defining features.
    for rid, record in kinds.get("contextual", []):
        table = pd.read_parquet(store.path_of(rid)).set_index(entity)
        zcols = [c for c in table.columns if c.startswith("z__")]
        flagged = table[table["flagged"] & ~table.index.isin(global_pool)]
        zmatrix = table[zcols] if zcols else space_of(flagged.index.tolist())
        for value, block in flagged.groupby("context"):
            for group in _group_entities(zmatrix, block.index.tolist(), cfg.group_distance, cfg.min_group_size):
                candidates.append(
                    {
                        "type": "outlier_group",
                        "members": set(group),
                        "methods": [f"contextual:{record.params['context_column']}"],
                        "source_runs": [rid],
                        "scores": {"mean_context_score": round(float(table.loc[group, "score"].mean()), 3)},
                        "context": {
                            "column": record.params["context_column"],
                            "value": str(value),
                            "contextual_only": True,
                        },
                    }
                )

    # Small clusters.
    for rid, record in kinds["cluster"]:
        table = pd.read_parquet(store.path_of(rid), columns=[entity, "label", "flagged"])
        found = [
            {
                "type": "cluster",
                "members": set(block[entity]),
                "methods": [record.params["method"]],
                "source_runs": [rid],
                "scores": {"cluster_share": round(len(block) / n_entities, 5)},
                "context": {"label": int(label)},
            }
            for label, block in table[table["flagged"]].groupby("label")
        ]
        candidates += largest(found, rid)

    # Graph groups.
    for rid, record in kinds.get("graph", []):
        table = pd.read_parquet(store.path_of(rid))
        if "group_size" not in table.columns:
            continue
        found = [
            {
                "type": "community",
                "members": set(block[entity]),
                "methods": [f"graph:{record.params['method']}"],
                "source_runs": [rid],
                "scores": {"group_size": len(block)},
                "context": {"group": int(gid)},
            }
            for gid, block in table[table["flagged"]].groupby("group")
        ]
        candidates += largest(found, rid)

    # Merge overlapping entity findings, keeping the highest-priority type. Repeat
    # until stable: a merge can create new overlaps.
    merged = sorted(candidates, key=lambda c: (_TYPE_PRIORITY[c["type"]], -len(c["members"])))
    changed = True
    while changed:
        changed = False
        result: list[dict[str, Any]] = []
        for cand in merged:
            for existing in result:
                # Overlap coefficient: a group mostly contained in another is merged
                # into it (e.g. graph components inside a cluster of the same entities).
                inter = len(cand["members"] & existing["members"])
                smaller = min(len(cand["members"]), len(existing["members"]))
                # Groups found in different feature spaces are different evidence: not merged.
                same_space = cand["context"].get("families") == existing["context"].get("families")
                if smaller and same_space and inter / smaller >= cfg.merge_overlap:
                    if _TYPE_PRIORITY[cand["type"]] < _TYPE_PRIORITY[existing["type"]]:
                        existing["type"] = cand["type"]
                    existing["members"] |= cand["members"]
                    existing["methods"] = sorted(set(existing["methods"]) | set(cand["methods"]))
                    existing["source_runs"] = sorted(set(existing["source_runs"]) | set(cand["source_runs"]))
                    existing["scores"] |= cand["scores"]
                    existing["context"] |= {k: v for k, v in cand["context"].items() if k not in existing["context"]}
                    changed = True
                    break
            else:
                result.append(cand)
        merged = result

    # Temporal windows (one finding per series, with all its windows).
    temporal: list[dict[str, Any]] = []
    for rid, record in kinds.get("timeseries", []):
        by_group: dict[str, list[dict[str, Any]]] = {}
        for w in record.params.get("windows", []):
            by_group.setdefault(w["group"], []).append(w)
        for group, windows in by_group.items():
            temporal.append(
                {
                    "type": "temporal",
                    "members": {group},
                    "windows": windows,
                    "methods": [f"timeseries:{record.params['measure']}"],
                    "source_runs": [rid],
                    "scores": {"max_z": max(w["max_z"] for w in windows), "windows": len(windows)},
                    "context": {
                        "measure": record.params["measure"],
                        "group_by": record.params["group_by"],
                        "granularity": record.params["granularity"],
                        "time_column": record.params["time_column"],
                        "input": record.params["input"],
                    },
                }
            )
    temporal.sort(key=lambda c: -c["scores"]["max_z"])

    # Cap the number of findings while keeping every kind of evidence represented:
    # 3 temporal findings, then a round-robin over buckets: multi-method agreement on
    # all features, each feature family (its own kind of evidence), contextual,
    # graph, other.
    def bucket(c: dict[str, Any]) -> str:
        if c["context"].get("contextual_only"):
            return "contextual"
        if c["context"].get("families"):
            return "family:" + ",".join(c["context"]["families"])
        if len(c["methods"]) >= 2:
            return "agreement"
        if c["type"] == "community":
            return "graph"
        return "other"

    buckets: dict[str, list[dict[str, Any]]] = {"agreement": [], "contextual": [], "graph": [], "other": []}
    # Within a bucket: more agreeing methods first, then extreme *and* coherent groups
    # (mean score percentile x log size: a group of dozens beats a trio of extremes).
    for cand in sorted(
        merged,
        key=lambda c: (
            -len(c["methods"]),
            -c["scores"].get("mean_score_percentile", 0.5) * np.log1p(len(c["members"])),
            -len(c["members"]),
        ),
    ):
        buckets.setdefault(bucket(cand), []).append(cand)
    # Feature-family groups have their own budget (the best few of each family), on
    # top of max_findings: runs without families are unchanged.
    family_names = sorted(b for b in buckets if b.startswith("family:"))
    family_chosen: list[dict[str, Any]] = []
    for _ in range(cfg.findings_per_family):
        for name in family_names:
            if len(buckets[name]) > 0 and len(family_chosen) < cfg.max_family_findings:
                family_chosen.append(buckets[name].pop(0))
    order = ["agreement", *family_names, "contextual", "graph", "other"]
    keep_temporal = temporal[: min(3, cfg.max_findings)]
    chosen: list[dict[str, Any]] = []
    while len(chosen) < cfg.max_findings - len(keep_temporal) and any(buckets.values()):
        for name in order:
            if buckets[name] and len(chosen) < cfg.max_findings - len(keep_temporal):
                chosen.append(buckets[name].pop(0))
    chosen = family_chosen + chosen
    # Slots left by entity findings go to more temporal windows (e.g. when the
    # entity is the series group, its windows are the main evidence).
    spare = cfg.max_findings - len(chosen) - len(keep_temporal)
    keep_temporal += temporal[len(keep_temporal) : len(keep_temporal) + max(0, spare)]
    selected = chosen + keep_temporal
    dropped = len(merged) + len(temporal) - len(selected)

    # Previous candidate findings are superseded (kept for audit).
    ctx = current_run()
    existing = all_findings(ctx)
    if existing:
        archive = ctx.findings_dir / f"superseded_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
        archive.mkdir(parents=True, exist_ok=True)
        for f in ctx.findings_dir.glob("F*.json"):
            shutil.move(str(f), archive / f.name)

    summaries = []
    for n, cand in enumerate(selected, 1):
        fid = f"F{n:03d}"
        if cand["type"] == "temporal":
            ids = pd.DataFrame([{"group": w["group"], "start": w["start"], "end": w["end"]} for w in cand["windows"]])
            entity_column = cand["context"]["group_by"] or "__window__"
            top = []
        else:
            ids = pd.DataFrame({entity: sorted(cand["members"], key=str)})
            entity_column = entity
            top = group_effect_sizes(cand["members"])
        ids_record = store.save_dataframe(
            f"finding_{fid.lower()}_entities",
            ids,
            parents=cand["source_runs"],
            params={"finding": fid, "entity": entity_column},
        )
        finding = Finding(
            id=fid,
            type=cand["type"],
            entity_column=entity_column,
            entity_ids_artifact=ids_record.id,
            size=len(cand["members"]) if cand["type"] != "temporal" else len(cand["windows"]),
            methods=cand["methods"],
            scores=cand["scores"],
            top_features=top,
            source_runs=cand["source_runs"],
            context=cand["context"] | ({"windows": cand["windows"][:20]} if cand["type"] == "temporal" else {}),
        )
        save_finding(finding, ctx)
        summaries.append(
            {
                "id": fid,
                "type": finding.type,
                "size": finding.size,
                "methods": finding.methods,
                "top_features": [f"{t['feature']} ({t['direction']}, {t['effect']})" for t in top[:3]],
                **(
                    {"context": {k: v for k, v in finding.context.items() if k != "windows"}} if finding.context else {}
                ),
            }
        )
    patch_spec({"phase": "critique"})
    return ToolResult.ok(
        {"findings": summaries, "pooled_outliers": len(pool), "dropped_by_cap": dropped}, warnings=warnings
    )
