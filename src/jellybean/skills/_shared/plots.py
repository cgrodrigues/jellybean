# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Overview plots of a run: cluster maps and the findings map.

Both use the same 2D projection of the entities: the first two components of
the PCA embedding recorded in the spec (or a PCA of the scaled features when
there is no embedding). Plots are written as PNG files; agents only receive
artifact ids, never images or paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: plots are only written to files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jellybean.runtime.context import RunContext
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.findings import Finding, finding_entities, numeric_matrix
from jellybean.spec import load_spec

_PALETTE = plt.get_cmap("tab20").colors
_VERDICT_MARKER = {"confirmed": "o", "explained": "D", "artifact": "x", "weak": "^", "unstable": "s", None: "."}


def projection_2d(ctx: RunContext) -> tuple[pd.DataFrame, str] | None:
    """(entity, x, y) for every entity, and the entity column; None without features."""
    spec = load_spec(ctx.spec_path)
    store = ArtifactStore(ctx)
    if spec.features.embedding:
        entity = store.get(spec.features.embedding).params["entity"]
        emb = pd.read_parquet(store.path_of(spec.features.embedding), columns=[entity, "c1", "c2"])
        return emb.rename(columns={"c1": "x", "c2": "y"}), entity
    if spec.features.scaled:
        from sklearn.decomposition import PCA

        scaled = pd.read_parquet(store.path_of(spec.features.scaled))
        entity = store.get(spec.features.scaled).params["entity"]
        matrix = numeric_matrix(scaled, entity)
        xy = PCA(n_components=2, random_state=spec.seed).fit_transform(matrix.to_numpy())
        return pd.DataFrame({entity: matrix.index, "x": xy[:, 0], "y": xy[:, 1]}), entity
    return None


def _max_points(ctx: RunContext | None = None) -> int:
    from jellybean.runtime.context import maybe_current_run
    from jellybean.settings import get_settings

    ctx = ctx or maybe_current_run()
    return (ctx.settings if ctx else get_settings()).config.detection.plot_max_points


def _thin(index: pd.Index, always: np.ndarray, limit: int, seed: int = 0) -> np.ndarray:
    """Boolean mask keeping every ``always`` row and a seeded sample of the others, about ``limit`` in all.

    A scatter of millions of points costs memory and time and shows nothing more
    than a sample of them; the rows that matter (members, small clusters) are all kept.
    """
    others = np.flatnonzero(~always)
    room = max(limit - int(always.sum()), 0)
    keep = always.copy()
    if len(others) > room:
        others = np.random.default_rng(seed).choice(others, size=room, replace=False)
    keep[others] = True
    return keep


def cluster_map(coords: pd.DataFrame, labels: pd.Series, title: str, small: set[int]) -> plt.Figure:
    """Scatter of the entities coloured by cluster; noise grey; small clusters highlighted and labelled.

    Above ``detection.plot_max_points`` the large clusters and the noise are thinned
    (the legend keeps the real sizes); small clusters are always drawn in full.
    """
    sizes = labels.value_counts()
    shown = _thin(labels.index, labels.isin(small).to_numpy(), _max_points())
    coords, labels = coords[shown], labels[shown]
    fig, ax = plt.subplots(figsize=(9, 7))
    try:
        _draw_cluster_map(ax, coords, labels, sizes, title, small)
        fig.tight_layout()
    except BaseException:
        plt.close(fig)  # a failed plot must not stay in pyplot's registry
        raise
    return fig


def _draw_cluster_map(
    ax: Any, coords: pd.DataFrame, labels: pd.Series, sizes: pd.Series, title: str, small: set[int]
) -> None:
    noise = labels == -1
    ax.scatter(
        coords.loc[noise, "x"],
        coords.loc[noise, "y"],
        s=3,
        color="lightgrey",
        alpha=0.5,
        label=f"noise ({int(sizes.get(-1, 0)):,})",
    )
    for i, (label, size) in enumerate(sizes.drop(index=-1, errors="ignore").items()):
        mask = labels == label
        is_small = int(label) in small
        ax.scatter(
            coords.loc[mask, "x"],
            coords.loc[mask, "y"],
            s=18 if is_small else 3,
            alpha=0.9 if is_small else 0.25,
            color=_PALETTE[i % len(_PALETTE)],
            edgecolors="black" if is_small else "none",
            linewidths=0.4,
            label=f"cluster {label} ({int(size):,}){' *' if is_small else ''}",
        )
        if is_small:
            cx, cy = coords.loc[mask, "x"].median(), coords.loc[mask, "y"].median()
            ax.annotate(
                f"{label} (n={int(size)})",
                (cx, cy),
                fontsize=8,
                weight="bold",
                xytext=(6, 6),
                textcoords="offset points",
            )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(title + "\n(* small clusters = candidate anomalous groups)", fontsize=10)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[:16], legend_labels[:16], fontsize=7, loc="best", markerscale=2)


def findings_map(ctx: RunContext, findings: list[Finding]) -> plt.Figure | None:
    """Population in grey with the members of every entity finding coloured and labelled."""
    projection = projection_2d(ctx)
    entity_findings = [f for f in findings if f.type != "temporal"]
    if projection is None or not entity_findings:
        return None
    coords, entity = projection
    coords = coords.set_index(entity)
    # The population is background: a seeded sample of it is enough. Members are drawn from `coords` in full.
    background = coords[_thin(coords.index, np.zeros(len(coords), dtype=bool), _max_points(ctx))]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    try:
        _draw_findings_map(ax, ctx, coords, background, entity_findings)
        fig.tight_layout()
    except BaseException:
        plt.close(fig)
        raise
    return fig


def _draw_findings_map(
    ax: Any, ctx: RunContext, coords: pd.DataFrame, background: pd.DataFrame, entity_findings: list[Finding]
) -> None:
    ax.scatter(
        background["x"], background["y"], s=2, color="lightgrey", alpha=0.4, label=f"population ({len(coords):,})"
    )
    for i, f in enumerate(entity_findings):
        ids = finding_entities(f, ctx)
        members = coords[coords.index.isin(set(ids[ids.columns[0]]))]
        if members.empty:
            continue
        marker = _VERDICT_MARKER.get(f.critic_verdict, ".")
        ax.scatter(
            members["x"],
            members["y"],
            s=22 if f.critic_verdict == "confirmed" else 14,
            marker=marker,
            color=_PALETTE[(2 * i) % len(_PALETTE)],
            alpha=0.9,
            label=f"{f.id} {f.type} n={f.size} [{f.critic_verdict or '?'}]",
        )
        ax.annotate(
            f.id,
            (members["x"].median(), members["y"].median()),
            fontsize=8,
            weight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Findings on the PCA map (o confirmed, x artifact, ^ weak, s unstable)", fontsize=10)
    ax.legend(fontsize=7, loc="best", markerscale=1.5)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cluster_run_plot(ctx: RunContext, run_id: str, max_cluster_share: float, min_group_size: int) -> Any:
    """Create the cluster-map artifact of a clustering detection run; returns its record."""
    store = ArtifactStore(ctx)
    record = store.get(run_id)
    if record.params.get("kind") != "cluster":
        raise ValueError(f"'{run_id}' is not a clustering run.")
    entity = record.params["entity"]
    table = pd.read_parquet(store.path_of(run_id))
    projection = projection_2d(ctx)
    if projection is not None and projection[1] == entity:
        coords = projection[0].set_index(entity).reindex(table[entity]).reset_index()
    else:  # project the clustering input itself
        from sklearn.decomposition import PCA

        features = pd.read_parquet(store.path_of(record.params["features"]))
        matrix = numeric_matrix(features, entity).reindex(table[entity])
        xy = PCA(n_components=2, random_state=0).fit_transform(np.nan_to_num(matrix.to_numpy()))
        coords = pd.DataFrame({entity: table[entity], "x": xy[:, 0], "y": xy[:, 1]})
    labels = table["label"].reset_index(drop=True)
    sizes = labels.value_counts()
    limit = max_cluster_share * len(labels)
    small = {int(lab) for lab, n in sizes.items() if lab != -1 and min_group_size <= n <= limit}
    fig = cluster_map(coords.reset_index(drop=True), labels, f"{record.params['method']} clusters ({run_id})", small)
    return store.create(
        f"plot_clusters_{record.params['method']}",
        "plot",
        lambda p: save_figure(fig, p),
        parents=[run_id],
        params={"kind": "clusters", "run": run_id},
    )
