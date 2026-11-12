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
        emb = pd.read_parquet(store.path_of(spec.features.embedding))
        entity = store.get(spec.features.embedding).params["entity"]
        return emb[[entity, "c1", "c2"]].rename(columns={"c1": "x", "c2": "y"}), entity
    if spec.features.scaled:
        from sklearn.decomposition import PCA

        scaled = pd.read_parquet(store.path_of(spec.features.scaled))
        entity = store.get(spec.features.scaled).params["entity"]
        matrix = numeric_matrix(scaled, entity)
        xy = PCA(n_components=2, random_state=spec.seed).fit_transform(matrix.to_numpy())
        return pd.DataFrame({entity: matrix.index, "x": xy[:, 0], "y": xy[:, 1]}), entity
    return None


def cluster_map(coords: pd.DataFrame, labels: pd.Series, title: str, small: set[int]) -> plt.Figure:
    """Scatter of the entities coloured by cluster; noise grey; small clusters highlighted and labelled."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sizes = labels.value_counts()
    noise = labels == -1
    ax.scatter(
        coords.loc[noise, "x"],
        coords.loc[noise, "y"],
        s=3,
        color="lightgrey",
        alpha=0.5,
        label=f"noise ({int(noise.sum()):,})",
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
    fig.tight_layout()
    return fig


def findings_map(ctx: RunContext, findings: list[Finding]) -> plt.Figure | None:
    """Population in grey with the members of every entity finding coloured and labelled."""
    projection = projection_2d(ctx)
    entity_findings = [f for f in findings if f.type != "temporal"]
    if projection is None or not entity_findings:
        return None
    coords, entity = projection
    coords = coords.set_index(entity)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(coords["x"], coords["y"], s=2, color="lightgrey", alpha=0.4, label=f"population ({len(coords):,})")
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
    fig.tight_layout()
    return fig


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
