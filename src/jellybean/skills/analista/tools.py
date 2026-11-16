"""Analyst tools (spec section 7.7).

Describe confirmed findings in understandable terms: the features that differ
most from the population (with effect sizes), a shallow decision tree that
explains membership (rules with precision and coverage), representative
exemplars (masked) and plots. The LLM only writes the final description text
(``write_finding``); every number comes from these tools.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import matplotlib

matplotlib.use("Agg")  # headless: plots are only written to files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import Field

from jellybean.runtime.context import current_run
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.findings import (
    Finding,
    FindingNotFoundError,
    feature_source,
    finding_entities,
    group_effect_sizes,
    group_signature,
    load_finding,
    numeric_matrix,
    read_features,
    save_finding,
)
from jellybean.skills._shared.masking import Masker
from jellybean.skills._shared.pattern_registry import add_pattern, registry_key
from jellybean.skills._shared.spec_tools import SpecMissingError, read_spec

_MAX_ENCODED_LEVELS = 100  # attributes with more values are not one-hot encoded for surrogate rules


def _store() -> ArtifactStore:
    return ArtifactStore(current_run())


def _masker() -> Masker:
    ctx = current_run()
    return Masker(ctx.settings.config.masking, salt=ctx.run_id)


def _members(f: Finding) -> set[Any]:
    ids = finding_entities(f)
    return set(ids[f.entity_column]) if f.entity_column in ids.columns else set()


def _load(finding: str) -> Finding:
    return load_finding(finding)


@tool
def describe_group(finding: str) -> ToolResult:
    """Features that differ most between the finding and the population, with robust effect sizes.

    Numeric features: (group median - population median) / population MAD.
    Attributes: largest difference in the share of a value. Temporal findings:
    observed vs expected values of each window.

    Args:
        finding: Finding id.
    """
    try:
        f = _load(finding)
    except FindingNotFoundError as exc:
        return ToolResult.error(str(exc))
    if f.type == "temporal":
        windows = [
            {
                "group": w["group"],
                "start": w["start"],
                "end": w["end"],
                "observed": w["value"],
                "expected": w["expected"],
                "ratio": round(w["value"] / w["expected"], 2) if w["expected"] else None,
                "max_z": w["max_z"],
            }
            for w in f.context.get("windows", [])
        ]
        f.top_features = [
            {
                "feature": f.context.get("measure"),
                "kind": "temporal",
                "direction": "higher",
                "effect": f.scores.get("max_z"),
                "windows": len(windows),
            }
        ]
        save_finding(f)
        masker = _masker()
        group_by = f.context.get("group_by")
        if group_by and masker.is_sensitive_name(group_by):
            windows = [w | {"group": masker.mask_value(w["group"])} for w in windows]
        return ToolResult.ok(
            {
                "finding": f.id,
                "type": f.type,
                "measure": f.context.get("measure"),
                "group_by": group_by,
                "windows": windows[:10],
            }
        )
    members = _members(f)
    try:
        top = group_effect_sizes(members, top=10)
        population = feature_source().n_rows()
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    f.top_features = top
    save_finding(f)
    return ToolResult.ok(
        {"finding": f.id, "type": f.type, "size": len(members), "population": population, "top_features": top}
    )


@tool
def fit_surrogate_rules(
    finding: str, max_depth: Annotated[int, Field(ge=1, le=5)] = 3, seed: int | None = None
) -> ToolResult:
    """Fit a shallow decision tree that separates the finding's members from the rest; return its rules.

    Each rule describes a leaf predicting membership, with its precision (share
    of matched entities that are members) and coverage (share of members it matches).
    On large populations the tree is fitted on the members and a seeded sample of
    the other entities (`detection.surrogate_max_rows`); precision is then
    estimated from that sample. Attributes with more than 100 values are not used.

    Args:
        finding: Finding id.
        max_depth: Maximum depth of the tree (1-5).
        seed: Random seed.
    """
    from sklearn.tree import DecisionTreeClassifier

    try:
        f = _load(finding)
        source = feature_source()
    except (FindingNotFoundError, SpecMissingError) as exc:
        return ToolResult.error(str(exc))
    if f.type == "temporal":
        return ToolResult.error("Surrogate rules apply to entity findings, not to temporal windows.")
    members = _members(f)
    entity = source.entity
    cap = current_run().settings.config.detection.surrogate_max_rows
    others_weight = 1.0  # entities of the population each sampled non-member stands for
    warnings: list[str] = []
    if source.n_rows() - len(members) <= cap:
        raw = pd.read_parquet(source.raw_path)
    else:
        ids = read_features(source.raw_path, entity, columns=[entity])[entity]
        others = ids[~ids.isin(members)].to_numpy()
        chosen = others[np.sort(np.random.default_rng(seed).choice(len(others), cap, replace=False))]
        raw = read_features(source.raw_path, entity, members=[*members, *chosen.tolist()])
        others_weight = len(others) / cap
        warnings.append(
            f"The tree was fitted on the members and a sample of {cap} of the {len(others)} other entities; "
            "precision is estimated from that sample."
        )
    X = raw.drop(columns=[entity])
    numeric = X.select_dtypes(include=["number", "bool"]).astype(float)
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    text = X.select_dtypes(exclude=["number", "bool"])
    wide = [c for c in text.columns if text[c].nunique() > _MAX_ENCODED_LEVELS]
    if wide:
        text = text.drop(columns=wide)
        warnings.append(f"Attributes with more than {_MAX_ENCODED_LEVELS} values were not used: {wide[:5]}.")
    # No categorical attribute (e.g. a dataset without enrichment): numeric features only.
    categorical = (
        pd.get_dummies(text.astype("string").fillna("__missing__"), prefix_sep="=", dtype=float)
        if text.shape[1]
        else pd.DataFrame(index=X.index)
    )
    data = pd.concat([numeric, categorical], axis=1)
    y = raw[entity].isin(members).to_numpy()
    names = list(data.columns)
    rules: list[dict[str, Any]] = []

    def walk(t: Any, node: int, conditions: list[str]) -> None:
        if t.children_left[node] == -1:
            leaf_mask = np.ones(len(data), dtype=bool)
            for cond in conditions:
                feat, op, value = cond.rsplit(" ", 2)
                col = data[feat].to_numpy()
                leaf_mask &= col <= float(value) if op == "<=" else col > float(value)
            hits = (leaf_mask & y).sum()
            matched = hits + (leaf_mask & ~y).sum() * others_weight
            if matched and hits / matched >= 0.3:
                rules.append(
                    {
                        "rule": " AND ".join(conditions) or "(all)",
                        "precision": round(float(hits / matched), 4),
                        "coverage": round(float(hits / y.sum()), 4),
                        "matched": round(matched),
                    }
                )
            return
        feat, thr = names[t.feature[node]], round(float(t.threshold[node]), 4)
        walk(t, t.children_left[node], [*conditions, f"{feat} <= {thr}"])
        walk(t, t.children_right[node], [*conditions, f"{feat} > {thr}"])

    # Two trees: unweighted (pure, smaller leaves) and class-balanced (broader leaves);
    # the best rules of both are kept, ranked by precision x coverage.
    for weight in (None, "balanced"):
        tree = DecisionTreeClassifier(
            max_depth=max_depth, class_weight=weight, min_samples_leaf=3, random_state=seed
        ).fit(data.to_numpy(), y)
        walk(tree.tree_, 0, [])
    unique = {r["rule"]: r for r in rules}
    rules = sorted(unique.values(), key=lambda r: -(r["precision"] * r["coverage"]))
    f.rules = rules[:3]
    save_finding(f)
    return ToolResult.ok(
        {"finding": f.id, "rules": f.rules, "total_coverage": round(sum(r["coverage"] for r in f.rules), 4)},
        warnings=warnings,
    )


@tool
def get_exemplars(finding: str, k: Annotated[int, Field(ge=1, le=10)] = 5) -> ToolResult:
    """Representative members of a finding (closest to the group's median profile), with masked ids.

    Args:
        finding: Finding id.
        k: Number of exemplars (1-10).
    """
    try:
        f = _load(finding)
    except FindingNotFoundError as exc:
        return ToolResult.error(str(exc))
    masker = _masker()
    if f.type == "temporal":
        group_by = f.context.get("group_by")
        sensitive = bool(group_by) and masker.is_sensitive_name(group_by)
        rows = [
            [
                masker.mask_value(w["group"]) if sensitive else w["group"],
                w["start"],
                w["end"],
                w["value"],
                w["expected"],
            ]
            for w in f.context.get("windows", [])[:k]
        ]
        f.exemplars = [dict(zip(["group", "start", "end", "observed", "expected"], r, strict=True)) for r in rows]
        save_finding(f)
        return ToolResult.ok(
            {"finding": f.id, "columns": ["group", "start", "end", "observed", "expected"], "rows": rows}
        )
    try:
        source = feature_source()
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    entity = source.entity
    members = _members(f)
    group = numeric_matrix(read_features(source.scaled_path, entity, members=members), entity)
    distance = ((group - group.median()) ** 2).sum(axis=1).sort_values()
    chosen = distance.index[:k]
    features = [t["feature"] for t in (f.top_features or group_effect_sizes(members))[:5]]
    rows_of = read_features(source.raw_path, entity, columns=[entity, *features], members=chosen)
    table = rows_of.set_index(entity).loc[chosen, features].reset_index()
    rows = [
        [masker.mask_value(r[0]), *[round(v, 4) if isinstance(v, float) else v for v in r[1:]]]
        for r in table.itertuples(index=False, name=None)
    ]
    columns = [entity, *features]
    f.exemplars = [dict(zip(columns, r, strict=True)) for r in rows]
    save_finding(f)
    return ToolResult.ok({"finding": f.id, "columns": columns, "rows": rows})


@tool
def make_plot(finding: str, kind: Literal["feature_distributions", "embedding", "timeseries"]) -> ToolResult:
    """Draw a PNG for a finding and return only its path.

    feature_distributions: histograms of the top 3 features (finding vs population);
    embedding: the PCA embedding with the finding highlighted;
    timeseries: the anomalous series around its windows (temporal findings).

    Args:
        finding: Finding id.
        kind: Plot type.
    """
    # Whatever happens while drawing, no figure stays open (each one holds its data).
    open_before = set(plt.get_fignums())
    try:
        return _make_plot(finding, kind)
    finally:
        for number in set(plt.get_fignums()) - open_before:
            plt.close(number)


def _make_plot(finding: str, kind: str) -> ToolResult:
    try:
        f = _load(finding)
        spec = read_spec()
    except (FindingNotFoundError, SpecMissingError) as exc:
        return ToolResult.error(str(exc))
    store = _store()
    fig = None
    if kind == "timeseries":
        if f.type != "temporal":
            return ToolResult.error("timeseries plots are for temporal findings.")
        from jellybean.skills.critico.tools import _rebuild_series

        params = store.get(f.source_runs[0]).params
        series = _rebuild_series(params)
        groups = {str(w["group"]) for w in f.context.get("windows", [])}
        fig, ax = plt.subplots(figsize=(10, 4))
        for g in sorted(groups)[:3]:
            block = series[series["group"] == g]
            ax.plot(block["ts"], block["value"], label=f"{params['group_by'] or 'all'}={g}", linewidth=0.8)
        for w in f.context.get("windows", []):
            ax.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]), color="red", alpha=0.2)
        ax.set_title(f"{f.id}: {params['measure']} per {params['granularity']}")
        ax.legend(loc="upper left", fontsize=8)
    else:
        if f.type == "temporal":
            return ToolResult.error("Use kind 'timeseries' for temporal findings.")
        try:
            source = feature_source()
        except SpecMissingError as exc:
            return ToolResult.error(str(exc))
        entity = source.entity
        members = _members(f)
        if kind == "embedding":
            if not spec.features.embedding:
                return ToolResult.error("No embedding in the spec.")
            from jellybean.skills._shared.plots import _max_points, _thin

            emb = pd.read_parquet(store.path_of(spec.features.embedding), columns=[entity, "c1", "c2"])
            inside = emb[entity].isin(members)
            # A sample of the population is enough for the background; every member is drawn.
            emb = emb[_thin(emb.index, inside.to_numpy(), _max_points())].reset_index(drop=True)
            inside = emb[entity].isin(members)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(emb.loc[~inside, "c1"], emb.loc[~inside, "c2"], s=2, alpha=0.2, color="grey", label="population")
            ax.scatter(emb.loc[inside, "c1"], emb.loc[inside, "c2"], s=12, color="red", label=f.id)
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.legend()
        else:
            top = [
                t["feature"] for t in (f.top_features or group_effect_sizes(members)) if t.get("kind") != "categorical"
            ][:3]
            raw = read_features(source.raw_path, entity, columns=[entity, *top])
            fig, axes = plt.subplots(1, max(1, len(top)), figsize=(4 * max(1, len(top)), 3.5))
            axes = np.atleast_1d(axes)
            inside = raw[entity].isin(members)
            for ax, feat in zip(axes, top, strict=False):
                values = raw[feat].astype(float)
                bins = np.histogram_bin_edges(values.dropna(), bins=40)
                ax.hist(values[~inside].dropna(), bins=bins, alpha=0.5, density=True, label="population")
                ax.hist(values[inside].dropna(), bins=bins, alpha=0.7, density=True, label=f.id, color="red")
                ax.set_title(feat, fontsize=9)
            axes[0].legend(fontsize=8)
        fig.suptitle(f"{f.id} ({f.type}, n={len(members)})")
    fig.tight_layout()

    def write(path: Any) -> None:
        fig.savefig(path, dpi=110)
        plt.close(fig)

    record = store.create(
        f"plot_{f.id.lower()}_{kind}",
        "plot",
        write,
        parents=[f.entity_ids_artifact],
        params={"finding": f.id, "kind": kind},
    )
    f.plots = [*f.plots, record.path]
    save_finding(f)
    return ToolResult.ok({"finding": f.id, "path": record.path}, artifact_id=record.id)


@tool
def write_finding(finding: str, description: str) -> ToolResult:
    """Complete a confirmed finding with its human-readable description (in the report language).

    Run describe_group (and fit_surrogate_rules for entity findings) first, and
    use only their numbers in the description.

    Args:
        finding: Finding id.
        description: 2-8 sentences: what the group is, how it differs (effect sizes), the rule, why it matters.
    """
    try:
        f = _load(finding)
    except FindingNotFoundError as exc:
        return ToolResult.error(str(exc))
    if f.critic_verdict != "confirmed":
        return ToolResult.error(f"{f.id} is not confirmed by the critic (verdict: {f.critic_verdict}).")
    if not f.top_features:
        return ToolResult.error("Run describe_group first.")
    if not 20 <= len(description) <= 2000:
        return ToolResult.error("The description must have between 20 and 2000 characters.")
    f.description = description.strip()
    save_finding(f)
    return ToolResult.ok({"finding": f.id, "described": True})


@tool
def register_pattern(finding: str) -> ToolResult:
    """Store the finding's signature in the persistent pattern registry (only after positive human feedback).

    Args:
        finding: Finding id with human_feedback "confirmed".
    """
    try:
        f = _load(finding)
        spec = read_spec()
    except (FindingNotFoundError, SpecMissingError) as exc:
        return ToolResult.error(str(exc))
    if f.human_feedback != "confirmed":
        return ToolResult.error(f"{f.id} has no positive human feedback (feedback: {f.human_feedback}).")
    if f.registered_pattern:
        return ToolResult.ok({"finding": f.id, "pattern": f.registered_pattern, "already_registered": True})
    ctx = current_run()
    if f.type == "temporal":
        sig = {
            "dominant_features": [
                {
                    "feature": f"{f.context.get('measure')}@{f.context.get('group_by')}",
                    "direction": "higher",
                    "effect": f.scores.get("max_z", 0),
                }
            ],
            "centroid": {},
        }
    else:
        sig = group_signature(_members(f), f.top_features or None, f.rules)
    pattern_id = add_pattern(
        ctx,
        registry_key(spec),
        {
            "type": f.type,
            "signature": sig,
            "description": f.description,
            "run_id": ctx.run_id,
            "finding": f.id,
            "size": f.size,
            "methods": f.methods,
        },
    )
    f.registered_pattern = pattern_id
    save_finding(f)
    return ToolResult.ok({"finding": f.id, "pattern": pattern_id, "key": registry_key(spec)})
