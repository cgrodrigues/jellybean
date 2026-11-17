# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Critic tools (spec section 7.6).

The critic tries to **refute** each finding, in an isolated context: it gets
the finding id and artifact ids only, never the detector's reasoning. Every
check is deterministic and records its result in the finding (``checks``), and
:func:`set_verdict` refuses verdicts contradicted by the recorded evidence:

* if ``artifact_check`` suspects a data artefact, the verdict must be ``artifact``;
* ``confirmed`` needs stability at or above ``critic.min_stability``;
* ``explained`` (a real, stable pattern with a legitimate cause visible in the
  reference tables) needs :func:`explain_by_reference` to have found that cause,
  and a finding it explains cannot be ``confirmed``.
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from jellybean.runtime.context import current_run
from jellybean.runtime.tool_registry import ToolResult, run_with_timeout, tool
from jellybean.skills._shared.artifacts import ArtifactNotFoundError, ArtifactStore
from jellybean.skills._shared.catalog import (
    FILE_COLUMN,
    SourceNotFoundError,
    connect,
    dataset_relation_with_files,
    discover_datasets,
    quote_ident,
)
from jellybean.skills._shared.column_stats import describe, type_family
from jellybean.skills._shared.detection import (
    cluster_labels,
    component_table,
    contextual_scores,
    event_unit,
    flag_top,
    flag_within_groups,
    grouped_series_anomalies,
    low_support,
    outlier_scores,
    seasonal_residual_anomalies,
)
from jellybean.skills._shared.findings import (
    Finding,
    FindingNotFoundError,
    family_columns,
    feature_source,
    finding_entities,
    group_signature,
    load_finding,
    numeric_columns,
    read_features,
    read_numeric_matrix,
    save_finding,
)
from jellybean.skills._shared.pattern_registry import best_match, classify_novelty, registry_key
from jellybean.skills._shared.spec_tools import SpecMissingError, read_spec

Verdict = Literal["confirmed", "explained", "artifact", "unstable", "weak"]


def _store() -> ArtifactStore:
    return ArtifactStore(current_run())


def _config():
    return current_run().settings.config


def _load(finding: str) -> Finding:
    return load_finding(finding)


def _record_check(finding: Finding, name: str, result: dict[str, Any]) -> None:
    finding.checks[name] = result
    save_finding(finding)


def _members(finding: Finding) -> set[Any]:
    ids = finding_entities(finding)
    return set(ids[finding.entity_column]) if finding.entity_column in ids.columns else set()


def _runs(finding: Finding) -> list[tuple[str, dict[str, Any]]]:
    store = _store()
    return [(rid, store.get(rid).params) for rid in finding.source_runs]


def _matrix(features: str, columns: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    store = _store()
    entity = store.get(features).params["entity"]
    return read_numeric_matrix(store.path_of(features), entity, columns=columns), entity


# --------------------------------------------------------------------------- #
# stability_check                                                             #
# --------------------------------------------------------------------------- #
# What a detection run finds again on one subsample does not depend on the finding
# being checked: it is computed once per (run, subsample) and reused for every
# finding of the run. Keys start with the run folder; values are sets of entities
# (or the cluster labels).
_REDETECT_CACHE: dict[tuple[Any, ...], Any] = {}
_REDETECT_CACHE_MAX = 64


def clear_redetect_cache() -> None:
    _REDETECT_CACHE.clear()


def _cached(key: tuple[Any, ...] | None, compute: Any) -> Any:
    if key is None:
        return compute()
    key = (str(current_run().run_dir), *key)
    if key not in _REDETECT_CACHE:
        if len(_REDETECT_CACHE) >= _REDETECT_CACHE_MAX:
            _REDETECT_CACHE.clear()
        _REDETECT_CACHE[key] = compute()
    return _REDETECT_CACHE[key]


def _redetect(
    finding: Finding,
    sample: set[Any],
    seed: int,
    sample_key: tuple[Any, ...] | None = None,
    tables: dict[Any, Any] | None = None,
) -> set[Any]:
    """Entities re-detected by the finding's source methods on a subsample.

    ``sample_key`` identifies the subsample (results are then shared between
    findings); ``tables`` keeps the matrices loaded during one stability check.
    """
    cfg = _config().detection
    runs = _runs(finding)
    tables = {} if tables is None else tables

    def loaded(key: Any, load: Any) -> Any:
        if key not in tables:
            tables[key] = load()
        return tables[key]

    def key_of(*parts: Any) -> tuple[Any, ...] | None:
        return None if sample_key is None else (*parts, seed, *sample_key)

    found: set[Any] = set()
    # Outlier runs vote within their own feature space (all features, or one family).
    spaces: dict[tuple[str, ...], list[tuple[str, dict[str, Any]]]] = {}
    for rid, p in runs:
        if p["kind"] == "outlier":
            spaces.setdefault(tuple(p.get("families") or ()), []).append((rid, p))
    for family, space_runs in spaces.items():

        def outliers(family: tuple[str, ...] = family, space_runs: list[Any] = space_runs) -> set[Any]:
            features = space_runs[0][1]["features"]

            def load() -> pd.DataFrame:
                store = _store()
                entity = store.get(features).params["entity"]
                names = numeric_columns(store.path_of(features), entity)
                return _matrix(features, family_columns(names, list(family) or None))[0]

            matrix = loaded(("matrix", features, family), load)
            sub = matrix[matrix.index.isin(sample)]
            votes = np.zeros(len(sub))
            for _, p in space_runs:
                scores = outlier_scores(
                    sub.to_numpy(),
                    p["method"],
                    p.get("method_params", {}),
                    seed,
                    cfg.lof_max_rows,
                    fit_max_rows=cfg.fit_max_rows,
                )
                votes += flag_top(scores, p.get("contamination", cfg.contamination))
            return set(sub.index[votes >= min(cfg.min_votes, len(space_runs))])

        found |= _cached(key_of("outlier", tuple(sorted(rid for rid, _ in space_runs))), outliers)
    for rid, p in runs:
        if p["kind"] == "contextual":

            def contextual(p: dict[str, Any] = p) -> set[Any]:
                raw = loaded(("raw", p["features"]), lambda: pd.read_parquet(_store().path_of(p["features"])))
                raw = raw[raw[p["entity"]].isin(sample)].reset_index(drop=True)
                scores, _ = contextual_scores(
                    raw,
                    p["entity"],
                    p["context_column"],
                    method=p.get("method_name", "context_defining"),
                    n_features=p.get("n_features", 8),
                )
                groups = raw[p["context_column"]].astype("string").fillna("__missing__")
                quantile = p.get("min_support_quantile", 0.0)
                if quantile and "n_events" in raw.columns:
                    scores = scores.where(~low_support(raw["n_events"], quantile), 0.0)
                flags = flag_within_groups(scores, groups, p.get("contamination", cfg.contamination))
                return set(raw.loc[flags.to_numpy(), p["entity"]])

            found |= _cached(key_of("contextual", rid), contextual)
        elif p["kind"] == "cluster":

            def clusters(p: dict[str, Any] = p) -> pd.Series:
                matrix = loaded(("matrix", p["features"], ()), lambda: _matrix(p["features"])[0])
                sub = matrix[matrix.index.isin(sample)]
                return pd.Series(
                    cluster_labels(
                        sub.to_numpy(), p["method"], p.get("method_params", {}), seed, max_rows=cfg.cluster_max_rows
                    ),
                    index=sub.index,
                )

            labels = _cached(key_of("cluster", rid), clusters)
            # The cluster that best matches the finding's members.
            inside = labels[labels.index.isin(_members(finding))]
            inside = inside[inside != -1]
            if not inside.empty:
                found |= set(labels.index[labels == inside.mode().iloc[0]])
        elif p["kind"] == "graph" and p.get("method") in ("components", "communities"):

            def components(p: dict[str, Any] = p) -> set[Any]:
                edges = loaded(
                    ("edges", p["graph"]), lambda: pd.read_parquet(_store().path_of(p["graph"]), columns=["src", "dst"])
                )
                edges = edges.sample(frac=0.8, random_state=seed)
                edges = edges[edges["src"].isin(sample) & edges["dst"].isin(sample)]
                groups = component_table(edges, "node")
                return set(groups.loc[groups["group_size"] >= cfg.min_group_size, "node"])

            found |= _cached(key_of("graph", rid), components)
    return found


@tool
def stability_check(
    finding: str,
    n_runs: Annotated[int | None, Field(ge=2, le=20)] = None,
    subsample: Annotated[float, Field(gt=0.3, lt=1.0)] = 0.8,
    seed: int | None = None,
) -> ToolResult:
    """Repeat the finding's detection with other seeds and subsamples and measure how much of it comes back.

    Entity findings: share of the finding's members (present in each subsample)
    re-detected by the same methods, plus the Jaccard with the re-detected set
    of clusters/graph groups. Temporal findings: share of the anomalous points
    still flagged when the seasonal baseline uses a random 70% of the days.

    Args:
        finding: Finding id.
        n_runs: Repetitions (default from config).
        subsample: Share of entities kept in each repetition.
        seed: Base seed (defaults to the run seed).
    """
    try:
        f = _load(finding)
    except FindingNotFoundError as exc:
        return ToolResult.error(str(exc))
    cfg = _config()
    n_runs = n_runs or cfg.critic.stability_runs
    requested = n_runs
    rng = np.random.default_rng(seed)
    recalls, jaccards = [], []
    # The check must always leave an answer. It works inside a time budget below the tool
    # limit and keeps what it measured; and it is recorded as attempted before any work,
    # so that even a check killed at the tool limit does not block the verdict.
    limit = cfg.limits.tool_timeouts.get("stability_check") or cfg.limits.tool_timeout_s
    budget = cfg.critic.stability_budget_share * limit
    started = time.monotonic()
    attempt: dict[str, Any] = {
        "stability": None,
        "reason": "timeout",
        "runs_completed": 0,
        "runs_requested": requested,
        "subsample": subsample,
    }
    _record_check(f, "stability", attempt)
    completed, stopped_by, sample_rows = 0, None, None
    durations: list[float] = []

    def out_of_time() -> bool:
        remaining = budget - (time.monotonic() - started)
        return remaining <= 0 or bool(durations and max(durations) > remaining)

    if f.type == "temporal":
        rid, p = _runs(f)[0]
        points = pd.read_parquet(_store().path_of(rid))
        if not len(points):
            return _unmeasured(f, attempt, "no anomalous points stored for this temporal finding")
        windows = f.context.get("windows", [])
        group = {str(w["group"]) for w in windows}
        series = _rebuild_series(p)
        step = pd.Timedelta(hours=1) if p["granularity"] == "hour" else pd.Timedelta(days=1)
        days = series["ts"].dt.normalize().unique()
        groups = series["group"].unique()
        mine = points[points["group"].astype(str).isin(group)]
        for _ in range(n_runs):
            if out_of_time():
                stopped_by = "time budget"
                break
            tick = time.monotonic()
            keep = set(rng.choice(days, max(2, int(len(days) * 0.7)), replace=False))
            if p.get("model") == "grouped":
                # Same model as the detector; bursts are compared with 70% of the other groups.
                others = set(rng.choice(groups, max(2, int(len(groups) * 0.7)), replace=False)) | group
                again = grouped_series_anomalies(
                    series, step, p["threshold"], keep_days=keep, keep_groups=others, sums=p["measure"] != "count"
                )
                again = again[again["group"].isin(group)]
            else:
                unit = event_unit(series) if p["measure"] != "count" else None
                again = seasonal_residual_anomalies(
                    series[series["group"].isin(group)], p["threshold"], keep_days=keep, unit=unit
                )
            hits = mine.merge(again[["group", "ts"]], on=["group", "ts"], how="inner")
            recalls.append(len(hits) / len(mine) if len(mine) else 0.0)
            completed += 1
            durations.append(time.monotonic() - tick)
    else:
        members = _members(f)
        try:
            source = feature_source()
        except SpecMissingError as exc:
            return ToolResult.error(str(exc))
        ids = read_features(source.scaled_path, source.entity, columns=[source.entity])[source.entity]
        full_rows = len(ids)
        # Large populations: the finding is re-detected among a seeded sample of the entities
        # plus its own members. Each repetition then costs about the same whatever the size
        # of the population, and quadratic methods (LOF, HDBSCAN) stay cheap.
        cap = cfg.critic.stability_max_rows
        estimate = _repetition_estimate(f, full_rows)
        if estimate and estimate * min(cap / full_rows, 1.0) > budget:
            cap = max(_MIN_STABILITY_ROWS, int(cap * budget / (estimate * min(cap / full_rows, 1.0)) * 0.8))
        capped = full_rows > cap
        if capped:
            chosen = np.sort(np.random.default_rng(seed).choice(full_rows, cap, replace=False))
            keep_rows = np.zeros(full_rows, dtype=bool)
            keep_rows[chosen] = True
            keep_rows |= ids.isin(members).to_numpy()
            ids = ids[keep_rows]
            sample_rows = len(ids)
        if estimate:  # repetitions that fit the budget, from what the detector's methods took
            per_repetition = estimate * len(ids) / full_rows
            n_runs = max(1, min(n_runs, int(budget // max(per_repetition, 1e-9))))
        population = ids.tolist()
        tables: dict[Any, Any] = {}
        for i in range(n_runs):
            if out_of_time():
                stopped_by = "time budget"
                break
            tick = time.monotonic()
            sample = set(rng.choice(population, int(len(population) * subsample), replace=False))
            sampled_members = members & sample
            if not sampled_members:
                continue
            # The i-th subsample is the same for every finding checked with this seed and share
            # (not when the sample holds the finding's own members).
            key = None if capped else (str(source.scaled_path), seed, subsample, i)
            try:
                found = run_with_timeout(
                    _redetect,
                    {"finding": f, "sample": sample, "seed": int(seed) + i + 1, "sample_key": key, "tables": tables},
                    budget - (time.monotonic() - started),
                    "stability-repetition",
                )
            except TimeoutError:
                stopped_by = "timeout"
                break
            recalls.append(len(sampled_members & found) / len(sampled_members))
            union = len(found | sampled_members)
            jaccards.append(len(found & sampled_members) / union if union else 0.0)
            completed += 1
            durations.append(time.monotonic() - tick)
        if stopped_by is None and n_runs < requested:
            stopped_by = "time budget"
    extra: dict[str, Any] = {"runs_completed": completed, "runs_requested": requested, "subsample": subsample}
    if sample_rows is not None:
        extra["sample_rows"] = sample_rows
    if stopped_by:
        extra["stopped_by"] = stopped_by
    if not recalls:
        reason = "timeout" if stopped_by else "no members of the finding in the subsamples"
        return _unmeasured(f, attempt | extra, reason)
    result = {
        "stability": round(float(np.mean(recalls)), 4),
        "stability_std": round(float(np.std(recalls)), 4),
        "runs": len(recalls),
        "mean_jaccard": round(float(np.mean(jaccards)), 4) if jaccards else None,
        **extra,
    }
    _record_check(f, "stability", result)
    warnings = (
        [f"Stopped after {completed} of {requested} repetitions ({stopped_by}): the result uses those."]
        if stopped_by
        else []
    )
    return ToolResult.ok(
        {"finding": f.id, **result, "min_stability_for_confirmed": cfg.critic.min_stability}, warnings=warnings
    )


_MIN_STABILITY_ROWS = 5000


def _unmeasured(f: Finding, attempt: dict[str, Any], reason: str) -> ToolResult:
    """Record that stability was attempted and could not be measured; the verdict is not blocked."""
    result = attempt | {"stability": None, "reason": reason}
    _record_check(f, "stability", result)
    return ToolResult.ok(
        {"finding": f.id, **result},
        warnings=[
            f"Stability could not be measured ({reason}). Do not call stability_check again for this finding: "
            "continue with the other checks and set the verdict to 'weak' (or 'unstable'); 'confirmed' and "
            "'explained' are not possible without a measured stability."
        ],
    )


def _repetition_estimate(f: Finding, rows: int) -> float | None:
    """Seconds one repetition on all ``rows`` would take, from the durations the detector recorded
    for the finding's own runs (None when the trace does not have them)."""
    path = current_run().trace_path
    if not path.exists() or not rows:
        return None
    wanted = set(f.source_runs)
    took: dict[str, float] = {}
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if '"tool_call"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("artifact_id") in wanted and event.get("status") == "ok":
                took[event["artifact_id"]] = float(event.get("duration_s") or 0.0)
    return sum(took.values()) if took else None


def _rebuild_series(p: dict[str, Any]) -> pd.DataFrame:
    """Recompute the aggregated series of a time-series run (needed for re-detection)."""
    from jellybean.skills._shared.catalog import resolve_source

    source = resolve_source(p["input"])
    con = connect()
    try:
        t = quote_ident(p["time_column"])
        g = quote_ident(p["group_by"]) if p["group_by"] else "'__all__'"
        value = "count(*)" if p["measure"] == "count" else f"sum({quote_ident(p['measure'])})"
        series = con.execute(
            f"SELECT CAST({g} AS VARCHAR) AS \"group\", date_trunc('{p['granularity']}', {t}) AS ts, {value} AS value, "
            f"count(*) AS n "
            f"FROM {source.view} WHERE {t} IS NOT NULL GROUP BY 1, 2"
        ).df()
    finally:
        con.close()
    if p.get("model") == "grouped":
        series["value"] = series["value"].astype(float)
        return series  # active buckets only: the grouped model builds each group's grid
    step = pd.Timedelta(hours=1) if p["granularity"] == "hour" else pd.Timedelta(days=1)
    full = pd.date_range(series["ts"].min(), series["ts"].max(), freq=step)
    grid = pd.MultiIndex.from_product([series["group"].unique(), full], names=["group", "ts"])
    series = series.set_index(["group", "ts"]).reindex(grid, fill_value=0).reset_index()
    series["value"] = series["value"].astype(float)
    return series


# --------------------------------------------------------------------------- #
# ensemble_agreement                                                          #
# --------------------------------------------------------------------------- #
@tool
def ensemble_agreement(finding: str) -> ToolResult:
    """Agreement between the run's outlier detectors on the finding's members.

    For every outlier detection run of this run: share of the members it flags
    (vs its base rate), mean number of methods flagging a member, and the share
    of members flagged by at least two methods.

    Args:
        finding: Finding id.
    """
    try:
        f = _load(finding)
    except FindingNotFoundError as exc:
        return ToolResult.error(str(exc))
    if f.type == "temporal":
        result = {"applicable": False, "reason": "temporal findings are not scored by entity outlier methods"}
        _record_check(f, "ensemble", result)
        return ToolResult.ok({"finding": f.id, **result})
    members = _members(f)
    store = _store()
    runs = [r for r in store.list("table") if r.params.get("kind") == "outlier"]
    if not runs:
        return ToolResult.error("No outlier detection runs in this run.")
    per_method, votes = [], None
    for record in runs:
        table = pd.read_parquet(store.path_of(record.id)).set_index(record.params["entity"])
        flags = table["flagged"].astype(int)
        votes = flags if votes is None else votes.add(flags, fill_value=0)
        rate = float(flags[flags.index.isin(members)].mean()) if members else 0.0
        per_method.append(
            {
                "method": record.params["method"],
                "run": record.id,
                "member_flag_rate": round(rate, 4),
                "base_rate": round(float(flags.mean()), 4),
            }
        )
    member_votes = votes[votes.index.isin(members)]
    result = {
        "applicable": True,
        "methods": len(runs),
        "per_method": per_method,
        "mean_votes": round(float(member_votes.mean()), 3),
        "share_flagged_by_2_plus": round(float((member_votes >= 2).mean()), 4),
    }
    _record_check(f, "ensemble", result)
    return ToolResult.ok({"finding": f.id, **result})


# --------------------------------------------------------------------------- #
# artifact_check                                                              #
# --------------------------------------------------------------------------- #
def _primary_with_files() -> tuple[str, Any]:
    spec = read_spec()
    primary = spec.primary()
    if primary is None:
        raise SpecMissingError("No primary dataset in the spec.")
    info = discover_datasets(current_run().settings.data_root).get(primary.name)
    if info is None:
        raise SourceNotFoundError(f"Primary dataset '{primary.name}' not found.")
    return dataset_relation_with_files(info), info


def _null_shares(path: Any) -> dict[str, float]:
    """Share of nulls of every column of a Parquet file, from its metadata (no data is read)."""
    import pyarrow.parquet as pq

    meta = pq.read_metadata(path)
    if meta.num_rows == 0:
        return {}
    nulls: dict[str, int] = {}
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        for c in range(group.num_columns):
            column = group.column(c)
            stats = column.statistics
            if stats is None or not stats.has_null_count:
                return _null_shares_by_reading(path)
            nulls[column.path_in_schema] = nulls.get(column.path_in_schema, 0) + stats.null_count
    return {name: n / meta.num_rows for name, n in nulls.items()}


def _null_shares_by_reading(path: Any) -> dict[str, float]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return {name: table.column(name).null_count / table.num_rows for name in table.column_names}


def _association(name: str, finding_share: float, population_share: float, cfg: Any, **extra: Any) -> dict[str, Any]:
    lift = finding_share / population_share if population_share > 0 else (np.inf if finding_share > 0 else 1.0)
    suspicious = finding_share >= cfg.artifact_min_share and lift >= cfg.artifact_min_lift
    return {
        "source": name,
        "finding_share": round(float(finding_share), 4),
        "population_share": round(float(population_share), 4),
        "lift": round(float(min(lift, 1e6)), 2),
        "suspicious": bool(suspicious),
        **extra,
    }


# Facts about the primary dataset that every artifact check of a run needs (row count,
# null shares, rows per file, per-file medians): {(run folder, dataset): {fact: value}}.
_SOURCE_FACTS: dict[tuple[str, str], dict[Any, Any]] = {}
_SOURCE_FACTS_MAX = 8


def clear_source_facts() -> None:
    _SOURCE_FACTS.clear()


def _source_facts(dataset: str) -> dict[Any, Any]:
    key = (str(current_run().run_dir), dataset)
    if key not in _SOURCE_FACTS:
        if len(_SOURCE_FACTS) >= _SOURCE_FACTS_MAX:
            _SOURCE_FACTS.clear()
        _SOURCE_FACTS[key] = {}
    return _SOURCE_FACTS[key]


def _shifted_files(medians: list[tuple[Any, float | None]]) -> list[Any]:
    """Files whose median of a measure is at least 10x away from the typical file median."""
    values = [abs(v) for _, v in medians if v is not None]
    if len(values) < 3:
        return []
    typical = float(np.median(values)) or 0.0
    shifted = [
        fl for fl, v in medians if v is not None and typical and (abs(v) / typical >= 10 or 0 < abs(v) / typical <= 0.1)
    ]
    return sorted(shifted, key=str)  # a stable order: the first files are shown as examples


# Issue kinds that describe the observation itself (a day with more events), not
# how the data was produced: they cannot make a temporal finding an artefact.
_OBSERVATION_KINDS = {"volume_spike_day"}


def _unresolved_issues() -> list[dict[str, Any]]:
    """Quality issues of the run that the applied cleaning did not resolve."""
    spec = read_spec()
    store = _store()
    issues_id = spec.evidence_artifacts.get("quality_issues")
    if not issues_id:
        return []
    issues = store.load_json(issues_id)["issues"]
    resolved: set[str] = set()
    if spec.cleaning_plan.artifact_id and spec.cleaning_plan.status == "applied":
        plan = store.load_json(spec.cleaning_plan.artifact_id)
        for op in plan["ops"]:
            if op["id"] in spec.cleaning_plan.approved_ops:
                resolved |= set(op["issue_ids"])
    return [i for i in issues if i["id"] not in resolved]


@tool
def artifact_check(finding: str) -> ToolResult:
    """Test whether a finding is explained by how the data was produced or prepared, not by behaviour.

    Measures the association of the finding's rows/entities with: source files,
    files whose measures have a scale shift, timestamps outside their partition,
    null values, cleaning operations, features imputed after nulls, and
    unresolved quality issues on the same dates/files. Reports shares and lifts
    against the population; never file contents or rows.

    Args:
        finding: Finding id.
    """
    try:
        f = _load(finding)
        relation, info = _primary_with_files()
        spec = read_spec()
    except (FindingNotFoundError, SpecMissingError, SourceNotFoundError) as exc:
        return ToolResult.error(str(exc))
    cfg = _config().critic
    time_col = spec.time_column
    entity = spec.entity.column if spec.entity else None
    associations: list[dict[str, Any]] = []
    reasons: list[str] = []
    con = connect()
    try:
        con.execute(f"CREATE TEMP VIEW src AS SELECT * FROM {relation}")
        schema = dict(describe(con, "src"))
        if f.type == "temporal":
            windows = f.context.get("windows", [])
            group_by = f.context.get("group_by")
            conds = []
            for w in windows:
                cond = (
                    f"({quote_ident(time_col)} >= TIMESTAMP '{w['start']}' AND {quote_ident(time_col)} < "
                    f"TIMESTAMP '{w['end']}')"
                )
                if group_by:
                    cond += f" AND CAST({quote_ident(group_by)} AS VARCHAR) = '{str(w['group']).replace(chr(39), '')}'"
                conds.append(f"({cond})")
            con.execute(f"CREATE TEMP TABLE fin AS SELECT * FROM src WHERE {' OR '.join(conds) or 'false'}")
            measures = [f.context.get("measure")] if f.context.get("measure") not in (None, "count") else []
        else:
            members = pd.DataFrame({"m": sorted(_members(f), key=str)})
            con.register("members_df", members)
            con.execute(
                f"CREATE TEMP TABLE fin AS SELECT * FROM src WHERE CAST({quote_ident(entity)} AS VARCHAR) IN "
                "(SELECT CAST(m AS VARCHAR) FROM members_df)"
            )
            sources = {t["feature"] for t in f.top_features}
            dictionary = {}
            if spec.features.dictionary:
                dictionary = {d["name"]: d["source"] for d in _store().load_json(spec.features.dictionary)["features"]}
            measures = sorted(
                {dictionary.get(s) for s in sources}
                & {c for c, t in schema.items() if type_family(t) in ("integer", "float")}
            )
        # The finding's rows are materialised once (table ``fin``); what is measured on
        # the whole dataset does not depend on the finding and is kept for the next ones.
        facts = _source_facts(info.name)
        columns = [c for c in schema if c != FILE_COLUMN]
        null_exprs = ", ".join(f"avg(({quote_ident(c)} IS NULL)::DOUBLE)" for c in columns)
        fin_row = con.execute(f"SELECT count(*), {null_exprs} FROM fin").fetchall()[0]
        n_fin, fin_nulls = fin_row[0], dict(zip(columns, fin_row[1:], strict=True))
        if "rows" not in facts:
            pop_row = con.execute(f"SELECT count(*), {null_exprs} FROM src").fetchall()[0]
            facts["rows"], facts["nulls"] = pop_row[0], dict(zip(columns, pop_row[1:], strict=True))
        n_pop = facts["rows"]
        if n_fin == 0:
            return ToolResult.error("The finding matches no rows of the primary dataset.")

        # 1) Source file concentration (entity findings only: temporal windows are trivially in their day file).
        if f.type != "temporal":
            rows = con.execute(
                f"SELECT {FILE_COLUMN}, count(*) FILTER (WHERE true) FROM fin GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
            ).fetchall()
            if rows:
                file, n = rows[0]
                if "files" not in facts:
                    facts["files"] = dict(con.execute(f"SELECT {FILE_COLUMN}, count(*) FROM src GROUP BY 1").fetchall())
                pop = facts["files"].get(file, 0)
                associations.append(_association("single_source_file", n / n_fin, pop / n_pop, cfg, file=file))

        # 2) Files whose measure distribution is shifted (e.g. unit errors).
        exact = n_pop <= _config().profiling.exact_stats_max_rows
        missing = [m for m in measures if ("shift", m) not in facts]
        if missing:
            # DOUBLE: DECIMAL columns would come back as Decimal and not mix with floats.
            median = "median({q})::DOUBLE" if exact else "approx_quantile({q}, 0.5)::DOUBLE"
            exprs = ", ".join(median.format(q=quote_ident(m)) for m in missing)
            per_file = con.execute(f"SELECT {FILE_COLUMN}, {exprs} FROM src GROUP BY 1").fetchall()
            for pos, m in enumerate(missing, 1):
                facts["shift", m] = _shifted_files([(row[0], row[pos]) for row in per_file])
        for m in measures:
            shifted = facts["shift", m]
            if not shifted:
                continue
            q = quote_ident(m)
            placeholders = ", ".join("?" for _ in shifted)
            # Compare like with like. Temporal findings are rows in time: share of
            # their rows from shifted files vs the population's row share. Entity
            # findings: share of their measure mass from shifted files vs the
            # population's mass share (a dataset-wide shift affects everyone).
            shares = (
                f"SELECT avg(({FILE_COLUMN} IN ({placeholders}))::DOUBLE), "
                f"(sum(abs({q})) FILTER (WHERE {FILE_COLUMN} IN ({placeholders})) / nullif(sum(abs({q})), 0))::DOUBLE"
            )
            if ("shift_share", m) not in facts:
                facts["shift_share", m] = con.execute(f"{shares} FROM src", shifted + shifted).fetchall()[0]
            fin_shares = con.execute(f"{shares} FROM fin", shifted + shifted).fetchall()[0]
            basis, pos = ("rows", 0) if f.type == "temporal" else ("measure mass", 1)
            associations.append(
                _association(
                    "scale_shifted_file",
                    fin_shares[pos] or 0.0,
                    facts["shift_share", m][pos] or 0.0,
                    cfg,
                    column=m,
                    files=shifted[:3],
                    basis=basis,
                )
            )

        # 3) Timestamps outside their partition.
        part = next((c for c in info.partition_columns if type_family(schema.get(c, "")) == "temporal"), None)
        if part and time_col in schema:
            expr = f"avg((CAST({quote_ident(time_col)} AS DATE) <> CAST({quote_ident(part)} AS DATE))::DOUBLE)"
            fin_share = con.execute(f"SELECT {expr} FROM fin").fetchall()[0][0] or 0.0
            if "outside_partition" not in facts:
                facts["outside_partition"] = con.execute(f"SELECT {expr} FROM src").fetchall()[0][0] or 0.0
            associations.append(_association("timestamp_outside_partition", fin_share, facts["outside_partition"], cfg))

        # 4) Nulls concentrated in the finding.
        for col in columns:
            fin_null = fin_nulls[col] or 0.0
            if fin_null == 0:
                continue
            associations.append(_association("nulls", fin_null, facts["nulls"][col] or 0.0, cfg, column=col))
    finally:
        con.close()

    store = _store()
    if f.type != "temporal":
        members = _members(f)
        # 5) Cleaning operations (transform logs). Compared on ROWS: the share of the
        # finding's rows an operation touched vs the population's share. (Comparing
        # entities touched would be confounded by volume: an entity with many rows is
        # almost surely touched by a random deduplication.)
        logs = [r.id for r in store.list("table") if r.id.startswith("transform_log_")]
        if logs:
            # Counted in DuckDB: a log has one row per affected source row and is never loaded whole.
            con = connect()
            try:
                con.register("members_df", pd.DataFrame({"m": sorted(members, key=str)}))
                for log_id in logs:
                    if entity not in dict(describe(con, quote_ident(log_id))):
                        continue
                    per_op = con.execute(
                        f"SELECT op_id, count(*), count(*) FILTER (WHERE CAST({quote_ident(entity)} AS VARCHAR) IN "
                        f"(SELECT CAST(m AS VARCHAR) FROM members_df)) FROM {quote_ident(log_id)} "
                        "GROUP BY op_id ORDER BY op_id"
                    ).fetchall()
                    for op_id, touched, in_finding in per_op:
                        associations.append(
                            _association(
                                "cleaning_operation",
                                in_finding / n_fin,
                                touched / n_pop,
                                cfg,
                                operation=op_id,
                                log=log_id,
                                basis="rows",
                            )
                        )
            finally:
                con.close()
        # 6) Features that were null (imputed when scaling).
        features = feature_source()
        entity_col, raw_columns = features.entity, features.columns("raw")
        # Low support: entities with very few events have unreliable features. This
        # does not make a finding an artefact, but it prevents "confirmed".
        if "n_events" in raw_columns:
            support = read_features(features.raw_path, entity_col, columns=[entity_col, "n_events"])
            in_group = support[entity_col].isin(members)
            low_support_share = float(low_support(support["n_events"])[in_group].mean())
        # Only features of the finding's space matter: a group found on one family of
        # features is not explained by nulls in other features.
        families = f.context.get("families")
        space = set(family_columns(raw_columns, families)) if families else set(raw_columns)
        # Null counts come from the file's statistics; only the members' rows are read.
        null_share = _null_shares(features.raw_path)
        with_nulls = [c for c in raw_columns if c != entity_col and c in space and null_share.get(c, 0.0) > 0]
        if with_nulls:
            group = read_features(features.raw_path, entity_col, columns=[entity_col, *with_nulls], members=members)
            for col in with_nulls:
                associations.append(
                    _association(
                        "imputed_feature",
                        float(group[col].isna().mean()) if len(group) else float("nan"),
                        null_share[col],
                        cfg,
                        feature=col,
                    )
                )

    # 7) Unresolved quality issues on the same dates/files (temporal) or columns.
    unresolved = _unresolved_issues()
    if f.type == "temporal":
        dates = {str(w["start"])[:10] for w in f.context.get("windows", [])} | {
            str(pd.Timestamp(w["end"]) - pd.Timedelta(seconds=1))[:10] for w in f.context.get("windows", [])
        }
        for issue in unresolved:
            if issue["kind"] in _OBSERVATION_KINDS:
                continue  # the issue describes the same spike: it cannot explain it
            d = issue.get("detail", {})
            issue_dates = {str(d.get("date", ""))[:10]} | {str(d.get("partition", "")).split("=")[-1][:10]}
            issue_file = str(d.get("file", ""))
            if dates & issue_dates or any(day in issue_file for day in dates):
                reasons.append(
                    f"window overlaps unresolved quality issue {issue['id']} ({issue['kind']} on {issue.get('column')})"
                )

    for a in associations:
        if a["suspicious"]:
            detail = {k: v for k, v in a.items() if k in ("column", "file", "operation", "feature")}
            reasons.append(
                f"{a['source']} {detail}: {a['finding_share']:.0%} of the finding vs "
                f"{a['population_share']:.0%} of the population (lift {a['lift']})"
            )
    suspected = bool(reasons)
    top = sorted(associations, key=lambda a: (-a["suspicious"], -a["lift"] * a["finding_share"]))[:8]
    result = {"suspected": suspected, "reasons": reasons[:8], "associations": top}
    if f.type != "temporal" and "n_events" in raw_columns:
        result["low_support_share"] = round(low_support_share, 4)
    _record_check(f, "artifact", result)
    return ToolResult.ok({"finding": f.id, **result})


# --------------------------------------------------------------------------- #
# explain_by_reference                                                        #
# --------------------------------------------------------------------------- #
_VALIDITY_START = {"start", "from", "begin", "valid", "effective", "distribution", "activation"}
_VALIDITY_END = {"end", "to", "until", "expiry", "expires", "expiration"}


def _column_dataset(spec: Any, column: str) -> str | None:
    """Dataset a column of the enriched table comes from (enrichment columns may carry a suffix)."""
    names = {d.name for d in spec.datasets}
    for key, info in spec.columns.items():
        if key == column or key.endswith(f".{column}"):
            return info.dataset
    for name in names:
        if column.endswith(f"_{name}"):
            return name
    return None


def _validity_columns(spec: Any, dataset: str) -> tuple[str, str] | None:
    from jellybean.skills._shared.column_stats import name_tokens

    stamps = [
        k.split(".")[-1] for k, v in spec.columns.items() if v.dataset == dataset and v.semantic_type == "timestamp"
    ]
    start = next((c for c in stamps if name_tokens(c) & _VALIDITY_START), None)
    end = next((c for c in stamps if name_tokens(c) & _VALIDITY_END and c != start), None)
    return (start, end) if start and end else None


def _rule_on_windows(spec: Any, entity: str, rule: dict[str, Any], windows: list[dict[str, Any]]) -> tuple[int, int]:
    """(rows checked, rows off) of a conformity rule on the rows of temporal windows."""
    t = quote_ident(spec.time_column)
    conds = [
        f"(CAST(ev.{quote_ident(entity)} AS VARCHAR) = '{str(w['group']).replace(chr(39), '')}' "
        f"AND ev.{t} >= TIMESTAMP '{w['start']}' AND ev.{t} < TIMESTAMP '{w['end']}')"
        for w in windows
    ]
    con = connect()
    try:
        relation = _rule_relation(con, spec, entity, rule)
        if relation is None:
            return 0, 0
        rows, off = con.execute(
            f"SELECT count(*), count(*) FILTER (WHERE abs(CAST(({rule['observed']}) AS DOUBLE) - "
            f"CAST(({rule['expected']}) AS DOUBLE)) > {float(rule['tolerance'])}) FROM {relation} "
            f"WHERE ({' OR '.join(conds) or 'false'}) AND coalesce(({rule['where']}), false)"
        ).fetchall()[0]
    finally:
        con.close()
    return int(rows), int(off)


@tool
def explain_by_reference(finding: str) -> ToolResult:
    """Test whether a finding is a real pattern with a legitimate cause visible in the reference tables.

    Two explanations are tested:

    * **reference category**: the members are (most of) one category of an
      attribute from a reference table, e.g. every large retail chain. The
      pattern is how that category works, not an anomaly of its members
      (at least 80% of the members share the value and they are at least half of
      that category, a minority of the population).
    * **reference rules**: the entity is the key of a reference table (e.g. a
      promotion) and its members follow every conformity rule built from that
      table (at most 1% of rows off; for temporal findings, on the rows of the
      window), and, for temporal findings, the window lies inside
      the validity period of the reference row (e.g. a campaign day within the
      campaign's dates).

    The result is recorded in the finding; when it is explained, set_verdict
    requires the verdict "explained" (unless it is an artefact).

    Args:
        finding: Finding id.
    """
    try:
        f = _load(finding)
        spec = read_spec()
        features = feature_source()
    except (FindingNotFoundError, SpecMissingError) as exc:
        return ToolResult.error(str(exc))
    store = _store()
    entity = features.entity
    dictionary = store.load_json(spec.features.dictionary)["features"] if spec.features.dictionary else []
    # Only the columns the tests below read: attributes and the counts of the conformity rules.
    available = set(features.columns("raw"))
    needed = [entity]
    for d in dictionary:
        if d["template"] == "attribute":
            needed.append(d["name"])
        elif d["template"] == "conformity" and d["name"].endswith("_violations"):
            needed += [d["name"], d["name"][: -len("_violations")] + "_rows"]
    raw = read_features(features.raw_path, entity, columns=[c for c in needed if c in available])
    primary = spec.primary().name if spec.primary() else None
    references = {j.right for j in spec.joins if j.status == "confirmed"}
    own = next((j.right for j in spec.joins if j.status == "confirmed" and j.keys[0][0] == entity), None)
    if f.type == "temporal":
        group_by = f.context.get("group_by")
        members = {str(w["group"]) for w in f.context.get("windows", [])} if group_by == entity else set()
    else:
        members = {str(m) for m in _members(f)}
    ids = raw[entity].astype(str)
    inside = ids.isin(members)
    result: dict[str, Any] = {"explained": False, "by": None, "members_in_features": int(inside.sum())}
    cfg = _config().detection

    # 1) A whole reference category.
    groups = []
    for d in dictionary:
        name = d["name"]
        if d["template"] != "attribute" or name not in raw.columns or pd.api.types.is_numeric_dtype(raw[name]):
            continue
        dataset = _column_dataset(spec, d["source"])
        if dataset not in references or dataset == primary:
            continue
        values = raw.loc[inside, name].astype("string")
        if values.empty or values.isna().all():
            continue
        top = values.value_counts()
        value, count = top.index[0], int(top.iloc[0])
        category = raw[name].astype("string") == value
        size = int(category.sum())
        groups.append(
            {
                "attribute": d["source"],
                "dataset": dataset,
                "value": str(value),
                "member_share": round(count / max(int(inside.sum()), 1), 4),
                "category_size": size,
                "category_coverage": round(count / max(size, 1), 4),
                "population_share": round(size / len(raw), 4),
            }
        )
    groups.sort(key=lambda g: -g["member_share"] * g["category_coverage"])
    result["reference_categories"] = groups[:4]
    best = next(
        (
            g
            for g in groups
            if g["member_share"] >= 0.8
            and g["category_coverage"] >= 0.5
            and g["category_size"] >= cfg.min_group_size
            and g["population_share"] <= 0.5
        ),
        None,
    )
    if best:
        result |= {
            "explained": True,
            "by": "reference_category",
            "summary": f"{best['member_share']:.0%} of the members are {best['attribute']} = {best['value']} "
            f"({best['dataset']}), and they are {best['category_coverage']:.0%} of that category: the pattern is how "
            "this category works.",
        }

    # 2) Rules of the entity's own reference table.
    if own and not result["explained"]:
        rules: dict[str, dict[str, Any]] = {}
        for d in dictionary:
            if d["template"] != "conformity" or not d["name"].endswith("_violations"):
                continue
            if not any(_column_dataset(spec, c) == own for c in d.get("columns", [])):
                continue
            prefix = d["name"][: -len("_violations")]
            if f.type == "temporal" and d.get("rule") and spec.enriched_artifact:
                # A temporal finding is explained by the activity of its window: check the rule there.
                rows, off = _rule_on_windows(spec, entity, d["rule"], f.context.get("windows", []))
            elif f"{prefix}_rows" in raw.columns:
                rows = int(raw.loc[inside, f"{prefix}_rows"].fillna(0).sum())
                off = int(raw.loc[inside, d["name"]].fillna(0).sum())
            else:
                continue
            rules[d["source"]] = {"rows": rows, "violations": off}
        applicable = {k: v for k, v in rules.items() if v["rows"] > 0}
        result["reference_rules"] = {"table": own, "rules": rules}
        validity = None
        if f.type == "temporal" and spec.enriched_artifact and spec.time_column:
            columns = _validity_columns(spec, own)
            if columns:
                start, end = (quote_ident(c) for c in columns)
                t = quote_ident(spec.time_column)
                conds = [
                    f"(CAST({quote_ident(entity)} AS VARCHAR) = '{str(w['group']).replace(chr(39), '')}' "
                    f"AND {t} >= TIMESTAMP '{w['start']}' AND {t} < TIMESTAMP '{w['end']}')"
                    for w in f.context.get("windows", [])
                ]
                con = connect()
                try:
                    share = con.execute(
                        f"SELECT avg((CAST({t} AS DATE) BETWEEN CAST({start} AS DATE) "
                        f"AND CAST({end} AS DATE))::DOUBLE) "
                        f'FROM "{spec.enriched_artifact}" WHERE {" OR ".join(conds) or "false"}'
                    ).fetchall()[0][0]
                finally:
                    con.close()
                validity = {"columns": list(columns), "share_inside": None if share is None else round(float(share), 4)}
                result["reference_rules"]["validity"] = validity
        valid = validity is None or (validity["share_inside"] or 0) >= 0.99
        # A few stray rows (<= 1%) off a rule do not make the activity irregular.
        if applicable and valid and all(v["violations"] <= 0.01 * v["rows"] for v in applicable.values()):
            checked = sum(v["rows"] for v in applicable.values())
            off = sum(v["violations"] for v in applicable.values())
            where = (
                f" and the window is inside its validity period ({' / '.join(validity['columns'])})" if validity else ""
            )
            result |= {
                "explained": True,
                "by": "reference_rules",
                "summary": f"the members follow every rule of '{own}' ({len(applicable)} rule(s), {off} rows off "
                f"of {checked}){where}: the activity is what the reference table allows.",
            }
    # 3) The pattern belongs to another entity: the members' rule violations come from
    # one reference object that breaks the rule as a whole (e.g. a misconfigured promotion).
    if f.type != "temporal" and members and spec.enriched_artifact:
        result["attributed_to"] = _attribution(spec, entity, members, dictionary)
    _record_check(f, "explanation", result)
    return ToolResult.ok({"finding": f.id, **result})


def _attribution(spec: Any, entity: str, members: set[str], dictionary: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A reference object that breaks a rule as a whole, and causes most of the members' violations.

    For each conformity rule that at least half of the members break: their rows
    off the rule, by value of each reference key (e.g. promo_id). When at least
    80% of those rows have one value and that value also holds at least half of
    all the rows off the rule in the population, the rule is broken by that
    reference object (e.g. a misconfigured promotion), and the members are only
    its consequence.
    """
    keys = sorted({j.keys[0][0] for j in spec.joins if j.status == "confirmed" and j.keys[0][0] != entity})
    rules = {d["source"]: d["rule"] for d in dictionary if d.get("template") == "conformity" and d.get("rule")}
    if not keys or not rules:
        return None
    con = connect()
    best = None
    try:
        con.register("members_df", pd.DataFrame({"m": sorted(members)}))
        key_columns = ", ".join(f"CAST(ev.{quote_ident(k)} AS VARCHAR) AS k{i}" for i, k in enumerate(keys))
        mine_only = "ent IN (SELECT m FROM members_df)"
        for name, rule in rules.items():
            relation = _rule_relation(con, spec, entity, rule)
            if relation is None:
                continue
            # The rows off the rule are found once; every count below reads this small table.
            con.execute(
                f"CREATE OR REPLACE TEMP TABLE off_rows AS SELECT CAST(ev.{quote_ident(entity)} AS VARCHAR) AS ent, "
                f"{key_columns} FROM {relation} WHERE coalesce(({rule['where']}), false) AND "
                f"abs(CAST(({rule['observed']}) AS DOUBLE) - CAST(({rule['expected']}) AS DOUBLE)) > "
                f"{float(rule['tolerance'])}"
            )
            breakers = con.execute(f"SELECT count(DISTINCT ent) FROM off_rows WHERE {mine_only}").fetchall()[0][0]
            if breakers < 0.5 * len(members):
                continue  # the rule does not describe this group
            for i, key in enumerate(keys):
                mine = con.execute(
                    f"SELECT k{i}, count(*) FROM off_rows WHERE {mine_only} GROUP BY 1 ORDER BY 2 DESC"
                ).fetchall()
                total_mine = sum(n for _, n in mine)
                if not total_mine or mine[0][0] is None:
                    continue
                value, n = mine[0]
                everyone = con.execute(
                    f"SELECT count(*), count(*) FILTER (WHERE k{i} = ?) FROM off_rows", [value]
                ).fetchall()[0]
                share, of_all = n / total_mine, everyone[1] / max(everyone[0], 1)
                if share >= 0.8 and of_all >= 0.5 and (best is None or share > best["share_of_member_violations"]):
                    best = {
                        "column": key,
                        "value": value,
                        "rule": name,
                        "share_of_member_violations": round(share, 4),
                        "share_of_all_violations": round(of_all, 4),
                    }
    finally:
        con.close()
    return best


def _rule_relation(con: Any, spec: Any, entity: str, rule: dict[str, Any]) -> str | None:
    """The enriched table aliased ``ev`` for one conformity rule.

    A rule that reads ``ref_<column>`` gets those columns of the referred row
    (the dataset's self-reference); None when the spec has no self-reference.
    As in build_conformity_features, the referred rows are reduced to the
    columns the rule reads before they are deduplicated and joined.
    """
    table = f'"{spec.enriched_artifact}"'
    text = f"{rule['observed']} {rule['expected']} {rule['where']}"
    uses_ref = rule.get("uses_ref")
    if uses_ref is None:  # rules recorded before the flag existed
        uses_ref = "ref_" in text
    if not uses_ref:
        return f"(SELECT * FROM {table}) ev"
    primary = spec.primary().name if spec.primary() else None
    ref = next((r for r in spec.references if r.kind == "reference" and r.dataset == r.ref_dataset == primary), None)
    if ref is None:
        return None
    columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()]
    keep = [c for c in columns if c in (entity, ref.column, ref.ref_column) or c in text]
    ref_cols = ", ".join(f"p.{quote_ident(c)} AS {quote_ident('ref_' + c)}" for c in keep if "ref_" + c not in columns)
    referred = f"(SELECT DISTINCT {', '.join(quote_ident(c) for c in keep)} FROM {table})"
    return (
        f"(SELECT ev.*, {ref_cols} FROM {table} ev LEFT JOIN {referred} p "
        f"ON ev.{quote_ident(ref.column)} = p.{quote_ident(ref.ref_column)}) ev"
    )


@tool
def rerun_on_version(finding: str, artifact_version: str, seed: int | None = None) -> ToolResult:
    """Repeat the finding's outlier detection on another version of the scaled feature table.

    Args:
        finding: Finding id.
        artifact_version: Another scaled feature table (e.g. built from a differently cleaned dataset).
        seed: Random seed.
    """
    try:
        f = _load(finding)
        matrix, _ = _matrix(artifact_version)
    except (FindingNotFoundError, ArtifactNotFoundError, KeyError) as exc:
        return ToolResult.error(f"Cannot rerun: {exc}")
    cfg = _config().detection
    outlier_runs = [p for _, p in _runs(f) if p["kind"] == "outlier"]
    if not outlier_runs:
        return ToolResult.error("This finding was not produced by outlier methods; use stability_check instead.")
    votes = np.zeros(len(matrix))
    for p in outlier_runs:
        scores = outlier_scores(
            matrix.to_numpy(),
            p["method"],
            p.get("method_params", {}),
            int(seed),
            cfg.lof_max_rows,
            fit_max_rows=cfg.fit_max_rows,
        )
        votes += flag_top(scores, p.get("contamination", cfg.contamination))
    found = set(matrix.index[votes >= min(cfg.min_votes, len(outlier_runs))])
    members = _members(f) & set(matrix.index)
    share = len(members & found) / len(members) if members else 0.0
    result = {"version": artifact_version, "redetected_share": round(share, 4), "members_in_version": len(members)}
    _record_check(f, "rerun_on_version", result)
    return ToolResult.ok({"finding": f.id, **result})


@tool
def inject_synthetic_anomalies(
    features: str,
    n: Annotated[int, Field(ge=5, le=500)] = 20,
    kind: Literal["extreme", "group"] = "extreme",
    seed: int | None = None,
) -> ToolResult:
    """Plant known anomalies in the scaled features and measure how often the run's outlier methods find them.

    "extreme": random entities with 3 features shifted by +8 robust units.
    "group": copies of one random profile shifted together (a coordinated group).

    Args:
        features: Scaled feature table.
        n: Number of synthetic anomalies.
        kind: "extreme" or "group".
        seed: Random seed.
    """
    try:
        matrix, _ = _matrix(features)
    except (ArtifactNotFoundError, KeyError) as exc:
        return ToolResult.error(f"Unknown feature table: {exc}")
    store = _store()
    runs = [r.params for r in store.list("table") if r.params.get("kind") == "outlier"]
    if not runs:
        return ToolResult.error("No outlier detection runs in this run.")
    rng = np.random.default_rng(seed)
    base = matrix.to_numpy()
    cfg = _config().detection
    sampled = len(base) > cfg.fit_max_rows
    if sampled:
        # The planted rows are judged against a seeded sample of the population: refitting
        # every method on millions of rows would cost as much as the detection itself.
        base = base[np.sort(rng.choice(len(base), cfg.fit_max_rows, replace=False))]
    del matrix
    rows = base[rng.choice(len(base), n, replace=True)].copy()
    cols = rng.choice(base.shape[1], 3, replace=False)
    if kind == "extreme":
        for row in rows:
            row[rng.choice(base.shape[1], 3, replace=False)] += 8.0
    else:
        rows[:] = rows[0]
        rows[:, cols] += 6.0
        rows += rng.normal(0, 0.1, rows.shape)
    X = np.vstack([base, rows])
    del base
    per_method, votes = [], np.zeros(n)
    for p in runs:
        scores = outlier_scores(
            X, p["method"], p.get("method_params", {}), int(seed), cfg.lof_max_rows, fit_max_rows=cfg.fit_max_rows
        )
        flags = flag_top(scores, p.get("contamination", cfg.contamination))[-n:]
        votes += flags
        per_method.append({"method": p["method"], "detection_rate": round(float(flags.mean()), 4)})
    return ToolResult.ok(
        {
            "kind": kind,
            "planted": n,
            "per_method": per_method,
            "detected_by_2_plus": round(float((votes >= 2).mean()), 4),
        },
        warnings=[f"Measured against a sample of {cfg.fit_max_rows} entities."] if sampled else [],
    )


@tool
def compare_to_known_labels(finding: str, labels_artifact: str | None = None) -> ToolResult:
    """Compare the finding's members with user-provided confirmed cases (`--labels`).

    Args:
        finding: Finding id.
        labels_artifact: Dataset or table artifact with confirmed entities (defaults to spec.labels_dataset).
    """
    from jellybean.skills._shared.catalog import resolve_source

    try:
        f = _load(finding)
        spec = read_spec()
    except (FindingNotFoundError, SpecMissingError) as exc:
        return ToolResult.error(str(exc))
    name = labels_artifact or spec.labels_dataset
    if not name:
        return ToolResult.error("No labels: the run was started without --labels.")
    try:
        source = resolve_source(name)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        cols = [c for c, _ in describe(con, source.view)]
        if f.entity_column not in cols:
            return ToolResult.error(f"Labels have no column '{f.entity_column}'.")
        labels = {
            r[0] for r in con.execute(f"SELECT DISTINCT {quote_ident(f.entity_column)} FROM {source.view}").fetchall()
        }
    finally:
        con.close()
    members = _members(f)
    hits = len(members & labels)
    result = {
        "labels": len(labels),
        "overlap": hits,
        "precision": round(hits / len(members), 4) if members else 0.0,
        "recall": round(hits / len(labels), 4) if labels else 0.0,
    }
    _record_check(f, "labels", result)
    return ToolResult.ok({"finding": f.id, **result})


@tool
def match_pattern_registry(finding: str) -> ToolResult:
    """Compare the finding's signature with the persistent pattern registry: known, variant or new.

    Args:
        finding: Finding id.
    """
    try:
        f = _load(finding)
        spec = read_spec()
    except (FindingNotFoundError, SpecMissingError) as exc:
        return ToolResult.error(str(exc))
    ctx = current_run()
    thresholds = ctx.settings.config.pattern_registry
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
    pattern, score = best_match(ctx, registry_key(spec), sig, f.type)
    novelty = classify_novelty(score, thresholds.known_threshold, thresholds.variant_threshold)
    f.novelty = novelty
    f.pattern_match = {"pattern": pattern["id"] if pattern else None, "similarity": score}
    _record_check(f, "registry", {"novelty": novelty, "best_pattern": f.pattern_match["pattern"], "similarity": score})
    return ToolResult.ok(
        {
            "finding": f.id,
            "novelty": novelty,
            "best_pattern": f.pattern_match["pattern"],
            "similarity": score,
            "best_pattern_description": (pattern or {}).get("description", "")[:300],
        }
    )


@tool
def set_verdict(finding: str, verdict: Verdict, evidence: dict[str, Any]) -> ToolResult:
    """Record the verdict on a finding: confirmed, explained, artifact, unstable or weak.

    Requires stability_check and artifact_check first. If artifact_check
    suspected an artefact, the verdict must be "artifact". "confirmed" requires
    stability at or above the configured minimum and fewer than half of the
    members with low activity (low_support_share < 0.5). "explained" requires
    explain_by_reference to have found a legitimate cause, and stability at or
    above the minimum; a finding it explains cannot be "confirmed".

    Args:
        finding: Finding id.
        verdict: confirmed | explained | artifact | unstable | weak.
        evidence: Short dict justifying the verdict with numbers from the checks, e.g. {"stability": 0.95, ...}.
    """
    try:
        f = _load(finding)
    except FindingNotFoundError as exc:
        return ToolResult.error(str(exc))
    missing = [c for c in ("stability", "artifact") if c not in f.checks]
    if missing:
        return ToolResult.error(
            f"Run {missing} before setting a verdict (call each once; a check that cannot measure records why "
            "and still allows a verdict)."
        )
    stability = f.checks["stability"].get("stability")
    if stability is None and not f.checks["artifact"]["suspected"]:
        # Attempted but not measured (timeout, no members): the finding still gets a verdict,
        # but not one that claims stability.
        why = f.checks["stability"].get("reason", "unknown")
        if verdict not in ("weak", "unstable"):
            return ToolResult.error(
                f"Stability was not measured ({why}): the verdict cannot be '{verdict}'. Use 'weak' (the other "
                "checks pass) or 'unstable'. Do not run stability_check again."
            )
        evidence = {**evidence, "stability": f"not measured ({why})"}
    if f.checks["artifact"]["suspected"] and verdict != "artifact":
        return ToolResult.error(
            "artifact_check suspected a data artefact: the verdict must be 'artifact'.",
            reasons=f.checks["artifact"]["reasons"][:3],
        )
    minimum = _config().critic.min_stability
    explanation = f.checks.get("explanation") or {}
    if verdict == "explained" and not explanation.get("explained"):
        return ToolResult.error(
            "'explained' needs a legitimate cause found by explain_by_reference "
            f"({'not run yet' if not explanation else 'it found none'})."
        )
    if verdict == "explained" and stability is not None and stability < minimum:
        return ToolResult.error(
            f"Stability {f.checks['stability']['stability']} is below {minimum}: an explained pattern must be "
            "stable (use 'unstable')."
        )
    attributed = explanation.get("attributed_to")
    if verdict == "confirmed" and attributed:
        return ToolResult.error(
            f"{attributed['share_of_member_violations']:.0%} of the members' rows off rule '{attributed['rule']}' "
            f"have {attributed['column']} = {attributed['value']}, which holds "
            f"{attributed['share_of_all_violations']:.0%} of all its violations: the pattern belongs to that "
            f"{attributed['column']}, not to these entities. Use 'weak' and say which entity to analyse.",
        )
    if verdict == "confirmed" and explanation.get("explained"):
        return ToolResult.error(
            "explain_by_reference found a legitimate cause: the verdict must be 'explained'.",
            explanation=explanation.get("summary"),
        )
    low_support = f.checks["artifact"].get("low_support_share", 0.0)
    if verdict == "confirmed" and low_support >= 0.5:
        return ToolResult.error(
            f"{low_support:.0%} of the members have very few events (bottom 10% of activity): "
            "their features are unreliable, the verdict cannot be 'confirmed' (use 'weak')."
        )
    if verdict == "confirmed" and stability is not None and stability < minimum:
        return ToolResult.error(
            f"Stability {f.checks['stability']['stability']} is below {minimum}: "
            "the verdict cannot be 'confirmed' (use 'unstable' or 'weak')."
        )
    f.critic_verdict = verdict
    f.critic_evidence = {
        "evidence": evidence,
        "checks": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("associations", "per_method")} for k, v in f.checks.items()
        },
    }
    save_finding(f)
    return ToolResult.ok({"finding": f.id, "verdict": verdict, "novelty": f.novelty})
