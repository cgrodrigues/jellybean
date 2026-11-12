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

from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from jellybean.runtime.context import current_run
from jellybean.runtime.tool_registry import ToolResult, tool
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
    feature_tables,
    finding_entities,
    load_finding,
    numeric_matrix,
    save_finding,
    signature,
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


def _matrix(features: str) -> tuple[pd.DataFrame, str]:
    store = _store()
    entity = store.get(features).params["entity"]
    return numeric_matrix(pd.read_parquet(store.path_of(features)), entity), entity


# --------------------------------------------------------------------------- #
# stability_check                                                             #
# --------------------------------------------------------------------------- #
def _redetect(finding: Finding, sample: set[Any], seed: int) -> set[Any]:
    """Entities re-detected by the finding's source methods on a subsample."""
    cfg = _config().detection
    runs = _runs(finding)
    outlier_runs = [p for _, p in runs if p["kind"] == "outlier"]
    found: set[Any] = set()
    # Outlier runs vote within their own feature space (all features, or one family).
    spaces: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for p in outlier_runs:
        spaces.setdefault(tuple(p.get("families") or ()), []).append(p)
    for family, space_runs in spaces.items():
        matrix, _ = _matrix(space_runs[0]["features"])
        matrix = matrix[family_columns(list(matrix.columns), list(family) or None)]
        sub = matrix[matrix.index.isin(sample)]
        votes = np.zeros(len(sub))
        for p in space_runs:
            scores = outlier_scores(sub.to_numpy(), p["method"], p.get("method_params", {}), seed, cfg.lof_max_rows)
            votes += flag_top(scores, p.get("contamination", cfg.contamination))
        found |= set(sub.index[votes >= min(cfg.min_votes, len(space_runs))])
    for _, p in runs:
        if p["kind"] == "contextual":
            raw = pd.read_parquet(_store().path_of(p["features"]))
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
            found |= set(raw.loc[flags.to_numpy(), p["entity"]])
        elif p["kind"] == "cluster":
            matrix, _ = _matrix(p["features"])
            sub = matrix[matrix.index.isin(sample)]
            labels = pd.Series(
                cluster_labels(sub.to_numpy(), p["method"], p.get("method_params", {}), seed), index=sub.index
            )
            # The cluster that best matches the finding's members.
            inside = labels[labels.index.isin(_members(finding))]
            inside = inside[inside != -1]
            if not inside.empty:
                found |= set(labels.index[labels == inside.mode().iloc[0]])
        elif p["kind"] == "graph" and p.get("method") in ("components", "communities"):
            import networkx as nx

            edges = pd.read_parquet(_store().path_of(p["graph"]))
            edges = edges.sample(frac=0.8, random_state=seed)
            edges = edges[edges["src"].isin(sample) & edges["dst"].isin(sample)]
            g = nx.Graph()
            g.add_edges_from(edges[["src", "dst"]].itertuples(index=False, name=None))
            for component in nx.connected_components(g):
                if len(component) >= cfg.min_group_size:
                    found |= component
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
    n_runs = n_runs or _config().critic.stability_runs
    rng = np.random.default_rng(seed)
    recalls, jaccards = [], []
    if f.type == "temporal":
        rid, p = _runs(f)[0]
        points = pd.read_parquet(_store().path_of(rid))
        if not len(points):
            return ToolResult.error("No anomalous points stored for this temporal finding.")
        windows = f.context.get("windows", [])
        group = {str(w["group"]) for w in windows}
        series = _rebuild_series(p)
        step = pd.Timedelta(hours=1) if p["granularity"] == "hour" else pd.Timedelta(days=1)
        days = series["ts"].dt.normalize().unique()
        groups = series["group"].unique()
        mine = points[points["group"].astype(str).isin(group)]
        for _ in range(n_runs):
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
            recalls.append(len(hits) / len(mine))
    else:
        members = _members(f)
        _, scaled, entity = feature_tables()
        population = scaled[entity].tolist()
        for i in range(n_runs):
            sample = set(rng.choice(population, int(len(population) * subsample), replace=False))
            sampled_members = members & sample
            if not sampled_members:
                continue
            found = _redetect(f, sample, int(seed) + i + 1)
            recalls.append(len(sampled_members & found) / len(sampled_members))
            union = len(found | sampled_members)
            jaccards.append(len(found & sampled_members) / union if union else 0.0)
    if not recalls:
        return ToolResult.error("Stability could not be measured (no members in the subsamples).")
    result = {
        "stability": round(float(np.mean(recalls)), 4),
        "stability_std": round(float(np.std(recalls)), 4),
        "runs": len(recalls),
        "subsample": subsample,
        "mean_jaccard": round(float(np.mean(jaccards)), 4) if jaccards else None,
    }
    _record_check(f, "stability", result)
    return ToolResult.ok({"finding": f.id, **result, "min_stability_for_confirmed": _config().critic.min_stability})


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
            con.execute(f"CREATE TEMP VIEW fin AS SELECT * FROM src WHERE {' OR '.join(conds) or 'false'}")
            measures = [f.context.get("measure")] if f.context.get("measure") not in (None, "count") else []
        else:
            members = pd.DataFrame({"m": sorted(_members(f), key=str)})
            con.register("members_df", members)
            con.execute(
                f"CREATE TEMP VIEW fin AS SELECT * FROM src WHERE CAST({quote_ident(entity)} AS VARCHAR) IN "
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
        n_fin = con.execute("SELECT count(*) FROM fin").fetchall()[0][0]
        n_pop = con.execute("SELECT count(*) FROM src").fetchall()[0][0]
        if n_fin == 0:
            return ToolResult.error("The finding matches no rows of the primary dataset.")

        # 1) Source file concentration (entity findings only: temporal windows are trivially in their day file).
        if f.type != "temporal":
            rows = con.execute(
                f"SELECT {FILE_COLUMN}, count(*) FILTER (WHERE true) FROM fin GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
            ).fetchall()
            if rows:
                file, n = rows[0]
                pop = con.execute(f"SELECT count(*) FROM src WHERE {FILE_COLUMN} = ?", [file]).fetchall()[0][0]
                associations.append(_association("single_source_file", n / n_fin, pop / n_pop, cfg, file=file))

        # 2) Files whose measure distribution is shifted (e.g. unit errors).
        numeric = [c for c, t in schema.items() if type_family(t) in ("integer", "float") and c != FILE_COLUMN]
        check_measures = measures or numeric
        for m in check_measures:
            q = quote_ident(m)
            # DOUBLE: DECIMAL columns would come back as Decimal and not mix with floats.
            med = con.execute(f"SELECT {FILE_COLUMN}, median({q})::DOUBLE FROM src GROUP BY 1").fetchall()
            values = [abs(v) for _, v in med if v is not None]
            if len(values) < 3:
                continue
            typical = float(np.median(values)) or 0.0
            shifted = [
                fl
                for fl, v in med
                if v is not None and typical and (abs(v) / typical >= 10 or 0 < abs(v) / typical <= 0.1)
            ]
            if not shifted:
                continue
            placeholders = ", ".join("?" for _ in shifted)
            if m in measures:
                # Compare like with like. Temporal findings are rows in time: share of
                # their rows from shifted files vs the population's row share. Entity
                # findings: share of their measure mass from shifted files vs the
                # population's mass share (a dataset-wide shift affects everyone).
                if f.type == "temporal":
                    basis = "rows"
                    fin_share = (
                        con.execute(
                            f"SELECT avg(({FILE_COLUMN} IN ({placeholders}))::DOUBLE) FROM fin", shifted
                        ).fetchall()[0][0]
                        or 0.0
                    )
                    pop_share = (
                        con.execute(
                            f"SELECT avg(({FILE_COLUMN} IN ({placeholders}))::DOUBLE) FROM src", shifted
                        ).fetchall()[0][0]
                        or 0.0
                    )
                else:
                    basis = "measure mass"
                    mass = (
                        f"(sum(abs({q})) FILTER (WHERE {FILE_COLUMN} IN ({placeholders})) / nullif(sum(abs({q})), 0))"
                        "::DOUBLE"
                    )
                    fin_share = con.execute(f"SELECT {mass} FROM fin", shifted).fetchall()[0][0] or 0.0
                    pop_share = con.execute(f"SELECT {mass} FROM src", shifted).fetchall()[0][0] or 0.0
                associations.append(
                    _association(
                        "scale_shifted_file", fin_share, pop_share, cfg, column=m, files=shifted[:3], basis=basis
                    )
                )

        # 3) Timestamps outside their partition.
        part = next((c for c in info.partition_columns if type_family(schema.get(c, "")) == "temporal"), None)
        if part and time_col in schema:
            expr = f"avg((CAST({quote_ident(time_col)} AS DATE) <> CAST({quote_ident(part)} AS DATE))::DOUBLE)"
            fin_share = con.execute(f"SELECT {expr} FROM fin").fetchall()[0][0] or 0.0
            pop_share = con.execute(f"SELECT {expr} FROM src").fetchall()[0][0] or 0.0
            associations.append(_association("timestamp_outside_partition", fin_share, pop_share, cfg))

        # 4) Nulls concentrated in the finding.
        for col in schema:
            if col in (FILE_COLUMN,):
                continue
            q = quote_ident(col)
            fin_null = con.execute(f"SELECT avg(({q} IS NULL)::DOUBLE) FROM fin").fetchall()[0][0] or 0.0
            if fin_null == 0:
                continue
            pop_null = con.execute(f"SELECT avg(({q} IS NULL)::DOUBLE) FROM src").fetchall()[0][0] or 0.0
            associations.append(_association("nulls", fin_null, pop_null, cfg, column=col))
    finally:
        con.close()

    store = _store()
    if f.type != "temporal":
        members = _members(f)
        # 5) Cleaning operations (transform logs). Compared on ROWS: the share of the
        # finding's rows an operation touched vs the population's share. (Comparing
        # entities touched would be confounded by volume: an entity with many rows is
        # almost surely touched by a random deduplication.)
        for record in store.list("table"):
            if not record.id.startswith("transform_log_"):
                continue
            log = pd.read_parquet(store.path_of(record.id))
            if entity not in log.columns:
                continue
            for op_id, block in log.groupby("op_id"):
                in_finding = int(block[entity].isin(members).sum())
                associations.append(
                    _association(
                        "cleaning_operation",
                        in_finding / n_fin,
                        len(block) / n_pop,
                        cfg,
                        operation=op_id,
                        log=record.id,
                        basis="rows",
                    )
                )
        # 6) Features that were null (imputed when scaling).
        raw, _, entity_col = feature_tables()
        in_group = raw[entity_col].isin(members)
        # Low support: entities with very few events have unreliable features. This
        # does not make a finding an artefact, but it prevents "confirmed".
        if "n_events" in raw.columns:
            low_support_share = float(low_support(raw["n_events"])[in_group].mean())
        # Only features of the finding's space matter: a group found on one family of
        # features is not explained by nulls in other features.
        families = f.context.get("families")
        space = set(family_columns(list(raw.columns), families)) if families else set(raw.columns)
        for col in raw.columns:
            if col == entity_col or col not in space or not raw[col].isna().any():
                continue
            associations.append(
                _association(
                    "imputed_feature",
                    float(raw.loc[in_group, col].isna().mean()),
                    float(raw[col].isna().mean()),
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
    if f.type != "temporal" and "n_events" in feature_tables()[0].columns:
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
    table = f'"{spec.enriched_artifact}"'
    con = connect()
    try:
        relation = f"(SELECT * FROM {table}) ev"
        if rule.get("uses_ref"):
            primary = spec.primary().name if spec.primary() else None
            ref = next(
                (r for r in spec.references if r.kind == "reference" and r.dataset == r.ref_dataset == primary), None
            )
            if ref is None:
                return 0, 0
            columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()]
            ref_cols = ", ".join(f"p.{quote_ident(c)} AS {quote_ident('ref_' + c)}" for c in columns)
            relation = (
                f"(SELECT ev.*, {ref_cols} FROM {table} ev LEFT JOIN (SELECT DISTINCT * FROM {table}) p "
                f"ON ev.{quote_ident(ref.column)} = p.{quote_ident(ref.ref_column)}) ev"
            )
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
        raw, _, entity = feature_tables()
    except (FindingNotFoundError, SpecMissingError) as exc:
        return ToolResult.error(str(exc))
    store = _store()
    dictionary = store.load_json(spec.features.dictionary)["features"] if spec.features.dictionary else []
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
        relation = _rule_relation(con, spec)
        for name, rule in rules.items():
            off = (
                f"coalesce(({rule['where']}), false) AND abs(CAST(({rule['observed']}) AS DOUBLE) - "
                f"CAST(({rule['expected']}) AS DOUBLE)) > {float(rule['tolerance'])}"
            )
            breakers = con.execute(
                f"SELECT count(DISTINCT CAST(ev.{quote_ident(entity)} AS VARCHAR)) FROM {relation} WHERE {off} "
                f"AND CAST(ev.{quote_ident(entity)} AS VARCHAR) IN (SELECT m FROM members_df)"
            ).fetchall()[0][0]
            if breakers < 0.5 * len(members):
                continue  # the rule does not describe this group
            for key in keys:
                k = quote_ident(key)
                mine = con.execute(
                    f"SELECT CAST(ev.{k} AS VARCHAR), count(*) FROM {relation} WHERE {off} AND "
                    f"CAST(ev.{quote_ident(entity)} AS VARCHAR) IN (SELECT m FROM members_df) "
                    "GROUP BY 1 ORDER BY 2 DESC"
                ).fetchall()
                total_mine = sum(n for _, n in mine)
                if not total_mine or mine[0][0] is None:
                    continue
                value, n = mine[0]
                everyone = con.execute(
                    f"SELECT count(*), count(*) FILTER (WHERE CAST(ev.{k} AS VARCHAR) = ?) FROM {relation} WHERE {off}",
                    [value],
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


def _rule_relation(con: Any, spec: Any) -> str:
    """The enriched table aliased ``ev``, with ``ref_<column>`` of the referred row when there is a self-reference."""
    table = f'"{spec.enriched_artifact}"'
    primary = spec.primary().name if spec.primary() else None
    ref = next((r for r in spec.references if r.kind == "reference" and r.dataset == r.ref_dataset == primary), None)
    if ref is None:
        return f"(SELECT * FROM {table}) ev"
    columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()]
    ref_cols = ", ".join(f"p.{quote_ident(c)} AS {quote_ident('ref_' + c)}" for c in columns)
    return (
        f"(SELECT ev.*, {ref_cols} FROM {table} ev LEFT JOIN (SELECT DISTINCT * FROM {table}) p "
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
        scores = outlier_scores(matrix.to_numpy(), p["method"], p.get("method_params", {}), int(seed), cfg.lof_max_rows)
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
    cfg = _config().detection
    per_method, votes = [], np.zeros(n)
    for p in runs:
        scores = outlier_scores(X, p["method"], p.get("method_params", {}), int(seed), cfg.lof_max_rows)
        flags = flag_top(scores, p.get("contamination", cfg.contamination))[-n:]
        votes += flags
        per_method.append({"method": p["method"], "detection_rate": round(float(flags.mean()), 4)})
    return ToolResult.ok(
        {
            "kind": kind,
            "planted": n,
            "per_method": per_method,
            "detected_by_2_plus": round(float((votes >= 2).mean()), 4),
        }
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
        raw, scaled, entity = feature_tables()
        sig = signature(raw, scaled, entity, _members(f), f.top_features or None, f.rules)
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
        return ToolResult.error(f"Run {missing} before setting a verdict.")
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
    if verdict == "explained" and f.checks["stability"]["stability"] < minimum:
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
    if verdict == "confirmed" and f.checks["stability"]["stability"] < minimum:
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
