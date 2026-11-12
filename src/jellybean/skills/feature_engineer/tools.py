"""Feature engineer tools (spec section 7.4).

Builds the per-entity table from the (enriched) event table using one feature
template per semantic type:

| semantic type                     | features                                                        |
| --------------------------------- | --------------------------------------------------------------- |
| measure                           | sum, mean, std, max, median, p95; max daily sum                 |
| category (<= 30 levels)           | distinct, entropy, dominant share, share of each level (<= 15)  |
| timestamp (the time column)       | night/work-hours/weekend shares, hour entropy, gaps, bursts      |
| identifier of another entity      | distinct counterparties, distinct ratio, top share, new ratio    |
| constant per entity (enrichment)  | kept as an attribute (category) or numeric attribute             |

All aggregations run in DuckDB; only the per-entity table is loaded into
memory (for scaling and dimensionality reduction). A feature dictionary
(name, source column, template, formula) is stored next to every table.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from jellybean.runtime.context import current_run
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.skills._shared.artifacts import ArtifactNotFoundError
from jellybean.skills._shared.catalog import (
    SourceNotFoundError,
    connect,
    discover_datasets,
    quote_ident,
    resolve_source,
    sql_escape,
)
from jellybean.skills._shared.column_stats import describe
from jellybean.skills._shared.spec_tools import SpecMissingError, patch_spec, read_spec
from jellybean.skills.feature_engineer._common import (
    _entity_of,
    _feature_name,
    _load_table,
    _semantic_types,
    _store,
    latest_features,
)
from jellybean.skills.feature_engineer.models import (  # noqa: F401  (re-exported: the agent's tools)
    build_category_features,
    build_conformity_features,
    build_interaction_graph,
    build_linked_event_features,
    build_sequence_features,
)

Template = Literal["measure", "category", "timestamp", "counterparty"]
Window = Literal["all", "day"]
_MAX_CATEGORY_LEVELS = 30
_MAX_SHARE_LEVELS = 15
_ATTRIBUTE_CONSTANCY = 0.99  # share of entities where a column is constant -> attribute


# --------------------------------------------------------------------------- #
# build_entity_features                                                       #
# --------------------------------------------------------------------------- #
@tool
def build_entity_features(
    input: str,
    entity: str,
    time_column: str | None = None,
    windows: list[Window] | None = None,
    templates: list[Template] | None = None,
) -> ToolResult:
    """Build the per-entity feature table with DuckDB aggregations, plus a feature dictionary.

    Columns are handled by semantic type (from the spec, or inferred). Columns
    constant within each entity (e.g. enrichment attributes such as a segment)
    are kept as attributes. Grain and partition columns are excluded.

    Args:
        input: Enriched table artifact (or dataset) with one row per event.
        entity: Entity column (one output row per value).
        time_column: Event time column (enables timestamp, daily and new-counterparty features).
        windows: "all" (whole period) and/or "day" (daily aggregates: max/mean per day). Default both.
        templates: Feature templates to apply. Default all: measure, category, timestamp, counterparty.
    """
    windows = windows or ["all", "day"]
    templates = templates or ["measure", "category", "timestamp", "counterparty"]
    try:
        spec = read_spec()
        source = resolve_source(input)
    except (SpecMissingError, SourceNotFoundError) as exc:
        return ToolResult.error(str(exc))
    con = connect()
    try:
        schema = dict(describe(con, source.view))
        for col in [entity] + ([time_column] if time_column else []):
            if col not in schema:
                return ToolResult.error(f"Unknown column '{col}'. Columns: {list(schema)}")
        partitions: set[str] = set()
        for info in discover_datasets(current_run().settings.data_root).values():
            partitions |= set(info.partition_columns)
        grain = set(spec.grain.columns) if spec.grain else set()
        candidates = [c for c in schema if c not in {entity, time_column} | partitions | grain]
        types = _semantic_types(spec, con, source.view, candidates)
        e, view = quote_ident(entity), source.view
        con.execute(f"CREATE TEMP TABLE ev AS SELECT * FROM {view} WHERE {e} IS NOT NULL")

        # Attributes: columns constant within (almost) every entity.
        attr_candidates = [c for c in candidates if types[c] in ("category", "boolean", "measure", "timestamp")]
        attributes: list[str] = []
        if attr_candidates:
            exprs = ", ".join(f"count(DISTINCT {quote_ident(c)}) <= 1 AS c{i}" for i, c in enumerate(attr_candidates))
            share = con.execute(
                f"SELECT {', '.join(f'avg(c{i}::INT)' for i in range(len(attr_candidates)))} "
                f"FROM (SELECT {exprs} FROM ev GROUP BY {e})"
            ).fetchall()[0]
            attributes = [c for c, s in zip(attr_candidates, share, strict=True) if s >= _ATTRIBUTE_CONSTANCY]

        select: list[str] = ["count(*) AS n_events"]
        dictionary: list[dict[str, str]] = [
            {"name": "n_events", "source": "*", "template": "count", "formula": "number of rows per entity"}
        ]

        def feat(expr: str, name: str, src: str, template: str, formula: str) -> None:
            select.append(f"{expr} AS {quote_ident(name)}")
            dictionary.append({"name": name, "source": src, "template": template, "formula": formula})

        t = quote_ident(time_column) if time_column else None
        max_time = con.execute(f"SELECT max({t}) FROM ev").fetchall()[0][0] if t else None
        measures, categories, counterparties = [], [], []
        for col in candidates:
            q, kind = quote_ident(col), types[col]
            if col in attributes:
                if kind == "measure":
                    feat(f"any_value({q})", _feature_name("attr", col), col, "attribute", "constant value per entity")
                elif kind == "timestamp" and max_time is not None:
                    feat(
                        f"date_diff('day', CAST(any_value({q}) AS TIMESTAMP), TIMESTAMP '{max_time}')",
                        _feature_name("attr", col, "age_days"),
                        col,
                        "attribute",
                        "days from the value to the end of the period",
                    )
                else:
                    feat(
                        f"CAST(mode({q}) AS VARCHAR)",
                        _feature_name("attr", col),
                        col,
                        "attribute",
                        "constant value per entity",
                    )
                continue
            if kind == "measure" and "measure" in templates:
                measures.append(col)
                for agg, label in (("sum", "sum"), ("avg", "mean"), ("stddev_samp", "std"), ("max", "max")):
                    feat(f"{agg}({q})", _feature_name(col, label), col, "measure", f"{label} of {col}")
                feat(f"quantile_cont({q}, 0.5)", _feature_name(col, "p50"), col, "measure", f"median of {col}")
                feat(
                    f"quantile_cont({q}, 0.95)", _feature_name(col, "p95"), col, "measure", f"95th percentile of {col}"
                )
            elif kind in ("category", "boolean") and "category" in templates:
                levels = con.execute(
                    f"SELECT CAST({q} AS VARCHAR) v, count(*) n FROM ev WHERE {q} IS NOT NULL "
                    f"GROUP BY 1 ORDER BY 2 DESC, 1"
                ).fetchall()
                if len(levels) > _MAX_CATEGORY_LEVELS:
                    counterparties.append(col)
                    continue
                categories.append(col)
                feat(
                    f"count(DISTINCT {q})", _feature_name(col, "distinct"), col, "category", f"distinct values of {col}"
                )
                feat(
                    f"coalesce(entropy({q}), 0)",
                    _feature_name(col, "entropy"),
                    col,
                    "category",
                    f"entropy (bits) of {col}",
                )
                if len(levels) <= _MAX_SHARE_LEVELS:
                    for value, _ in levels:
                        feat(
                            f"avg((CAST({q} AS VARCHAR) = '{sql_escape(value)}')::DOUBLE)",
                            _feature_name(col, "share", value),
                            col,
                            "category",
                            f"share of rows with {col} = {value}",
                        )
            elif kind == "identifier" and "counterparty" in templates:
                counterparties.append(col)
            elif kind == "timestamp":
                continue  # other timestamps that vary per entity are not aggregated

        if t and "timestamp" in templates:
            feat(
                f"avg((hour({t}) < 6)::DOUBLE)",
                "time_share_night",
                time_column,
                "timestamp",
                "share of events 00:00-05:59",
            )
            feat(
                f"avg((isodow({t}) <= 5 AND hour({t}) BETWEEN 9 AND 17)::DOUBLE)",
                "time_share_work_hours",
                time_column,
                "timestamp",
                "share of events Mon-Fri 09:00-17:59",
            )
            feat(
                f"avg((isodow({t}) >= 6)::DOUBLE)",
                "time_share_weekend",
                time_column,
                "timestamp",
                "share of events on weekends",
            )
            feat(
                f"coalesce(entropy(hour({t})), 0)",
                "time_hour_entropy",
                time_column,
                "timestamp",
                "entropy of the hour of day",
            )
            feat(
                f"count(DISTINCT CAST({t} AS DATE))",
                "time_active_days",
                time_column,
                "timestamp",
                "days with at least one event",
            )

        con.execute(f"CREATE TEMP TABLE f_main AS SELECT {e}, {', '.join(select)} FROM ev GROUP BY {e}")
        joins = []

        if t and "timestamp" in templates:
            con.execute(f"""CREATE TEMP TABLE f_gaps AS SELECT {e},
                    avg(gap) AS time_gap_mean_s, quantile_cont(gap, 0.5) AS time_gap_p50_s, min(gap) AS time_gap_min_s,
                    CASE WHEN avg(gap) + stddev_samp(gap) > 0
                         THEN (stddev_samp(gap) - avg(gap)) / (stddev_samp(gap) + avg(gap)) END AS time_burstiness
                FROM (SELECT {e}, epoch({t}) - lag(epoch({t})) OVER (PARTITION BY {e} ORDER BY {t}) AS gap FROM ev)
                WHERE gap IS NOT NULL GROUP BY {e}""")
            con.execute(f"""CREATE TEMP TABLE f_hour AS SELECT {e}, max(n) AS time_max_events_per_hour
                FROM (SELECT {e}, date_trunc('hour', {t}) h, count(*) n FROM ev GROUP BY 1, 2) GROUP BY {e}""")
            joins += ["f_gaps", "f_hour"]
            for name, formula in (
                ("time_gap_mean_s", "mean seconds between consecutive events"),
                ("time_gap_p50_s", "median seconds between consecutive events"),
                ("time_gap_min_s", "minimum seconds between consecutive events"),
                ("time_burstiness", "(std - mean) / (std + mean) of gaps"),
                ("time_max_events_per_hour", "maximum events in one clock hour"),
            ):
                dictionary.append({"name": name, "source": time_column, "template": "timestamp", "formula": formula})

        if t and "day" in windows:
            day_sums = ", ".join(f"sum({quote_ident(m)}) AS {quote_ident('s_' + m)}" for m in measures)
            day_aggs = ", ".join(
                f"max({quote_ident('s_' + m)}) AS {quote_ident(_feature_name(m, 'max_daily_sum'))}" for m in measures
            )
            con.execute(f"""CREATE TEMP TABLE f_day AS SELECT {e}, max(n) AS day_max_events, avg(n) AS day_mean_events
                    {", " + day_aggs if measures else ""}
                FROM (SELECT {e}, CAST({t} AS DATE) d, count(*) n {", " + day_sums if measures else ""}
                      FROM ev GROUP BY 1, 2) GROUP BY {e}""")
            joins.append("f_day")
            dictionary += [
                {
                    "name": "day_max_events",
                    "source": time_column,
                    "template": "window:day",
                    "formula": "maximum events in one day",
                },
                {
                    "name": "day_mean_events",
                    "source": time_column,
                    "template": "window:day",
                    "formula": "mean events per active day",
                },
            ]
            dictionary += [
                {
                    "name": _feature_name(m, "max_daily_sum"),
                    "source": m,
                    "template": "window:day",
                    "formula": f"maximum daily sum of {m}",
                }
                for m in measures
            ]

        if categories and "category" in templates:
            for col in categories:
                q, name = quote_ident(col), _feature_name(col, "dominant_share")
                table = f"f_dom_{len(joins)}"
                con.execute(f"""CREATE TEMP TABLE {table} AS SELECT {e}, max(n) / sum(n) AS {quote_ident(name)}
                    FROM (SELECT {e}, {q}, count(*) n FROM ev WHERE {q} IS NOT NULL GROUP BY 1, 2) GROUP BY {e}""")
                joins.append(table)
                dictionary.append(
                    {
                        "name": name,
                        "source": col,
                        "template": "category",
                        "formula": f"share of the most frequent {col} value",
                    }
                )

        if counterparties and "counterparty" in templates:
            mid = None
            if t:
                lo, hi = con.execute(f"SELECT min(epoch({t})), max(epoch({t})) FROM ev").fetchall()[0]
                mid = (lo + hi) / 2 if lo is not None else None
            for col in counterparties:
                q, table = quote_ident(col), f"f_cp_{len(joins)}"
                names = {k: _feature_name(col, k) for k in ("distinct", "distinct_ratio", "top_share", "new_ratio")}
                first_seen = f", min(epoch({t})) AS first_t" if t else ""
                new_ratio = (
                    f", avg((first_t > {mid})::DOUBLE) AS {quote_ident(names['new_ratio'])}" if mid is not None else ""
                )
                con.execute(f"""CREATE TEMP TABLE {table} AS SELECT {e},
                        count(*) AS {quote_ident(names["distinct"])},
                        count(*) / sum(n) AS {quote_ident(names["distinct_ratio"])},
                        max(n) / sum(n) AS {quote_ident(names["top_share"])} {new_ratio}
                    FROM (SELECT {e}, {q}, count(*) n {first_seen} FROM ev WHERE {q} IS NOT NULL GROUP BY 1, 2)
                    GROUP BY {e}""")
                joins.append(table)
                dictionary += [
                    {
                        "name": names["distinct"],
                        "source": col,
                        "template": "counterparty",
                        "formula": f"distinct {col}",
                    },
                    {
                        "name": names["distinct_ratio"],
                        "source": col,
                        "template": "counterparty",
                        "formula": f"distinct {col} / events with {col}",
                    },
                    {
                        "name": names["top_share"],
                        "source": col,
                        "template": "counterparty",
                        "formula": f"share of events with the most frequent {col}",
                    },
                ]
                if mid is not None:
                    dictionary.append(
                        {
                            "name": names["new_ratio"],
                            "source": col,
                            "template": "counterparty",
                            "formula": f"share of distinct {col} first seen in the second half of the period",
                        }
                    )

        join_sql = " ".join(f"LEFT JOIN {j} USING ({e})" for j in joins)
        query = f"SELECT * FROM f_main {join_sql} ORDER BY {e}"
        store = _store()
        record = store.save_table_from_sql(
            con,
            query,
            f"features_{_feature_name(entity)}",
            parents=[source.name],
            params={
                "entity": entity,
                "time_column": time_column,
                "windows": windows,
                "templates": templates,
                "attributes": attributes,
            },
        )
        n_columns = len(describe(con, f"read_parquet('{sql_escape(str(store.path_of(record.id)))}')"))
    finally:
        con.close()
    dict_record = store.save_json("feature_dictionary", {"entity": entity, "features": dictionary}, parents=[record.id])
    patch_spec(
        {
            "features": {
                "table": record.id,
                "dictionary": dict_record.id,
                "scaled": None,
                "embedding": None,
                "graph": None,
            }
        }
    )
    by_template: dict[str, int] = {}
    for d in dictionary:
        by_template[d["template"]] = by_template.get(d["template"], 0) + 1
    return ToolResult.ok(
        {
            "entities": record.row_count,
            "features": n_columns - 1,
            "by_template": by_template,
            "attributes": attributes,
            "counterparty_columns": counterparties,
            "category_columns": categories,
            "measure_columns": measures,
            "dictionary": dict_record.id,
        },
        artifact_id=record.id,
    )


# --------------------------------------------------------------------------- #
# build_graph                                                                 #
# --------------------------------------------------------------------------- #
@tool
def build_graph(
    input: str,
    entity: str,
    link_columns: Annotated[list[str], Field(min_length=1)],
    max_entities_per_value: Annotated[int, Field(ge=2, le=1000)] = 25,
    features: str | None = None,
) -> ToolResult:
    """Link entities that share values of `link_columns` (devices, addresses, accounts) and compute graph features.

    Values shared by more than `max_entities_per_value` entities are ignored
    (e.g. a cell tower is not evidence of a relationship). Node features:
    degree, weighted degree, component size and id, clustering. If `features`
    is given, a new feature-table version with the graph features is created.

    Args:
        input: Event table (enriched artifact or dataset).
        entity: Entity column.
        link_columns: Columns whose shared values link entities (devices/addresses, not counterparties/locations).
        max_entities_per_value: Ignore values shared by more entities than this.
        features: Optional feature table to extend with the graph features.
    """
    import networkx as nx

    try:
        source = resolve_source(input)
    except SourceNotFoundError as exc:
        return ToolResult.error(str(exc))
    con = connect()
    store = _store()
    try:
        schema = dict(describe(con, source.view))
        missing = [c for c in [entity, *link_columns] if c not in schema]
        if missing:
            return ToolResult.error(f"Unknown columns: {missing}")
        e = quote_ident(entity)
        link_stats, parts = [], []
        for i, col in enumerate(link_columns):
            q = quote_ident(col)
            con.execute(
                f"CREATE TEMP TABLE p{i} AS SELECT DISTINCT {e} AS ent, CAST({q} AS VARCHAR) AS v "
                f"FROM {source.view} WHERE {e} IS NOT NULL AND {q} IS NOT NULL"
            )
            values, shared, too_common, median = con.execute(
                f"SELECT count(*), count(*) FILTER (WHERE n BETWEEN 2 AND {max_entities_per_value}), "
                f"count(*) FILTER (WHERE n > {max_entities_per_value}), median(n) "
                f"FROM (SELECT v, count(*) n FROM p{i} GROUP BY v)"
            ).fetchall()[0]
            link_stats.append(
                {
                    "column": col,
                    "values": values,
                    "shared_values_used": shared,
                    "values_ignored_too_common": too_common,
                    "median_entities_per_value": median,
                }
            )
            parts.append(f"""SELECT a.ent AS src, b.ent AS dst, count(*) AS w FROM p{i} a JOIN p{i} b USING (v)
                WHERE a.ent < b.ent AND v IN (SELECT v FROM p{i} GROUP BY v
                                              HAVING count(*) BETWEEN 2 AND {max_entities_per_value})
                GROUP BY 1, 2""")
        con.execute(
            f"CREATE TEMP TABLE edges AS SELECT src, dst, sum(w) AS weight FROM ({' UNION ALL '.join(parts)}) "
            "GROUP BY 1, 2 ORDER BY 1, 2"
        )
        edges = con.execute("SELECT src, dst, weight FROM edges").df()
        all_entities = con.execute(f"SELECT DISTINCT {e} FROM {source.view} WHERE {e} IS NOT NULL").df()[entity]
    finally:
        con.close()

    graph = nx.Graph()
    graph.add_weighted_edges_from(edges.itertuples(index=False, name=None))
    components = sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(map(str, c))))
    comp_of = {node: idx + 1 for idx, comp in enumerate(components) for node in comp}
    clustering = nx.clustering(graph) if graph.number_of_nodes() else {}
    rows = []
    for node in all_entities:
        if node in graph:
            rows.append(
                (
                    node,
                    graph.degree(node),
                    graph.degree(node, weight="weight"),
                    len(components[comp_of[node] - 1]),
                    comp_of[node],
                    clustering.get(node, 0.0),
                )
            )
        else:
            rows.append((node, 0, 0, 1, 0, 0.0))
    nodes = pd.DataFrame(
        rows,
        columns=[
            entity,
            "graph_degree",
            "graph_weighted_degree",
            "graph_component_size",
            "graph_component_id",
            "graph_clustering",
        ],
    )
    edge_record = store.save_dataframe(
        "graph_edges",
        edges,
        parents=[source.name],
        params={"entity": entity, "link_columns": link_columns, "max_entities_per_value": max_entities_per_value},
    )
    node_record = store.save_dataframe("graph_features", nodes, parents=[edge_record.id], params={"entity": entity})
    patch: dict[str, Any] = {"features": {"graph": edge_record.id}}
    new_table = None
    if features:
        try:
            features, _ = latest_features(features)  # several extensions in one turn: no fork
            base = _load_table(features)
            if entity not in base.columns:
                return ToolResult.error(f"Feature table '{features}' has no column '{entity}'.")
            merged = base.drop(columns=[c for c in nodes.columns if c != entity and c in base.columns])
            merged = merged.merge(nodes.drop(columns=["graph_component_id"]), on=entity, how="left")
            params = dict(store.get(features).params)
            new_table = store.save_dataframe(
                f"features_{_feature_name(entity)}", merged, parents=[features, node_record.id], params=params
            ).id
            spec = read_spec()
            dictionary = store.load_json(spec.features.dictionary)["features"] if spec.features.dictionary else []
            dictionary = [d for d in dictionary if not d["name"].startswith("graph_")] + [
                {"name": n, "source": ",".join(link_columns), "template": "graph", "formula": f}
                for n, f in (
                    ("graph_degree", "entities sharing a link value"),
                    ("graph_weighted_degree", "shared link values with other entities"),
                    ("graph_component_size", "size of the connected component"),
                    ("graph_clustering", "local clustering coefficient"),
                )
            ]
            dict_id = store.save_json(
                "feature_dictionary", {"entity": entity, "features": dictionary}, parents=[new_table]
            ).id
            patch["features"] |= {"table": new_table, "dictionary": dict_id, "scaled": None, "embedding": None}
        except (ArtifactNotFoundError, FileNotFoundError) as exc:
            return ToolResult.error(f"Unknown feature table: {exc}")
    patch_spec(patch)
    sizes = [len(c) for c in components]
    return ToolResult.ok(
        {
            "nodes_with_links": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "components": len(components),
            "largest_components": sizes[:8],
            "components_size_3_plus": sum(1 for s in sizes if s >= 3),
            "link_columns": link_stats,
            "graph_features": node_record.id,
            "features_table": new_table,
        },
        artifact_id=edge_record.id,
    )


# --------------------------------------------------------------------------- #
# scale_encode / reduce_dimensions / feature_summary                          #
# --------------------------------------------------------------------------- #
@tool
def scale_encode(
    features: str, strategy: Literal["robust_log", "robust", "standard", "minmax"] = "robust_log"
) -> ToolResult:
    """Scale numeric features and one-hot encode attributes, producing a model-ready table.

    Nulls are imputed with the median (per feature), constant features are
    dropped. "robust_log" applies log1p to skewed non-negative features before
    robust scaling (median / IQR), which keeps heavy tails from dominating.

    Args:
        features: Feature table artifact id.
        strategy: robust_log (default), robust, standard or minmax.
    """
    try:
        entity = _entity_of(features)
        df = _load_table(features)
    except (ArtifactNotFoundError, FileNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    ids = df[entity]
    numeric = df.drop(columns=[entity]).select_dtypes(include=["number", "bool"]).astype(float)
    categorical = df.drop(columns=[entity]).select_dtypes(exclude=["number", "bool"])
    imputed = {c: int(numeric[c].isna().sum()) for c in numeric.columns if numeric[c].isna().any()}
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    constant = [c for c in numeric.columns if numeric[c].nunique() <= 1]
    numeric = numeric.drop(columns=constant)
    log_cols = []
    if strategy == "robust_log":
        for c in numeric.columns:
            col = numeric[c]
            if col.min() >= 0 and col.skew() > 1:
                numeric[c] = np.log1p(col)
                log_cols.append(c)
    if strategy in ("robust", "robust_log"):
        center = numeric.median()
        scale = numeric.quantile(0.75) - numeric.quantile(0.25)
        scale = scale.where(scale > 0, numeric.std()).replace(0, 1.0)
    elif strategy == "standard":
        center, scale = numeric.mean(), numeric.std().replace(0, 1.0)
    else:
        center, scale = numeric.min(), (numeric.max() - numeric.min()).replace(0, 1.0)
    scaled = (numeric - center) / scale
    encoded = []
    for c in categorical.columns:
        values = categorical[c].astype("string").fillna("__missing__")
        top = values.value_counts().index[:20]
        values = values.where(values.isin(top), "__other__")
        dummies = pd.get_dummies(values, prefix=c, dtype=float)
        dummies.columns = [_feature_name(col) for col in dummies.columns]
        encoded.append(dummies)
    out = pd.concat(
        [ids.reset_index(drop=True), scaled.reset_index(drop=True), *[d.reset_index(drop=True) for d in encoded]],
        axis=1,
    )
    store = _store()
    record = store.save_dataframe(
        "features_scaled",
        out,
        parents=[features],
        params={
            "entity": entity,
            "strategy": strategy,
            "log_transformed": log_cols,
            "dropped_constant": constant,
            "center": center.round(6).to_dict(),
            "scale": scale.round(6).to_dict(),
        },
    )
    patch_spec({"features": {"scaled": record.id, "embedding": None}})
    return ToolResult.ok(
        {
            "entities": len(out),
            "numeric_features": scaled.shape[1],
            "one_hot_columns": sum(d.shape[1] for d in encoded),
            "log_transformed": len(log_cols),
            "dropped_constant": constant,
            "imputed_nulls": imputed,
        },
        artifact_id=record.id,
    )


@tool
def reduce_dimensions(
    features: str,
    method: Literal["pca", "umap"] = "pca",
    n_components: Annotated[int, Field(ge=2, le=50)] = 10,
    seed: int | None = None,
) -> ToolResult:
    """Project a scaled feature table onto fewer dimensions (PCA, or UMAP if installed).

    Args:
        features: Scaled feature table (from scale_encode).
        method: "pca" (default) or "umap" (needs `uv sync --extra explain`).
        n_components: Number of output dimensions.
        seed: Random seed (defaults to the run seed).
    """
    try:
        entity = _entity_of(features)
        df = _load_table(features)
    except (ArtifactNotFoundError, FileNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    X = df.drop(columns=[entity]).to_numpy(dtype=float)
    if np.isnan(X).any():
        return ToolResult.error("Input has nulls: run scale_encode first.")
    k = min(n_components, X.shape[1], X.shape[0])
    summary: dict[str, Any] = {"method": method, "n_components": k, "entities": len(df)}
    names = list(df.columns.drop(entity))
    if method == "pca":
        from sklearn.decomposition import PCA

        model = PCA(n_components=k, random_state=seed)
        Z = model.fit_transform(X)
        ratios = model.explained_variance_ratio_
        summary["explained_variance"] = [round(float(r), 4) for r in ratios]
        summary["cumulative_explained"] = round(float(ratios.sum()), 4)
        summary["top_loadings"] = {
            f"c{i + 1}": [
                [names[j], round(float(model.components_[i][j]), 3)]
                for j in np.argsort(-np.abs(model.components_[i]))[:5]
            ]
            for i in range(min(3, k))
        }
    else:
        try:
            import umap
        except ImportError:
            return ToolResult.error("UMAP is not installed: run `uv sync --extra explain` or use method 'pca'.")
        Z = umap.UMAP(n_components=k, random_state=seed).fit_transform(X)
    out = pd.DataFrame(Z, columns=[f"c{i + 1}" for i in range(k)])
    out.insert(0, entity, df[entity].to_numpy())
    store = _store()
    record = store.save_dataframe(
        f"embedding_{method}",
        out,
        parents=[features],
        params={"entity": entity, "method": method, "n_components": k, "seed": seed},
    )
    patch_spec({"features": {"embedding": record.id}})
    return ToolResult.ok(summary, artifact_id=record.id)


@tool
def feature_summary(features: str) -> ToolResult:
    """Summarise a feature table: nulls, variance, constant features, highly correlated pairs, skew.

    Args:
        features: Feature table artifact id (raw or scaled).
    """
    try:
        entity = _entity_of(features)
        df = _load_table(features)
    except (ArtifactNotFoundError, FileNotFoundError, ValueError) as exc:
        return ToolResult.error(str(exc))
    data = df.drop(columns=[entity])
    numeric = data.select_dtypes(include=["number", "bool"]).astype(float)
    nulls = {c: round(float(v), 4) for c, v in numeric.isna().mean().items() if v > 0}
    constant = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]
    corr = numeric.drop(columns=constant).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack()
    high = pairs[pairs >= 0.95].sort_values(ascending=False)
    skew = numeric.drop(columns=constant).skew().abs().sort_values(ascending=False)
    return ToolResult.ok(
        {
            "entities": len(df),
            "numeric_features": numeric.shape[1],
            "categorical_features": list(data.columns.difference(numeric.columns)),
            "features_with_nulls": dict(sorted(nulls.items(), key=lambda kv: -kv[1])[:10]),
            "constant_features": constant,
            "highly_correlated_pairs": [[a, b, round(float(v), 3)] for (a, b), v in high.head(10).items()],
            "n_highly_correlated_pairs": len(high),
            "most_skewed": [[c, round(float(v), 1)] for c, v in skew.head(5).items()],
        }
    )
