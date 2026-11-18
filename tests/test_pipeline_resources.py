# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Memory and CPU bounds of the stages after the profiler (features, detection, critic, analyst).

Two kinds of tests: the bounded code paths give the same result as the simple
ones they replace (reference implementations are kept here), and the paths
that only start above a size threshold work and say what they did.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from jellybean.runtime.context import RunContext, use_run
from jellybean.skills._shared import detection, findings
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.findings import all_findings, finding_entities, load_finding
from jellybean.skills._shared.spec_tools import read_spec


# --------------------------------------------------------------------- helpers
@pytest.fixture
def run_copy(phase5: dict) -> Any:
    """A private copy of the phase-5 run: tools may change its findings."""
    ctx: RunContext = phase5["ctx"]
    target = ctx.settings.workspace_root / "resources_copy"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ctx.run_dir, target)
    for path in target.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    spec = yaml.safe_load((target / "spec.yaml").read_text())
    spec["run_id"] = "resources_copy"
    (target / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    copy = RunContext.open(ctx.settings, "resources_copy", seed=ctx.seed)
    with use_run(copy):
        yield copy
    shutil.rmtree(target, ignore_errors=True)


def _entity_finding() -> Any:
    return next(f for f in all_findings() if f.type != "temporal")


def _members(f: Any) -> set[Any]:
    return set(finding_entities(f)[f.entity_column])


# ------------------------------------------------------- feature table reading
def test_read_features_reads_only_the_members_and_columns(tmp_path: Path) -> None:
    table = pd.DataFrame({"id": [f"e{i}" for i in range(50)], "a": np.arange(50.0), "b": np.arange(50) % 3})
    path = tmp_path / "t.parquet"
    table.to_parquet(path, index=False)
    got = findings.read_features(path, "id", columns=["id", "a"], members={"e7", "e3", "missing"})
    assert got.to_dict("list") == {"id": ["e3", "e7"], "a": [3.0, 7.0]}  # file order, only what was asked
    assert findings.read_features(path, "id", members=[]).shape == (0, 3)
    assert findings.read_features(path, "id").equals(table)


def test_read_features_compares_ids_as_text_when_types_differ(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    pd.DataFrame({"id": [10, 11, 12], "a": [1.0, 2.0, 3.0]}).to_parquet(path, index=False)
    got = findings.read_features(path, "id", members={"11", "12"})
    assert got["id"].tolist() == [11, 12]


def test_group_statistics_match_the_full_table_versions(run_copy: RunContext) -> None:
    f = _entity_finding()
    members = _members(f)
    raw, scaled, entity = findings.feature_tables()
    assert findings.group_effect_sizes(members, top=10) == findings.effect_sizes(raw, entity, members, top=10)
    assert findings.group_signature(members, f.top_features, f.rules) == findings.signature(
        raw, scaled, entity, members, f.top_features, f.rules
    )


def test_large_populations_use_statistics_computed_once(run_copy: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    f = _entity_finding()
    members = _members(f)
    exact = findings.group_effect_sizes(members, top=5)
    monkeypatch.setattr(run_copy.settings.config.detection, "exact_population_max_rows", 0)
    findings.clear_population_cache()
    reads: list[Any] = []
    real = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: reads.append(a) or real(*a, **k))
    approx = findings.group_effect_sizes(members, top=5)
    assert reads  # the population statistics were computed, a few columns at a time
    reads.clear()
    again = findings.group_effect_sizes(members, top=5)
    assert not reads  # the next group reads its own rows only: the statistics are reused
    assert again == approx
    # A small group barely moves the population: same features, same directions.
    assert [(t["feature"], t["direction"]) for t in approx[:3]] == [(t["feature"], t["direction"]) for t in exact[:3]]


# ------------------------------------------------------------ outlier methods
@pytest.mark.parametrize("method", ["ecod", "hbos"])
def test_columnwise_scores_equal_the_whole_matrix_scores(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    X = np.random.default_rng(3).normal(size=(400, 7))
    X[:5] += 6
    whole = detection.outlier_scores(X, method, {}, 1, 50_000)
    monkeypatch.setattr(detection, "_COLUMNWISE_CELLS", 800)  # two columns at a time
    assert np.allclose(detection.outlier_scores(X, method, {}, 1, 50_000), whole)


def test_lof_sample_shrinks_with_the_number_of_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from sklearn.neighbors import LocalOutlierFactor

    fitted: list[int] = []
    real_fit = LocalOutlierFactor.fit
    monkeypatch.setattr(
        LocalOutlierFactor, "fit", lambda self, X, y=None: fitted.append(len(X)) or real_fit(self, X, y)
    )
    monkeypatch.setattr(detection, "_LOF_MIN_SAMPLE", 100)
    monkeypatch.setattr(detection, "_LOF_MAX_PAIRS", 2000 * 150)
    X = np.random.default_rng(0).normal(size=(2000, 3))
    scores = detection.outlier_scores(X, "lof", {"n_neighbors": 10}, 1, lof_max_rows=500)
    assert fitted == [150] and scores.shape == (2000,)


def test_life_grid_equals_one_date_range_per_group() -> None:
    step = pd.Timedelta(hours=1)
    base = pd.Timestamp("2026-01-01")
    active = pd.DataFrame(
        {"min": [base, base + 5 * step, base + 2 * step], "max": [base + 3 * step, base + 5 * step, base + 9 * step]},
        index=pd.Index(["a", "b", "c"], name="group"),
    ).astype("datetime64[us]")
    expected = pd.concat(
        [
            pd.DataFrame({"group": g, "ts": pd.date_range(row["min"], row["max"], freq=step)})
            for g, row in active.iterrows()
        ],
        ignore_index=True,
    )
    got = detection._life_grid(active, step)
    assert got["group"].tolist() == expected["group"].tolist()
    assert got["ts"].tolist() == expected["ts"].tolist()
    assert got["ts"].dtype == active["min"].dtype


# --------------------------------------------------------------------- graphs
def _random_edges(seed: int, ids: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 120, 300), rng.integers(0, 120, 300)
    edges = pd.DataFrame({"src": ids[a[a < b]], "dst": ids[b[a < b]]}).drop_duplicates()
    edges = edges.sort_values(["src", "dst"]).reset_index(drop=True)
    return edges.assign(weight=rng.integers(1, 5, len(edges)))


@pytest.mark.parametrize("ids", [np.array([f"e{i:03d}" for i in range(200)]), np.arange(1000, 1200)])
def test_sparse_graph_features_equal_networkx(ids: np.ndarray) -> None:
    import networkx as nx

    from jellybean.skills.feature_engineer.tools import _graph_nodes

    edges = _random_edges(5, ids)
    entities = pd.Series(np.random.default_rng(1).permutation(ids), name="ent")
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges.itertuples(index=False, name=None))
    components = sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(map(str, c))))
    comp_of = {node: idx + 1 for idx, comp in enumerate(components) for node in comp}
    clustering = nx.clustering(graph)
    rows = [
        (
            n,
            graph.degree(n),
            graph.degree(n, weight="weight"),
            len(components[comp_of[n] - 1]),
            comp_of[n],
            clustering[n],
        )
        if n in graph
        else (n, 0, 0, 1, 0, 0.0)
        for n in entities
    ]
    nodes, sizes = _graph_nodes(edges, entities, "ent")
    assert nodes.equals(pd.DataFrame(rows, columns=nodes.columns))
    assert sizes == [len(c) for c in components]
    # Components as the detector numbers them (ties by the smallest member).
    groups = sorted((sorted(c) for c in components), key=lambda c: (-len(c), str(c[0])))
    expected = [(node, gid, len(c)) for gid, c in enumerate(groups, 1) for node in c]
    table = detection.component_table(edges, "ent")
    assert list(table.itertuples(index=False, name=None)) == expected


def test_graph_without_edges_gives_every_entity_default_features() -> None:
    from jellybean.skills.feature_engineer.tools import _graph_nodes

    edges = pd.DataFrame({"src": pd.Series(dtype=str), "dst": pd.Series(dtype=str), "weight": pd.Series(dtype=int)})
    nodes, sizes = _graph_nodes(edges, pd.Series(["a", "b"]), "ent")
    assert sizes == [] and nodes["graph_component_size"].tolist() == [1, 1] and nodes["graph_degree"].sum() == 0


def test_graph_methods_that_do_not_scale_refuse_large_graphs(
    run_copy: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jellybean.skills.detector import tools as det

    graph = read_spec().features.graph
    monkeypatch.setattr(run_copy.settings.config.detection, "graph_max_edges", 1000)
    monkeypatch.setattr(det.pq, "read_metadata", lambda path: type("M", (), {"num_rows": 5000})())
    refused = det.run_graph_analysis(graph, "communities")
    assert refused.status == "error" and "graph_max_edges" in refused.summary["error"]
    assert det.run_graph_analysis(graph, "components").status == "ok"  # components always scale


# ------------------------------------------------------------------- scaling
def _reference_scale(df: pd.DataFrame, entity: str, strategy: str) -> pd.DataFrame:
    """The whole-table scaling that _scale_columns replaces."""
    from jellybean.skills.feature_engineer._common import _feature_name

    numeric = df.drop(columns=[entity]).select_dtypes(include=["number", "bool"]).astype(float)
    categorical = df.drop(columns=[entity]).select_dtypes(exclude=["number", "bool"])
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    numeric = numeric.drop(columns=[c for c in numeric.columns if numeric[c].nunique() <= 1])
    if strategy == "robust_log":
        for c in numeric.columns:
            if numeric[c].min() >= 0 and numeric[c].skew() > 1:
                numeric[c] = np.log1p(numeric[c])
    if strategy in ("robust", "robust_log"):
        center = numeric.median()
        scale = numeric.quantile(0.75) - numeric.quantile(0.25)
        scale = scale.where(scale > 0, numeric.std()).replace(0, 1.0)
    elif strategy == "standard":
        center, scale = numeric.mean(), numeric.std().replace(0, 1.0)
    else:
        center, scale = numeric.min(), (numeric.max() - numeric.min()).replace(0, 1.0)
    encoded = []
    for c in categorical.columns:
        values = categorical[c].astype("string").fillna("__missing__")
        values = values.where(values.isin(values.value_counts().index[:20]), "__other__")
        dummies = pd.get_dummies(values, prefix=c, dtype=float)
        dummies.columns = [_feature_name(col) for col in dummies.columns]
        encoded.append(dummies)
    return pd.concat([df[entity], (numeric - center) / scale, *encoded], axis=1)


@pytest.mark.parametrize("strategy", ["robust_log", "robust", "standard", "minmax"])
def test_columnwise_scaling_equals_whole_table_scaling(
    strategy: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jellybean.skills.feature_engineer import tools as fe

    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame(
        {
            "id": [f"e{i}" for i in range(n)],
            "skewed": rng.lognormal(0, 2, n),
            "attr_s": rng.choice(["x", "y", "z z", None], n),
            "normal": rng.normal(0, 1, n),
            "constant": 1.0,
            "with_nulls": np.where(rng.random(n) < 0.1, np.nan, rng.exponential(3, n)),
            "flag": rng.random(n) < 0.2,
            "all_null": np.nan,
            "mostly_zero": np.where(rng.random(n) < 0.9, 0.0, rng.random(n)),
            "attr_many": [f"v{i % 40}" for i in range(n)],
        }
    )
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    expected = _reference_scale(pd.read_parquet(path), "id", strategy)
    monkeypatch.setattr(fe, "_SCALE_CHUNK", 3)  # several blocks of columns
    ids, out, meta = fe._scale_file(path, "id", strategy)
    got = pd.concat([ids, pd.DataFrame(out)], axis=1)
    assert list(got.columns) == list(expected.columns)
    assert np.array_equal(got.drop(columns="id").to_numpy(), expected.drop(columns="id").to_numpy())
    assert meta["dropped_constant"] == ["constant", "all_null"] and meta["imputed"]["with_nulls"] > 0


def test_numeric_matrix_from_file_equals_the_dataframe_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(2)
    n = 500
    table = pd.DataFrame(
        {
            "id": [f"e{i}" for i in range(n)],
            "a": rng.normal(size=n),
            "label": rng.choice(["x", "y"], n),
            "count": rng.integers(0, 9, n),
            "flag": rng.random(n) < 0.5,
            "sparse": pd.array(np.where(rng.random(n) < 0.2, None, 3), dtype="Int64"),
            "b": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)),
        }
    )
    path = tmp_path / "t.parquet"
    table.to_parquet(path, index=False)
    monkeypatch.setattr(findings, "_COLUMN_CHUNK", 2)
    expected = findings.numeric_matrix(pd.read_parquet(path), "id")
    got = findings.read_numeric_matrix(path, "id")
    pd.testing.assert_frame_equal(got, expected)
    pd.testing.assert_frame_equal(findings.read_numeric_matrix(path, "id", columns=["b", "a"]), expected[["a", "b"]])
    some = {"e3", "e40"}
    pd.testing.assert_frame_equal(
        findings.read_numeric_matrix(path, "id", members=some), expected[expected.index.isin(some)]
    )


def test_event_copies_keep_only_the_columns_a_tool_reads(run_copy: RunContext) -> None:
    from jellybean.skills.feature_engineer.models import _open_events

    spec = read_spec()
    con, schema = _open_events(spec.enriched_artifact, "msisdn", ["duration_sec"], expressions="cell_id IS NOT NULL")
    try:
        copied = [r[0] for r in con.execute("DESCRIBE ev").fetchall()]
    finally:
        con.close()
    assert set(copied) == {"msisdn", "duration_sec", "cell_id"} == set(schema)


# -------------------------------------------------------------------- critic
def test_stability_reuses_the_redetection_of_a_subsample(run_copy: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    from jellybean.skills.critico import tools as cr

    store = ArtifactStore()
    entity_findings = [
        f for f in all_findings() if any(store.get(r).params.get("kind") == "outlier" for r in f.source_runs)
    ]
    same_runs = [f for f in entity_findings if f.source_runs == entity_findings[0].source_runs]
    f = entity_findings[0]
    recorded = load_finding(f.id).checks["stability"]
    calls: list[str] = []
    real = cr.outlier_scores
    monkeypatch.setattr(
        cr, "outlier_scores", lambda X, method, *a, **k: calls.append(method) or real(X, method, *a, **k)
    )
    cr.clear_redetect_cache()
    first = cr.stability_check(f.id, n_runs=3)
    fitted = len(calls)
    assert fitted > 0
    # Same result as the run that produced the fixture (no cache then).
    assert first.summary["stability"] == recorded["stability"]
    again = cr.stability_check(f.id, n_runs=3)
    assert len(calls) == fitted and again.summary["stability"] == first.summary["stability"]
    if len(same_runs) > 1:  # another finding of the same runs reuses the same subsamples
        cr.stability_check(same_runs[1].id, n_runs=3)
        assert len(calls) == fitted


def test_artifact_check_scans_the_dataset_once_per_run(run_copy: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    from jellybean.skills._shared import catalog
    from jellybean.skills.critico import tools as cr

    f = _entity_finding()
    statements: list[str] = []
    real_connect = catalog.connect

    class Recording:
        def __init__(self, con: Any) -> None:
            self._con = con

        def execute(self, query: str, *args: Any, **kwargs: Any) -> Any:
            statements.append(query)
            return self._con.execute(query, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._con, name)

    monkeypatch.setattr(cr, "connect", lambda *a, **k: Recording(real_connect(*a, **k)))
    cr.clear_source_facts()
    first = cr.artifact_check(f.id)
    assert first.status == "ok"
    on_source = [q for q in statements if "FROM src" in q and not q.startswith(("CREATE", "DESCRIBE"))]
    assert len(on_source) <= 6  # was one or two statements per column
    statements.clear()
    second = cr.artifact_check(f.id)
    assert second.summary == first.summary
    # The population facts are reused: nothing more is read from the whole dataset.
    assert not [q for q in statements if "FROM src" in q and not q.startswith(("CREATE", "DESCRIBE"))]


# ------------------------------------------------------------------- analyst
def test_surrogate_rules_on_a_sample_of_the_population(run_copy: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    from jellybean.skills.analista import tools as an

    f = next(f for f in all_findings() if f.type != "temporal" and f.critic_verdict == "confirmed")
    full = an.fit_surrogate_rules(f.id)
    assert full.status == "ok" and not full.warnings
    monkeypatch.setattr(run_copy.settings.config.detection, "surrogate_max_rows", 1000)
    sampled = an.fit_surrogate_rules(f.id)
    assert sampled.status == "ok" and any("sample of 1000" in w for w in sampled.warnings)
    assert sampled.summary["rules"] and sampled.summary["rules"][0]["coverage"] >= 0.5
    # Precision is scaled back to the whole population: it stays a share.
    assert all(0 < r["precision"] <= 1 for r in sampled.summary["rules"])


def test_exemplars_and_description_read_only_the_group(run_copy: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    from jellybean.skills.analista import tools as an

    f = next(f for f in all_findings() if f.type != "temporal" and f.critic_verdict == "confirmed")
    before = load_finding(f.id)
    monkeypatch.setattr(
        findings, "feature_tables", lambda *a, **k: pytest.fail("the whole feature tables must not be loaded")
    )
    assert an.get_exemplars(f.id, 3).status == "ok"
    assert an.describe_group(f.id).status == "ok"
    after = load_finding(f.id)
    # Ids are masked with the run's salt, so only the values are compared with the original run.
    values = [[{k: v for k, v in e.items() if k != f.entity_column} for e in x.exemplars] for x in (after, before)]
    assert values[0] == values[1] and after.top_features == before.top_features


def test_consolidation_keeps_the_largest_groups_of_a_run(
    run_copy: RunContext, monkeypatch: pytest.MonkeyPatch, phase5: dict
) -> None:
    from jellybean.skills.detector import tools as det

    monkeypatch.setattr(run_copy.settings.config.detection, "max_candidates_per_run", 1)
    result = det.consolidate_findings(phase5["runs"])
    assert result.status == "ok"
    assert any("only the 1 largest are used" in w for w in result.warnings)


def test_planted_anomalies_are_judged_against_a_sample(run_copy: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    from jellybean.skills.critico import tools as cr

    features = read_spec().features.scaled
    full = cr.inject_synthetic_anomalies(features, n=5)
    assert full.status == "ok" and not full.warnings
    # 5 planted rows stay well under the share each method flags (1%) in a sample of 2,000.
    monkeypatch.setattr(run_copy.settings.config.detection, "fit_max_rows", 2000)
    result = cr.inject_synthetic_anomalies(features, n=5)
    assert result.status == "ok" and any("sample of 2000" in w for w in result.warnings)
    assert result.summary["detected_by_2_plus"] >= full.summary["detected_by_2_plus"] - 0.2


def test_consolidation_ignores_runs_that_do_not_exist(run_copy: RunContext, phase5: dict) -> None:
    from jellybean.skills.detector import tools as det

    result = det.consolidate_findings([*phase5["runs"], "detection_hdbscan_v9", "features_scaled_v1"])
    assert result.status == "ok"
    assert any("detection_hdbscan_v9" in w and "features_scaled_v1" in w for w in result.warnings)
    missing = det.consolidate_findings(["detection_hdbscan_v9"])
    assert missing.status == "error" and "kmeans" in missing.summary["error"]
