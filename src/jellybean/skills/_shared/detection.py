"""Detection algorithms shared by the detector (first run) and the critic (re-runs).

Every function is deterministic for a given seed and works on the per-entity
matrices produced by the feature engineer. Scores are oriented so that
**higher means more anomalous**.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jellybean.runtime.cancellation import check_cancelled

OUTLIER_METHODS = ("isolation_forest", "lof", "ecod", "hbos", "autoencoder")
CLUSTER_METHODS = ("hdbscan", "gmm", "kmeans")


class DetectionError(ValueError):
    """Invalid method/parameters or missing optional dependency."""


# ECOD and HBOS score each column on its own and add the columns up: above this many
# cells they are run a few columns at a time (same scores, a fraction of the memory).
_COLUMNWISE_CELLS = 20_000_000
# LOF compares every row with the fitted sample: the sample shrinks so that
# rows x sample stays below this many pairs (never below _LOF_MIN_SAMPLE rows).
_LOF_MAX_PAIRS = 25_000_000_000
_LOF_MIN_SAMPLE = 5_000
_SCORE_BATCH = 100_000


def _by_columns(fit_scores: Any, X: np.ndarray) -> np.ndarray:
    """Sum of ``fit_scores`` over blocks of columns (for detectors that add per-column scores)."""
    if X.size <= _COLUMNWISE_CELLS or X.shape[1] < 2:
        return fit_scores(X)
    width = max(1, _COLUMNWISE_CELLS // len(X))
    total = np.zeros(len(X))
    for start in range(0, X.shape[1], width):
        check_cancelled()
        total += fit_scores(np.ascontiguousarray(X[:, start : start + width]))
    return total


def outlier_scores(
    X: np.ndarray,
    method: str,
    params: dict[str, Any],
    seed: int,
    lof_max_rows: int,
    fit_max_rows: int | None = None,
) -> np.ndarray:
    """Outlier score per row (higher = more anomalous).

    Memory stays bounded on large populations: ECOD and HBOS run a few columns
    at a time (same scores), LOF fits on a sample, and above ``fit_max_rows``
    the autoencoder is trained on a seeded sample and scores the rest in batches.
    """
    params = dict(params)
    if method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(n_estimators=int(params.get("n_estimators", 200)), random_state=seed, n_jobs=-1)
        model.fit(X)
        return -model.score_samples(X)
    if method == "lof":
        from sklearn.neighbors import LocalOutlierFactor

        k = int(params.get("n_neighbors", 35))
        if len(X) > lof_max_rows:
            # Quadratic method: fit on a sample, score everything (novelty mode).
            rng = np.random.default_rng(seed)
            size = min(lof_max_rows, max(_LOF_MIN_SAMPLE, _LOF_MAX_PAIRS // len(X)))
            sample = X[rng.choice(len(X), size, replace=False)]
            model = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(sample)
            return -model.score_samples(X)
        model = LocalOutlierFactor(n_neighbors=k)
        model.fit(X)
        return -model.negative_outlier_factor_
    if method == "ecod":
        import warnings

        from pyod.models.ecod import ECOD

        with warnings.catch_warnings():
            # Near-constant columns make scipy's skewness warn about precision;
            # pyod already replaces the resulting NaNs, so the warning is noise.
            warnings.simplefilter("ignore", RuntimeWarning)
            return _by_columns(lambda block: ECOD().fit(block).decision_scores_, X)
    if method == "hbos":
        from pyod.models.hbos import HBOS

        n_bins = int(params.get("n_bins", 20))
        return _by_columns(lambda block: HBOS(n_bins=n_bins).fit(block).decision_scores_, X)
    if method == "autoencoder":
        try:
            from pyod.models.auto_encoder import AutoEncoder
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise DetectionError("autoencoder needs PyTorch: run `uv sync --extra deep`") from exc
        try:
            model = AutoEncoder(epoch_num=int(params.get("epochs", 20)), random_state=seed, verbose=0)
        except TypeError:  # pragma: no cover - older/newer pyod signatures
            model = AutoEncoder(random_state=seed)
        if fit_max_rows is None or len(X) <= fit_max_rows:
            return model.fit(X).decision_scores_
        chosen = np.sort(np.random.default_rng(seed).choice(len(X), fit_max_rows, replace=False))
        model.fit(X[chosen])
        scores = np.empty(len(X))
        for start in range(0, len(X), _SCORE_BATCH):
            check_cancelled()
            scores[start : start + _SCORE_BATCH] = model.decision_function(X[start : start + _SCORE_BATCH])
        return scores
    raise DetectionError(f"Unknown outlier method '{method}'. Use one of {OUTLIER_METHODS}.")


def flag_top(scores: np.ndarray, contamination: float) -> np.ndarray:
    """Boolean mask of the top ``contamination`` share of scores.

    Ties at the minimum score are never flagged: when (almost) every score is
    equal there is nothing anomalous to flag. If the threshold lands in a large
    block of tied scores (more than twice the expected count), only scores
    strictly above it are flagged.
    """
    scores = np.asarray(scores, dtype=float)
    n = max(1, round(len(scores) * contamination))
    threshold = np.partition(scores, -n)[-n]
    flagged = (scores >= threshold) & (scores > scores.min())
    if flagged.sum() > 2 * n:
        # The threshold falls inside a large block of tied scores (discrete
        # features): flag only what is strictly above the tie.
        flagged = scores > threshold
    return flagged


def low_support(n_events: pd.Series, quantile: float = 0.10) -> pd.Series:
    """Entities with too few events for reliable features.

    Low support means **both** below half the median activity **and** in the
    bottom ``quantile``. A relative rule: when every entity has a similar number
    of events (e.g. periodic readings), no entity is low-support.
    """
    events = n_events.astype(float)
    return (events < 0.5 * float(events.median())) & (events <= float(events.quantile(quantile)))


def cluster_labels(
    X: np.ndarray, method: str, params: dict[str, Any], seed: int, max_rows: int | None = None
) -> np.ndarray:
    """Cluster label per row (-1 = noise, for HDBSCAN).

    HDBSCAN compares entities pairwise, so above ``max_rows`` it is fitted on a
    seeded sample of that size; the rows left out get the label -1 (not clustered).
    """
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=int(params.get("min_cluster_size", 10)), min_samples=params.get("min_samples"), copy=True
        )
        if max_rows is not None and len(X) > max_rows:
            chosen = np.sort(np.random.default_rng(seed).choice(len(X), size=max_rows, replace=False))
            labels = np.full(len(X), -1, dtype=int)
            labels[chosen] = model.fit_predict(X[chosen])
            return labels
        return model.fit_predict(X)
    if method == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=int(params.get("n_clusters", 12)), n_init=10, random_state=seed).fit_predict(X)
    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=int(params.get("n_components", 12)),
            random_state=seed,
            covariance_type=params.get("covariance_type", "diag"),
        )
        return model.fit(X).predict(X)
    raise DetectionError(f"Unknown clustering method '{method}'. Use one of {CLUSTER_METHODS}.")


def _varying_numeric(raw: pd.DataFrame, entity: str) -> list[str]:
    """Numeric feature columns with more than one value."""
    columns = raw.drop(columns=[entity]).select_dtypes(include=["number"]).columns
    return [c for c in columns if raw[c].nunique() > 1]


def component_table(edges: pd.DataFrame, entity: str, tie: str = "value") -> pd.DataFrame:
    """(entity, group, group_size) of the connected components of an edge list (columns ``src``, ``dst``).

    Uses a sparse matrix of integer codes, so millions of edges are fine. Groups
    are numbered from 1 by decreasing size; ties are broken by the smallest
    member (``tie="value"``) or by the smallest member as text (``tie="text"``).
    Rows are ordered by group, then by entity.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = edges.dropna(subset=["src", "dst"])
    codes, nodes = pd.factorize(pd.concat([edges["src"], edges["dst"]], ignore_index=True))
    if not len(nodes):
        return pd.DataFrame({entity: [], "group": pd.Series(dtype=int), "group_size": pd.Series(dtype=int)})
    src, dst = codes[: len(edges)], codes[len(edges) :]
    adjacency = coo_matrix((np.ones(len(edges), dtype=np.int8), (src, dst)), shape=(len(nodes), len(nodes)))
    _, labels = connected_components(adjacency, directed=False)
    table = pd.DataFrame({entity: np.asarray(nodes), "label": labels})
    sizes = table.groupby("label").size()
    first = (table[entity].astype(str) if tie == "text" else table[entity]).groupby(table["label"]).min()
    order = sorted(sizes.index, key=lambda lab: (-sizes.at[lab], str(first.at[lab])))
    rank = pd.Series(np.arange(1, len(order) + 1), index=order)
    table["group"] = table["label"].map(rank).astype(int)
    table["group_size"] = table["label"].map(sizes).astype(int)
    return table.sort_values(["group", entity]).drop(columns="label").reset_index(drop=True)


def context_defining_features(raw: pd.DataFrame, entity: str, context: str, n_features: int) -> list[str]:
    """Numeric features that differ most **between** context groups (eta squared on ranks).

    These are the features that characterise the contexts (e.g. the behaviour
    that tells one segment from another). They are chosen statistically, not by
    name, so no domain knowledge is assumed, and near-duplicates (|rank
    correlation| > 0.9 within the context groups, with a feature already chosen)
    are skipped.
    """
    numeric = _varying_numeric(raw, entity)
    groups = raw[context].astype("string").fillna("__missing__")
    # One column at a time: the ranks of every feature at once would be several
    # copies of the whole table.
    eta = {}
    for col in numeric:
        ranks = raw[col].astype(float).rank(pct=True)
        mean = ranks.mean()
        within_total = ((ranks - mean) ** 2).sum()
        if within_total == 0:
            continue
        eta[col] = ((ranks.groupby(groups).transform("mean") - mean) ** 2).sum() / within_total
    ranked = pd.Series(eta, dtype=float).dropna().sort_values(ascending=False).index.tolist()
    # Prefer diverse features: skip one that is almost a copy (|rank corr| > 0.9)
    # of a feature already chosen (e.g. mean / median / p95 of the same measure).
    # Redundancy is measured WITHIN context groups: features that only move
    # together because they differ between groups (e.g. consumption and rated
    # power across models) are not copies inside a group, where outliers live.
    top = ranked[: 4 * n_features]
    within = raw[top].astype(float).groupby(groups).rank(pct=True)
    corr = within.corr().abs()
    chosen: list[str] = []
    for feature in ranked:
        if len(chosen) == n_features:
            break
        if feature in corr.index and any(corr.loc[feature, c] > 0.9 for c in chosen):
            continue
        chosen.append(feature)
    return chosen


def contextual_scores(
    raw: pd.DataFrame,
    entity: str,
    context: str,
    method: str = "context_defining",
    n_features: int = 8,
    top_k: int = 3,
) -> tuple[pd.Series, pd.DataFrame]:
    """Score of each entity relative to its context group (e.g. its segment).

    Features are log1p-transformed when non-negative, then robustly standardised
    **within each context value** (median / MAD).

    * ``context_defining`` (default): root mean square of the z-scores on the
      features that best separate the contexts. Finds entities that behave like
      another context (e.g. a basic plan with a business profile).
    * ``all_features``: mean of the ``top_k`` largest |z| over all features.

    Returns:
        The scores and the per-entity z-scores used (columns = features).
    """
    if method == "context_defining":
        columns = context_defining_features(raw, entity, context, n_features)
    elif method == "all_features":
        columns = _varying_numeric(raw, entity)
    else:
        raise DetectionError(f"Unknown contextual method '{method}'. Use context_defining or all_features.")
    transformed = raw[columns].astype(float).apply(lambda c: np.log1p(c) if c.min() >= 0 else c)
    groups = raw[context].astype("string").fillna("__missing__")
    z = pd.DataFrame(index=raw.index, columns=transformed.columns, dtype=float)
    for _, idx in groups.groupby(groups).groups.items():
        block = transformed.loc[idx]
        med = block.median()
        mad = (block - med).abs().median() * 1.4826
        mad = mad.where(mad > 0, block.std()).replace(0, np.nan)
        z.loc[idx] = ((block - med) / mad).to_numpy()
    z = z.fillna(0.0).clip(-30, 30)
    if method == "context_defining":
        scores = np.sqrt((z**2).mean(axis=1))
    else:
        scores = pd.Series(np.sort(z.abs().to_numpy(), axis=1)[:, -top_k:].mean(axis=1), index=raw.index)
    return scores, z


def flag_within_groups(scores: pd.Series, groups: pd.Series, contamination: float) -> pd.Series:
    """Flag the top ``contamination`` share of scores inside each group."""
    flags = pd.Series(False, index=scores.index)
    for _, idx in groups.groupby(groups).groups.items():
        flags.loc[idx] = flag_top(scores.loc[idx].to_numpy(), contamination)
    return flags


def _baseline(df: pd.DataFrame, keep_days: set | None, rolling: int | None) -> pd.Series:
    """Usual value of each point of a (group, ts, value) table.

    Seasonal (default): median of the same group at the same hour of day and
    weekday type. Rolling (``rolling`` buckets, e.g. 7 days): centred moving
    median of the group, which follows trends and level shifts (a series that
    decays, a group whose share changes). ``keep_days`` limits the values used.
    """
    kept = df["ts"].dt.normalize().isin(keep_days) if keep_days is not None else pd.Series(True, index=df.index)
    if rolling:
        values = df["value"].where(kept)
        order = df.sort_values(["group", "ts"]).index
        usual = (
            values.loc[order]
            .groupby(df.loc[order, "group"])
            .transform(lambda v: v.rolling(rolling, center=True, min_periods=1).median())
        )
        return usual.reindex(df.index).fillna(df.groupby("group")["value"].transform("median"))
    hour, weekend = df["ts"].dt.hour, df["ts"].dt.day_of_week >= 5
    keys = pd.DataFrame({"group": df["group"], "hour": hour, "weekend": weekend})
    usual = df["value"][kept].groupby([keys["group"][kept], keys["hour"][kept], keys["weekend"][kept]]).median()
    mapped = pd.MultiIndex.from_frame(keys).map(usual.to_dict().get) if len(usual) else None
    out = pd.Series(mapped, index=df.index, dtype=float) if mapped is not None else pd.Series(np.nan, index=df.index)
    return out.fillna(df.groupby("group")["value"].transform("median"))


def common_level(series: pd.DataFrame, keep_days: set | None = None, rolling: int | None = None) -> pd.Series:
    """Level shared by all groups at each time, relative to their own seasonal baselines.

    For every active group, the ratio of its value to its usual value for that
    hour and weekday type; the level is the median ratio over the groups at that
    time. 1.0 is a usual time; 1.8 means the typical group has 80% more than usual
    (e.g. a month-end peak). A median is robust to one anomalous group and to
    groups starting or ending. Clipped to [0.2, 5].
    """
    df = series[["group", "ts", "value"]].reset_index(drop=True)
    df["usual"] = _baseline(df, keep_days, rolling)
    df = df[df["usual"] > 0]
    if df.empty:
        return pd.Series(dtype=float)
    ratio = (df["value"] / df["usual"]).groupby(df["ts"]).median()
    return ratio.clip(0.2, 5.0)


def event_unit(series: pd.DataFrame) -> pd.Series:
    """Typical value of one event per group, for series of sums.

    With the number of events per bucket (column ``n``): the median of sum / n.
    Without it: the 10th percentile of the non-zero buckets (about one event in
    a sparse series).
    """
    nonzero = series[series["value"] > 0]
    if "n" in nonzero.columns:
        per_event = (nonzero["value"] / nonzero["n"].where(nonzero["n"] > 0)).abs()
        return per_event.groupby(nonzero["group"]).median().clip(lower=1e-9)
    return nonzero.groupby("group")["value"].quantile(0.1).clip(lower=1e-9)


def seasonal_residual_anomalies(
    series: pd.DataFrame,
    threshold: float,
    keep_days: set | None = None,
    level: pd.Series | None = None,
    unit: pd.Series | None = None,
    rolling: int | None = None,
) -> pd.DataFrame:
    """Robust anomalies of a (group, ts, value) table against an hour-of-day baseline.

    The expected value of each point is the median of the same group at the same
    hour of day and weekday type (weekday/weekend) - the seasonal component -,
    multiplied by ``level`` at that time when given (see :func:`common_level`:
    a change shared by every group is not an anomaly of one group). The
    residual is standardised with the group's MAD, with a floor for sparse
    series: sqrt(expected) for counts (Poisson); for sums of events worth about
    ``unit`` each (see :func:`event_unit`), sqrt(expected * unit) in sparse
    groups (median of at most 10 events per bucket, column ``n``) and
    max(sqrt(expected), unit) in dense ones. ``keep_days``
    restricts the baseline to some days (used by the critic's stability check).
    """
    df = series.reset_index(drop=True)
    df["expected"] = _baseline(df, keep_days, rolling)
    if level is not None:
        df["expected"] = df["expected"] * df["ts"].map(level).fillna(1.0).to_numpy()
    df["resid"] = df["value"] - df["expected"]
    mad = df.groupby("group")["resid"].transform(lambda r: 1.4826 * (r - r.median()).abs().median())
    expected = df["expected"].to_numpy()
    if unit is None:
        floor = np.sqrt(np.maximum(expected, 1.0))
    else:
        one = df["group"].map(unit).fillna(1.0).to_numpy()
        # Sparse groups (few events per bucket): a sum of independent events, sd ~ sqrt(expected * one).
        # Dense groups: the usual floor, never below one event's value.
        sparse = (
            df.groupby("group")["n"].transform("median").to_numpy() <= 10
            if "n" in df.columns
            else np.ones(len(df), bool)
        )
        floor = np.where(
            sparse, np.sqrt(np.maximum(expected, one) * one), np.maximum(np.sqrt(np.maximum(expected, 1.0)), one)
        )
    scale = np.maximum(mad, floor)
    df["z"] = df["resid"] / scale
    return df[df["z"] >= threshold].drop(columns=["resid"])


def _life_grid(active: pd.DataFrame, step: pd.Timedelta) -> pd.DataFrame:
    """(group, ts) for every bucket between each group's first (``min``) and last (``max``) active bucket."""
    counts = ((active["max"] - active["min"]) // step + 1).to_numpy().astype(np.int64)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    offsets = np.arange(int(counts.sum()), dtype=np.int64) - first
    starts = pd.DatetimeIndex(active["min"]).repeat(counts)
    ts = (starts + pd.to_timedelta(offsets * step.value)).astype(active["min"].dtype)
    return pd.DataFrame({"group": active.index.repeat(counts), "ts": ts})


def grouped_series_anomalies(
    series: pd.DataFrame,
    step: pd.Timedelta,
    threshold: float,
    keep_days: set | None = None,
    keep_groups: set | None = None,
    sums: bool = False,
) -> pd.DataFrame:
    """Anomalies of one series per group (e.g. per promotion, per cell, per batch).

    Two complementary tests, both robust:

    * **Within the group's life.** Each group is compared with itself only
      between its first and last active bucket (a campaign that exists for a
      month is not anomalous for existing), against its seasonal baseline scaled
      by the common level of all groups (a month-end peak shared by everyone is
      not an anomaly of one group).
    * **Bursts.** A group whose peak bucket holds at least half of its whole
      volume, with a peak far above the other groups' peaks (robust z of the
      log peak >= threshold / 2), lived as a burst (e.g. a whole batch used in
      hours), which no comparison with itself can show.

    ``keep_days`` (baseline days) and ``keep_groups`` (groups for the burst
    distribution) subsample both tests for the critic's stability check.
    ``sums`` marks a series of sums of a measure (not counts).

    Args:
        series: (group, ts, value) rows of the active buckets (missing = 0).
    """
    columns = ["group", "ts", "value", "expected", "z", "test"]
    if series.empty:
        return pd.DataFrame(columns=columns)
    active = series[series["value"] > 0].groupby("group")["ts"].agg(["min", "max"])
    grid = _life_grid(active, step).merge(series, on=["group", "ts"], how="left")
    grid["value"] = grid["value"].fillna(0.0).astype(float)
    if "n" in grid.columns:
        grid["n"] = grid["n"].fillna(0)
    # Daily (or coarser) buckets: a centred 7-bucket moving median follows trends
    # and level shifts; hourly buckets keep the hour-of-day seasonal baseline.
    rolling = 7 if step >= pd.Timedelta(days=1) else None
    level = common_level(grid, keep_days, rolling) if grid["group"].nunique() > 1 else None
    buckets = grid.groupby("group")["ts"].transform("size")
    unit = event_unit(series) if sums else None
    life = seasonal_residual_anomalies(
        grid[buckets >= 3], threshold, keep_days=keep_days, level=level, unit=unit, rolling=rolling
    )
    life = life.assign(test="lifetime")[columns]

    stats = grid.groupby("group")["value"].agg(["max", "sum"])
    pool = stats if keep_groups is None else stats[stats.index.isin(keep_groups)]
    logs = np.log1p(pool["max"])
    center = float(logs.median())
    mad = 1.4826 * float((logs - center).abs().median()) or 1e-9
    z = (np.log1p(stats["max"]) - center) / mad
    burst = stats[(stats["max"] >= 0.5 * stats["sum"]) & (z >= threshold / 2) & (stats["max"] >= 10)]
    rows = []
    for group, row in burst.iterrows():
        peak = grid[(grid["group"] == group)].sort_values("value").iloc[-1]
        # z doubled: the burst test uses threshold / 2, so its z is reported on the threshold's scale.
        rows.append([group, peak["ts"], float(row["max"]), float(np.expm1(center)), 2 * float(z[group]), "burst"])
    out = pd.concat([life, pd.DataFrame(rows, columns=columns)], ignore_index=True) if rows else life
    out["ts"] = pd.to_datetime(out["ts"])
    # A burst point already flagged by the life test is kept once (the larger z).
    return out.sort_values("z", ascending=False).drop_duplicates(["group", "ts"]).reset_index(drop=True)


def merge_windows(points: pd.DataFrame, step: pd.Timedelta) -> list[dict[str, Any]]:
    """Merge consecutive anomalous timestamps of each group into windows."""
    windows = []
    for group, block in points.sort_values("ts").groupby("group", sort=False):
        start = prev = None
        zmax, vsum, esum = 0.0, 0.0, 0.0
        for row in block.itertuples():
            if start is None or row.ts - prev > step:
                if start is not None:
                    windows.append(
                        {
                            "group": group,
                            "start": start,
                            "end": prev + step,
                            "max_z": zmax,
                            "value": vsum,
                            "expected": esum,
                        }
                    )
                start, zmax, vsum, esum = row.ts, 0.0, 0.0, 0.0
            prev = row.ts
            zmax, vsum, esum = max(zmax, row.z), vsum + row.value, esum + row.expected
        if start is not None:
            windows.append(
                {"group": group, "start": start, "end": prev + step, "max_z": zmax, "value": vsum, "expected": esum}
            )
    return windows
