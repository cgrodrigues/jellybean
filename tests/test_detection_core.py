"""Detection helpers that must hold for any domain (found with an independent telemetry dataset)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jellybean.skills._shared.detection import context_defining_features, flag_top, low_support


def test_low_support_is_relative() -> None:
    uniform = pd.Series([336] * 100)  # periodic readings: every entity has the same activity
    assert not low_support(uniform).any()
    skewed = pd.Series([1, 2, 3] + [60] * 97)
    assert low_support(skewed).tolist()[:3] == [True, True, True] and not low_support(skewed)[3:].any()


def test_flag_top_ignores_ties() -> None:
    assert not flag_top(np.zeros(100), 0.05).any()  # all equal: nothing anomalous
    scores = np.array([0.0] * 80 + [1.0] * 19 + [5.0])  # threshold falls in a big tie
    flagged = flag_top(scores, 0.02)
    assert flagged.sum() == 1 and flagged[-1]
    regular = np.arange(100, dtype=float)
    assert flag_top(regular, 0.05).sum() == 5


def test_context_features_are_diverse_within_groups() -> None:
    rng = np.random.default_rng(0)
    group = np.repeat(["A", "B", "C"], 200)
    rated = np.repeat([15.0, 30.0, 55.0], 200) * rng.normal(1, 0.02, 600)
    load = rng.uniform(0.3, 1.0, 600)
    temp = np.repeat([55.0, 60.0, 65.0], 200) + rng.normal(0, 1, 600)
    raw = pd.DataFrame(
        {
            "id": range(600),
            "grp": group,
            "attr_rated": rated,
            "power_mean": rated * load,
            "temp_mean": temp,
            "temp_p95": temp + 0.01 * rng.normal(size=600),
        }
    )
    chosen = context_defining_features(raw, "id", "grp", 3)
    # temp_p95 is a copy of temp_mean inside every group; power is not a copy of rated inside a group.
    assert "power_mean" in chosen and not {"temp_mean", "temp_p95"} <= set(chosen)


# ------------------------------------------------------------ grouped time series
def _daily(groups: dict[str, list[float]], start: str = "2026-10-01") -> pd.DataFrame:
    import pandas as pd

    rows = []
    for group, values in groups.items():
        for i, v in enumerate(values):
            if v:
                rows.append({"group": group, "ts": pd.Timestamp(start) + pd.Timedelta(days=i), "value": float(v)})
    return pd.DataFrame(rows)


def test_a_group_is_judged_within_its_own_life() -> None:
    import numpy as np
    import pandas as pd

    from jellybean.skills._shared.detection import grouped_series_anomalies

    rng = np.random.default_rng(1)
    groups = {f"g{i}": list(rng.poisson(20, 60)) for i in range(8)}
    groups["campaign"] = [0] * 30 + list(rng.poisson(20, 30))  # exists only for 30 days
    groups["campaign"][45] = 120  # one real spike inside its life
    points = grouped_series_anomalies(_daily(groups), pd.Timedelta(days=1), 6.0)
    flagged = set(
        zip(points["group"], points["ts"].dt.day_of_year - pd.Timestamp("2026-10-01").day_of_year, strict=True)
    )
    assert ("campaign", 45) in flagged
    assert not [d for g, d in flagged if g == "campaign" and d != 45]  # its start is not an anomaly


def test_a_peak_shared_by_every_group_is_not_an_anomaly() -> None:
    import numpy as np
    import pandas as pd

    from jellybean.skills._shared.detection import grouped_series_anomalies

    rng = np.random.default_rng(2)
    groups = {f"g{i}": [float(v) for v in rng.poisson(30, 60)] for i in range(10)}
    for g in groups:
        for d in (28, 29, 30):  # a month-end peak for everyone
            groups[g][d] *= 2.5
    groups["g0"][10] = 200  # an anomaly of one group
    points = grouped_series_anomalies(_daily(groups), pd.Timedelta(days=1), 6.0)
    days = set(points["ts"].dt.day_of_year - pd.Timestamp("2026-10-01").day_of_year)
    assert 10 in days and not days & {28, 29, 30}


def test_a_group_living_as_one_burst_is_flagged() -> None:
    import numpy as np
    import pandas as pd

    from jellybean.skills._shared.detection import grouped_series_anomalies

    rng = np.random.default_rng(3)
    groups = {f"g{i}": [float(v) for v in rng.poisson(3, 40)] for i in range(30)}
    groups["burst"] = [0.0] * 20 + [300.0] + [0.0] * 19  # its whole life in one day
    points = grouped_series_anomalies(_daily(groups), pd.Timedelta(days=1), 6.0)
    burst = points[points["group"] == "burst"]
    assert len(burst) == 1 and burst["test"].iloc[0] == "burst" and burst["z"].iloc[0] >= 6


def test_sums_are_scaled_by_the_value_of_one_event() -> None:
    import pandas as pd

    from jellybean.skills._shared.detection import seasonal_residual_anomalies

    # A sparse series of sums: one or two events of about 50 a day, sometimes none.
    values = [50, 0, 100, 50, 0, 50, 100, 0, 50, 50] * 4
    series = _daily({"g": values})
    counts = seasonal_residual_anomalies(series, 6.0)  # treated as counts: everything looks huge
    sums = seasonal_residual_anomalies(series, 6.0, unit=pd.Series({"g": 50.0}))
    assert len(sums) == 0 and len(counts) >= len(sums)
