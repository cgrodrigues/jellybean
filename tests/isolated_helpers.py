# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Functions run in worker processes by tests/test_isolated_work.py."""

from __future__ import annotations

import numpy as np


def hold_gil(X: np.ndarray, n: int) -> np.ndarray:
    """One long C call: the interpreter lock is never handed over while it runs."""
    return np.array([sum(range(n))])


def double(X: np.ndarray, factor: float) -> dict[str, np.ndarray]:
    return {"result": X * factor, "rows": np.array(len(X))}


def fail(X: np.ndarray) -> np.ndarray:
    raise ValueError("no good")
