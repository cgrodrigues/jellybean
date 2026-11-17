# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the profiler's tool modules (``tools.py`` and ``integrity.py``)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import duckdb

from jellybean.runtime.context import current_run, maybe_current_run
from jellybean.runtime.tool_registry import current_tool
from jellybean.settings import get_settings
from jellybean.skills._shared.artifacts import ArtifactStore
from jellybean.skills._shared.catalog import Source, connect, resolve_source
from jellybean.skills._shared.masking import Masker


def _settings():
    run = maybe_current_run()
    return run.settings if run else get_settings()


def _masker() -> Masker:
    run = maybe_current_run()
    return Masker(_settings().config.masking, salt=run.run_id if run else "")


@contextmanager
def _open(name: str) -> Iterator[tuple[duckdb.DuckDBPyConnection, Source]]:
    """Resolve a dataset/artifact name and open a sandboxed connection."""
    source = resolve_source(name)
    con = connect()
    try:
        yield con, source
    finally:
        con.close()


def _issue(dataset: str, column: str | None, kind: str, severity: str, **detail: Any) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "column": column,
        "kind": kind,
        "severity": severity,
        "detail": detail,
        "source_tool": current_tool(),
    }


def _severity(share: float) -> str:
    return "high" if share >= 0.2 else "medium" if share >= 0.01 else "low"


def _store() -> ArtifactStore:
    return ArtifactStore(current_run())
