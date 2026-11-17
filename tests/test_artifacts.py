# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the artifact store and manifest."""

from __future__ import annotations

import json
import os
import threading

import duckdb
import pytest

from jellybean.runtime.context import RunContext
from jellybean.skills._shared.artifacts import ArtifactNotFoundError, ArtifactStore, slugify


def test_versions_increment_and_manifest_fields(run_ctx: RunContext) -> None:
    store = ArtifactStore()
    first = store.save_json("profile_events", {"a": 1}, parents=["events"], params={"x": 1})
    second = store.save_json("profile_events", {"a": 2}, parents=[first.id])
    assert (first.id, second.id) == ("profile_events_v1", "profile_events_v2")

    manifest = json.loads(run_ctx.manifest_path.read_text())
    entry = manifest["artifacts"][0]
    for key in ("id", "type", "path", "created_by", "parents", "params", "row_count", "created_at"):
        assert key in entry
    assert entry["created_by"]["agent"] == "system"


def test_artifacts_are_read_only(run_ctx: RunContext) -> None:
    record = ArtifactStore().save_json("doc", {"a": 1})
    path = run_ctx.run_dir / record.path
    assert not os.access(path, os.W_OK)


def test_lineage_walks_all_ancestors(run_ctx: RunContext) -> None:
    store = ArtifactStore()
    a = store.save_json("a", {})
    b = store.save_json("b", {}, parents=[a.id])
    c = store.save_json("c", {}, parents=[b.id, "events"])
    assert [r.id for r in store.lineage(c.id)] == [b.id, a.id]
    with pytest.raises(ArtifactNotFoundError):
        store.lineage("nope_v1")


def test_table_artifact_from_sql_and_run_db_view(run_ctx: RunContext) -> None:
    store = ArtifactStore()
    con = duckdb.connect()
    record = store.save_table_from_sql(con, "SELECT range AS x FROM range(42)", "numbers")
    con.close()
    assert record.type == "table" and record.row_count == 42
    db = duckdb.connect(str(run_ctx.db_path), read_only=True)
    assert db.execute("SELECT count(*) FROM numbers_v1").fetchall()[0][0] == 42
    db.close()


def test_concurrent_creation_gives_unique_ids(run_ctx: RunContext) -> None:
    store = ArtifactStore(run_ctx)
    ids: list[str] = []

    def worker() -> None:
        ids.append(store.save_json("parallel", {"t": 1}).id)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(ids)) == 12
    assert len(store.list()) == 12


def test_invalid_prefix_rejected(run_ctx: RunContext) -> None:
    with pytest.raises(ValueError):
        ArtifactStore().save_json("Bad Prefix!", {})
    assert slugify("Bad Prefix!") == "bad_prefix"
