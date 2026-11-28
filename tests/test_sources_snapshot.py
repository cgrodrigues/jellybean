# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Snapshots of an external source: what is asked of the source, what lands on disk, and how a run uses it.

A fake client stands in for ClickHouse (it answers the queries from small
tables); the SQL itself is checked against an embedded ClickHouse when ``chdb``
is installed (``uv run --with chdb pytest tests/test_sources_snapshot.py``).
"""

from __future__ import annotations

import io
import json
import re
import zlib
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from jellybean.settings import AppConfig, Settings, SourceConfig, reset_settings
from jellybean.sources import snapshot as snap
from jellybean.sources.clickhouse import connection_params
from jellybean.sources.snapshot import SnapshotError, make_request, take_snapshot

SOURCES = {
    "ch": {
        "database": "telco",
        "tables": {
            "events": {"time_column": "ts", "entity_column": "msisdn"},
            "subscribers": {"entity_column": "msisdn"},
            "plans": {},
        },
    }
}


def _tables() -> dict[str, pd.DataFrame]:
    n = 2000
    events = pd.DataFrame(
        {
            "event_id": range(n),
            "msisdn": [f"3519{i % 200:05d}" for i in range(n)],
            "ts": pd.Timestamp("2026-10-01") + pd.to_timedelta([i * 130 for i in range(n)], unit="s"),
            "amount": [float(i % 17) for i in range(n)],
        }
    )
    subscribers = pd.DataFrame({"msisdn": [f"3519{i:05d}" for i in range(200)], "plan": [i % 3 for i in range(200)]})
    plans = pd.DataFrame({"plan": [0, 1, 2], "name": ["a", "b", "c"]})
    return {"events": events, "subscribers": subscribers, "plans": plans}


KINDS = {"ts": "DateTime", "event_id": "UInt64", "amount": "Float64"}


class FakeClient:
    """Answers the snapshot's queries from data frames, and records them."""

    def __init__(self, tables: dict[str, pd.DataFrame], fail_on: str | None = None) -> None:
        self.tables, self.fail_on, self.queries, self.closed = tables, fail_on, [], False

    def columns(self, database: str, table: str) -> list[tuple[str, str]]:
        frame = self.tables.get(table)
        return [] if frame is None else [(c, KINDS.get(c, "String")) for c in frame.columns]

    def stream_parquet(self, sql: str):
        self.queries.append(sql)
        if self.fail_on and self.fail_on in sql:
            raise SnapshotError("ClickHouse query failed: connection lost")
        table = re.search(r"FROM `telco`\.`(\w+)`", sql).group(1)
        frame = self.tables[table]
        moments = re.findall(r"toDateTime\('([^']+)'", sql)
        if moments:
            frame = frame[(frame["ts"] >= moments[0]) & (frame["ts"] < moments[1])]
        sample = re.search(r"cityHash64\(toString\(`(\w+)`\)\) % 1000000 < (\d+)", sql)
        if sample:
            column, buckets = sample.group(1), int(sample.group(2))
            frame = frame[frame[column].map(lambda v: zlib.crc32(str(v).encode()) % 1_000_000 < buckets)]
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        yield buffer.getvalue()

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def source_settings(settings: Settings) -> Settings:
    return Settings(env=settings.env, config=AppConfig.model_validate({"sources": SOURCES}))


def _take(settings: Settings, client: FakeClient, start="2026-10-01", end="2026-10-04", sample=None, **kwargs):
    return take_snapshot(
        settings, "ch", make_request(start, end, sample), client_factory=lambda name, source: client, **kwargs
    )


# --------------------------------------------------------------------------- #
# Queries                                                                     #
# --------------------------------------------------------------------------- #
def test_the_source_does_the_filtering() -> None:
    cfg = SourceConfig.model_validate(
        SOURCES["ch"]
        | {
            "tables": {
                "events": {
                    "time_column": "ts",
                    "entity_column": "msisdn",
                    "columns": ["msisdn", "ts", "kind"],
                    "where": "country = 'PT'",
                }
            }
        }
    )
    columns = [("msisdn", "String"), ("ts", "DateTime('Europe/Lisbon')"), ("kind", "Enum8('a' = 1)"), ("x", "UInt8")]
    plan = snap.plan_table(
        "events", cfg.tables["events"], "telco", columns, make_request("2026-10-01", "2026-10-03", 0.1)
    )
    assert [p.file for p in plan.parts] == ["part-2026-10-01.parquet", "part-2026-10-02.parquet"]
    sql = plan.parts[1].sql
    assert "`x`" not in sql  # only the configured columns leave the source
    # Converted outside, filtered inside on the table's own columns (an alias would shift the window).
    outer, inner = sql.split(" FROM (", 1)
    assert "timezoneOffset" in outer and "timezoneOffset" not in inner and inner.startswith("SELECT `msisdn`, `ts`")
    assert "toString(`kind`) AS `kind`" in sql
    assert "addSeconds(toDateTime64(`ts`, 0), timezoneOffset(`ts`)) AS `ts`" in sql
    assert "(country = 'PT')" in sql
    assert "cityHash64(toString(`msisdn`)) % 1000000 < 100000" in sql
    # The window is one day, end excluded, in the time zone of the column.
    assert "`ts` >= toDateTime('2026-10-02 00:00:00', 'Europe/Lisbon')" in sql
    assert "`ts` < toDateTime('2026-10-03 00:00:00', 'Europe/Lisbon')" in sql


def test_a_window_inside_a_day_and_a_table_without_time() -> None:
    cfg = SourceConfig.model_validate(SOURCES["ch"])
    request = make_request("2026-10-01T06:00", "2026-10-02T06:00", None)
    plan = snap.plan_table("events", cfg.tables["events"], "telco", [("msisdn", "String"), ("ts", "DateTime")], request)
    assert [p.file for p in plan.parts] == ["part-2026-10-01T060000.parquet", "part-2026-10-02.parquet"]
    assert "toDateTime('2026-10-02 06:00:00')" in plan.parts[1].sql
    whole = snap.plan_table("plans", cfg.tables["plans"], "telco", [("plan", "UInt8")], request)
    assert [p.sql for p in whole.parts] == ["SELECT `plan` FROM `telco`.`plans`"]


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ([], "was not found"),
        ([("msisdn", "String")], "no column ts"),
        ([("msisdn", "String"), ("ts", "String")], "must be a Date or DateTime"),
    ],
)
def test_a_wrong_table_configuration_is_explained(columns: list, message: str) -> None:
    cfg = SourceConfig.model_validate(SOURCES["ch"])
    with pytest.raises(SnapshotError, match=message):
        snap.plan_table(
            "events", cfg.tables["events"], "telco", columns, make_request("2026-10-01", "2026-10-02", None)
        )


@pytest.mark.parametrize(
    ("start", "end", "sample", "message"),
    [
        ("2026-10-01", None, None, "both --from and --to"),
        ("2026-10-02", "2026-10-01", None, "before --to"),
        ("yesterday", "2026-10-01", None, "not a date"),
        ("2026-10-01", "2026-10-02", 0.0, "share of the entities"),
        ("2026-10-01", "2026-10-02", 1.5, "share of the entities"),
    ],
)
def test_invalid_requests_are_rejected(start, end, sample, message: str) -> None:
    with pytest.raises(SnapshotError, match=message):
        make_request(start, end, sample)


def test_names_in_the_configuration_cannot_carry_sql() -> None:
    for bad in ({"tables": {"ev; DROP": {}}}, {"tables": {"ev": {"time_column": "ts` OR 1"}}}, {"tables": {}}):
        with pytest.raises(ValueError):
            SourceConfig.model_validate(bad)


# --------------------------------------------------------------------------- #
# Snapshot on disk                                                            #
# --------------------------------------------------------------------------- #
def test_a_snapshot_is_a_folder_of_datasets(source_settings: Settings) -> None:
    client = FakeClient(_tables())
    messages: list[str] = []
    taken = _take(source_settings, client, progress=messages.append)
    assert not taken.reused and client.closed
    assert taken.path == source_settings.snapshots_root / "ch" / taken.manifest.id
    assert taken.manifest.id.startswith("20261001-20261104"[:9]) and not list(taken.path.parent.glob("*.partial"))
    tables = taken.manifest.tables
    events = _tables()["events"]
    assert tables["events"].rows == len(events[events["ts"] < "2026-10-04"])
    assert tables["events"].files == 3 and tables["events"].windowed and not tables["events"].sampled
    assert tables["subscribers"].rows == 200 and tables["plans"].rows == 3 and tables["plans"].files == 1
    assert any("events: part-2026-10-02.parquet" in m for m in messages)
    # The run reads it exactly like the data root.
    from jellybean.skills._shared.catalog import connect, discover_datasets

    reading = source_settings.with_data_root(taken.path)
    assert set(discover_datasets(reading.data_root)) == {"events", "subscribers", "plans"}
    con = connect(reading)
    try:
        assert con.execute("SELECT count(*), min(ts), max(ts) < TIMESTAMP '2026-10-04' FROM events").fetchone()[
            ::2
        ] == (
            tables["events"].rows,
            True,
        )
    finally:
        con.close()
    assert json.loads((taken.path / "snapshot.json").read_text())["source"] == "ch"


def test_the_same_request_reuses_the_snapshot(source_settings: Settings) -> None:
    first = _take(source_settings, FakeClient(_tables()))
    untouched = FakeClient(_tables())
    again = _take(source_settings, untouched)
    assert again.reused and again.path == first.path and untouched.queries == []
    other_window = _take(source_settings, FakeClient(_tables()), end="2026-10-03")
    assert other_window.path != first.path
    refreshed = FakeClient(_tables())
    assert not _take(source_settings, refreshed, refresh=True).reused and refreshed.queries
    assert len(snap.list_snapshots(source_settings.snapshots_root)) == 2


def test_the_sample_keeps_whole_entities_in_every_table(source_settings: Settings) -> None:
    taken = _take(source_settings, FakeClient(_tables()), sample=0.25)
    assert "_s25_" in taken.manifest.id
    events = pd.read_parquet(taken.path / "events")
    subscribers = pd.read_parquet(taken.path / "subscribers")
    kept = set(events["msisdn"])
    assert 0 < len(kept) < 200 and kept == set(subscribers["msisdn"])
    original = _tables()["events"]
    original = original[(original["ts"] < "2026-10-04") & original["msisdn"].isin(kept)]
    assert len(events) == len(original)  # every event of a sampled entity
    assert len(pd.read_parquet(taken.path / "plans")) == 3  # no entity column: copied whole
    assert taken.manifest.tables["events"].sampled and not taken.manifest.tables["plans"].sampled


def test_an_interrupted_copy_continues_where_it_stopped(source_settings: Settings) -> None:
    failing = FakeClient(_tables(), fail_on="2026-10-03 00:00:00') AND `ts` <")
    with pytest.raises(SnapshotError, match="connection lost"):
        _take(source_settings, failing)
    assert failing.closed and snap.list_snapshots(source_settings.snapshots_root) == []
    resumed = FakeClient(_tables())
    taken = _take(source_settings, resumed)
    days = [q for q in resumed.queries if "`events`" in q]
    assert len(days) == 1 and "2026-10-03 00:00:00" in days[0]  # the two finished days were kept
    assert taken.manifest.tables["events"].files == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start": None, "end": None}, "window is required"),
        ({"start": "2027-01-01", "end": "2027-01-02"}, "no rows"),
    ],
)
def test_requests_the_source_cannot_serve(source_settings: Settings, kwargs: dict, message: str) -> None:
    with pytest.raises(SnapshotError, match=message):
        _take(source_settings, FakeClient(_tables()), **kwargs)


def test_unknown_source_and_sample_without_entity(settings: Settings) -> None:
    with pytest.raises(SnapshotError, match="Unknown source 'ch'"):
        take_snapshot(settings, "ch", make_request(None, None, None))
    config = AppConfig.model_validate({"sources": {"ch": {"tables": {"plans": {}}}}})
    with pytest.raises(SnapshotError, match="entity_column"):
        take_snapshot(Settings(env=settings.env, config=config), "ch", make_request(None, None, 0.5))


def test_timestamps_are_stored_without_a_time_zone(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "part.parquet"
    stamps = pa.array([pd.Timestamp("2026-10-01 10:00", tz="UTC")], type=pa.timestamp("ms", tz="UTC"))
    pq.write_table(pa.table({"ts": stamps, "n": [1]}), path)
    snap._drop_time_zones(path)
    table = pq.read_table(path)
    assert table.schema.field("ts").type == pa.timestamp("ms")
    assert table.column("ts")[0].as_py() == pd.Timestamp("2026-10-01 10:00")


def test_credentials_come_from_the_environment() -> None:
    cfg = SourceConfig.model_validate(SOURCES["ch"] | {"host": "ch.internal", "port": 8443, "secure": True})
    assert connection_params("ch", cfg, {}) == {
        "host": "ch.internal",
        "port": 8443,
        "secure": True,
        "database": "telco",
        "username": None,
        "password": "",
    }
    env = {
        "JELLYBEAN_SOURCE_CH_USER": "reader",
        "JELLYBEAN_SOURCE_CH_PASSWORD": "s3cret",
        "JELLYBEAN_SOURCE_CH_HOST": "other",
        "JELLYBEAN_SOURCE_CH_PORT": "8123",
        "JELLYBEAN_SOURCE_CH_SECURE": "false",
    }
    params = connection_params("ch", cfg, env)
    assert (params["username"], params["password"], params["host"], params["port"], params["secure"]) == (
        "reader",
        "s3cret",
        "other",
        8123,
        False,
    )
    assert "password" not in SourceConfig.model_fields and "user" not in SourceConfig.model_fields
    with pytest.raises(SnapshotError, match="PORT must be a number"):
        connection_params("ch", cfg, {"JELLYBEAN_SOURCE_CH_PORT": "http"})


# --------------------------------------------------------------------------- #
# Runs and command line                                                       #
# --------------------------------------------------------------------------- #
@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A config.yaml with the source, and the fake client in place of ClickHouse."""
    import yaml

    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"sources": SOURCES}))
    for name in ("workspace", "data"):
        (tmp_path / name).mkdir(exist_ok=True)
    clients: list[FakeClient] = []

    def connect(name, source):
        clients.append(FakeClient(_tables()))
        return clients[-1]

    monkeypatch.setattr("jellybean.sources.clickhouse.connect", connect)
    reset_settings()
    return clients


def test_snapshot_commands(cli_env: list, tmp_path: Path) -> None:
    from jellybean.cli import app

    runner = CliRunner()
    args = ["snapshot", "ch", "--from", "2026-10-01", "--to", "2026-10-03", "--sample", "0.5"]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "Snapshot taken: ch, 2026-10-01 to 2026-10-03, 50% of entities" in first.output
    assert "Snapshot reused" in runner.invoke(app, args).output and len(cli_env) == 1
    listing = runner.invoke(app, ["snapshots"]).output
    assert "ch, 2026-10-01 to 2026-10-03, 50% of entities" in listing and str(tmp_path / "snapshots") in listing
    missing = runner.invoke(app, ["snapshot", "ch"])
    assert missing.exit_code == 2 and "window is required" in missing.output
    assert runner.invoke(app, ["snapshot", "nope"]).exit_code == 2


def test_a_run_remembers_its_source(source_settings: Settings) -> None:
    from jellybean.reporting import build_report
    from jellybean.runtime.agent_runner import Runtime
    from jellybean.runtime.context import RunContext
    from jellybean.runtime.llm_client import FakeLLM
    from jellybean.skills.orquestrador import tools as orchestrator
    from jellybean.spec import load_spec

    taken = _take(source_settings, FakeClient(_tables()), sample=0.5)
    reading = source_settings.with_data_root(taken.path)
    ctx = RunContext.create(reading, seed=1, source=snap.manifest_summary(taken.manifest))
    runtime = Runtime(ctx=ctx, llm=FakeLLM({}), options={"instruction": "find odd subscribers"})
    with runtime.active():
        result = orchestrator.init_run(instruction="find odd subscribers")
    assert result.status == "ok" and result.summary["available_datasets"] == ["events", "plans", "subscribers"]
    assert result.summary["spec"]["source"]["sample"] == 0.5
    spec = load_spec(ctx.spec_path)
    assert spec.source.snapshot == taken.manifest.id and spec.source.sampled_tables == ["events", "subscribers"]
    assert spec.source.label() == "ch, 2026-10-01 to 2026-10-04, 50% of entities"
    report = build_report(ctx)
    assert "**Source:** `ch`" in report and "a 50% sample of the entities of `events`, `subscribers`" in report
    # Agents cannot change where the data came from.
    from jellybean.spec import apply_patch

    with pytest.raises(ValueError):
        apply_patch(spec, {"source": None})


def test_reopening_a_run_reads_its_snapshot_again(cli_env: list, tmp_path: Path) -> None:
    import shutil

    from jellybean.cli import _open_run, app
    from jellybean.settings import get_settings
    from jellybean.spec import AnalysisSpec, DataSource, save_spec

    CliRunner().invoke(app, ["snapshot", "ch", "--from", "2026-10-01", "--to", "2026-10-02"])
    settings = get_settings()
    taken = snap.list_snapshots(settings.snapshots_root)[0]
    run_dir = settings.workspace_root / "r1"
    run_dir.mkdir(parents=True)
    source = DataSource(**snap.manifest_summary(taken.manifest))
    save_spec(run_dir / "spec.yaml", AnalysisSpec(run_id="r1", instruction="i", seed=1, source=source))
    ctx, _ = _open_run("r1", need_data=True)
    assert ctx.settings.data_root == taken.path and ctx.source["snapshot"] == taken.manifest.id
    shutil.rmtree(taken.path)
    assert _open_run("r1")[0].settings.data_root == taken.path  # the report can still be rebuilt
    result = CliRunner().invoke(app, ["resume", "r1"])
    assert result.exit_code == 2 and "no longer in" in result.output


def test_source_options_need_a_source(cli_env: list, monkeypatch: pytest.MonkeyPatch) -> None:
    from jellybean.cli import app

    monkeypatch.setattr("jellybean.runtime.llm_client.build_chat_model", lambda settings: None)
    result = CliRunner().invoke(app, ["run", "x", "--from", "2026-10-01", "--to", "2026-10-02"])
    assert result.exit_code == 2 and "need --source" in result.output


# --------------------------------------------------------------------------- #
# Explorer                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def explored(cli_env: list, tmp_path: Path):
    """A run made from a snapshot, as the explorer finds it on disk."""
    from jellybean.cli import app
    from jellybean.settings import get_settings
    from jellybean.spec import AnalysisSpec, DataSource, save_spec

    CliRunner().invoke(app, ["snapshot", "ch", "--from", "2026-10-01", "--to", "2026-10-03", "--sample", "0.5"])
    settings = get_settings()
    taken = snap.list_snapshots(settings.snapshots_root)[0]
    source = DataSource(**snap.manifest_summary(taken.manifest))
    for run_id, origin in (("from_source", source), ("from_local", None)):
        (settings.workspace_root / run_id).mkdir()
        spec = AnalysisSpec(run_id=run_id, instruction="find odd subscribers", seed=1, source=origin)
        save_spec(settings.workspace_root / run_id / "spec.yaml", spec)
    return settings, taken


def test_the_explorer_reads_the_snapshot_of_a_run(explored) -> None:
    import shutil

    from jellybean.ui import data

    settings, taken = explored
    runs = data.list_runs(settings.workspace_root).set_index("run")["source"].to_dict()
    assert runs == {"from_source": "ch · 2026-10-01 to 2026-10-03 · 50%", "from_local": "local"}
    run = data.Run(settings.workspace_root, "from_source", settings.data_root, settings.snapshots_root)
    assert run.raw_root == taken.path and not run.raw_missing
    assert set(run.tables()) == {"events", "subscribers", "plans"}
    frame, total = run.page("events", limit=5)
    assert len(frame) == 5 and total == taken.manifest.tables["events"].rows
    local = data.Run(settings.workspace_root, "from_local", settings.data_root, settings.snapshots_root)
    assert local.raw_root == settings.data_root and not local.raw_missing and local.tables() == {}
    listing = data.list_snapshots(settings.snapshots_root, settings.workspace_root)
    assert listing.iloc[0][["source", "from", "to", "sample", "runs"]].tolist() == [
        "ch",
        "2026-10-01",
        "2026-10-03",
        "50%",
        "from_source",
    ]
    shutil.rmtree(taken.path)
    assert run.raw_missing and run.tables() == {}
    assert data.list_snapshots(settings.snapshots_root, settings.workspace_root).empty


@pytest.mark.parametrize("missing", [False, True])
def test_explorer_pages_of_a_run_from_a_source(explored, missing: bool) -> None:
    import shutil

    testing = pytest.importorskip("streamlit.testing.v1")
    from jellybean.ui import data

    _, taken = explored
    if missing:
        shutil.rmtree(taken.path)
    script = str(Path(data.__file__).with_name("app.py"))
    for view in ("Snapshots", "Overview", "Tables"):
        app = testing.AppTest.from_file(script, default_timeout=120)
        app.session_state["run"] = "from_source"
        app.session_state["view"] = view
        app.run()
        assert not app.exception, [e.value for e in app.exception]
        assert not app.error, [e.value for e in app.error]
        if view == "Snapshots":
            assert app.header[0].value == "Snapshots" and bool(app.info) == missing
            continue
        assert any("50% of the entities of events, subscribers" in c.value for c in app.caption)
        assert bool(app.warning) == missing


# --------------------------------------------------------------------------- #
# The real SQL dialect                                                        #
# --------------------------------------------------------------------------- #
class EmbeddedClickHouse:
    """The snapshot client protocol on an in-process ClickHouse (``chdb``)."""

    def __init__(self, session) -> None:
        self.session = session

    def columns(self, database: str, table: str) -> list[tuple[str, str]]:
        answer = self.session.query(f"DESCRIBE TABLE `{database}`.`{table}`", "JSONCompact").bytes()
        return [(row[0], row[1]) for row in json.loads(answer)["data"]]

    def stream_parquet(self, sql: str):
        yield self.session.query(sql + " SETTINGS output_format_parquet_string_as_string = 1", "Parquet").bytes()

    def close(self) -> None:
        pass


def test_the_queries_run_on_clickhouse(source_settings: Settings) -> None:
    chdb_session = pytest.importorskip("chdb.session")
    session = chdb_session.Session()
    session.query("CREATE DATABASE telco")
    session.query(
        "CREATE TABLE telco.events (event_id UInt64, msisdn String, ts DateTime('Europe/Lisbon'), "
        "day Date, kind Enum8('call' = 1, 'sms' = 2), ref UUID, seen Nullable(DateTime64(3, 'UTC'))) "
        "ENGINE = MergeTree ORDER BY event_id"
    )
    session.query(
        "INSERT INTO telco.events SELECT number, concat('3519', leftPad(toString(number % 200), 5, '0')), "
        "toDateTime('2026-10-01 00:00:00', 'Europe/Lisbon') + number * 130, toDate('2026-10-01'), "
        "if(number % 2 = 0, 'call', 'sms'), generateUUIDv4(), if(number % 5 = 0, NULL, now64(3)) FROM numbers(2000)"
    )
    session.query("CREATE TABLE telco.subscribers (msisdn String, plan UInt8) ENGINE = MergeTree ORDER BY msisdn")
    session.query(
        "INSERT INTO telco.subscribers SELECT concat('3519', leftPad(toString(number), 5, '0')), number % 3 "
        "FROM numbers(200)"
    )
    session.query("CREATE TABLE telco.plans (plan UInt8, name String) ENGINE = MergeTree ORDER BY plan")
    session.query("INSERT INTO telco.plans VALUES (0, 'a'), (1, 'b'), (2, 'c')")

    request = make_request("2026-10-01", "2026-10-03", 0.25)
    taken = take_snapshot(
        source_settings, "ch", request, client_factory=lambda name, source: EmbeddedClickHouse(session)
    )
    events = pd.read_parquet(taken.path / "events")
    # Wall-clock time of Lisbon, without a zone: the window is exactly the two local days.
    assert events["ts"].dt.tz is None and events["seen"].dt.tz is None
    assert events["ts"].min() >= pd.Timestamp("2026-10-01") and events["ts"].max() < pd.Timestamp("2026-10-03")
    first = events.sort_values("event_id").iloc[0]
    assert first["ts"] == pd.Timestamp("2026-10-01") + pd.Timedelta(seconds=130 * int(first["event_id"]))
    assert set(events["kind"]) == {"call", "sms"} and events["ref"].str.len().eq(36).all()
    assert str(events["day"].iloc[0])[:10] == "2026-10-01" and events["seen"].isna().any()
    # Whole entities, the same ones in both tables.
    kept = set(events["msisdn"])
    assert 20 < len(kept) < 80 and kept == set(pd.read_parquet(taken.path / "subscribers")["msisdn"])
    in_window = 2 * 24 * 3600 // 130 + 1
    assert len(events) == sum(1 for i in range(in_window) if f"3519{i % 200:05d}" in kept)
    assert len(pd.read_parquet(taken.path / "plans")) == 3
