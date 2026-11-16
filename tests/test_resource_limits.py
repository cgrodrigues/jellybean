"""DuckDB resource limits, real tool timeouts and prompt interruption."""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import duckdb
import pytest

from jellybean.runtime.cancellation import ToolCancelledError
from jellybean.runtime.context import RunContext
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.skills._shared import catalog

# About 10^12 pairs: never finishes on its own, and DuckDB can interrupt it.
ENDLESS = "SELECT sum(a.range * b.range) FROM range(1000000) a, range(1000000) b"
OPENED: list[duckdb.DuckDBPyConnection] = []


@tool(timeout_s=1)
def endless_query() -> ToolResult:
    """Runs a query that never ends."""
    con = catalog.connect()
    OPENED.append(con)
    try:
        return ToolResult.ok({"total": con.execute(ENDLESS).fetchall()[0][0]})
    finally:
        con.close()


@tool(timeout_s=1)
def stubborn_queries() -> ToolResult:
    """Swallows the interruption of a query and starts another one, again and again."""
    con = catalog.connect()
    OPENED.append(con)
    for _ in range(50):
        try:
            con.execute(ENDLESS).fetchall()
        except duckdb.Error:
            continue
    return ToolResult.ok({})


def _tool_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("tool-") and t.is_alive()]


def _wait_until(condition, seconds: float = 8.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def _is_closed(con: duckdb.DuckDBPyConnection) -> bool:
    try:
        con.execute("SELECT 1")
    except duckdb.Error:
        return True
    return False


@pytest.mark.parametrize("fn", [endless_query, stubborn_queries])
def test_a_tool_is_stopped_at_its_timeout(run_ctx: RunContext, fn) -> None:
    OPENED.clear()
    started = time.monotonic()
    result = fn()
    elapsed = time.monotonic() - started
    assert result.status == "error" and "timed out after 1s and was stopped" in result.summary["error"]
    assert elapsed < 6  # the caller is not kept waiting for the work
    # The worker thread ends (it no longer burns CPU or holds memory) and its connection is closed.
    assert _wait_until(lambda: not _tool_threads()), [t.name for t in _tool_threads()]
    assert OPENED and all(_is_closed(con) for con in OPENED)


def test_a_cancelled_call_cannot_open_new_connections(run_ctx: RunContext) -> None:
    from jellybean.runtime.cancellation import CallScope, set_scope

    scope = CallScope()
    token = set_scope(scope)
    try:
        catalog.connect().close()  # fine before cancellation
        scope.cancel()
        with pytest.raises(ToolCancelledError):
            catalog.connect()
    finally:
        set_scope(None)
        del token


def test_cancelling_a_call_stops_the_calls_nested_in_it(run_ctx: RunContext) -> None:
    """A delegated agent's tool runs under the delegate call: stopping the outer one stops both."""

    @tool(timeout_s=60)
    def inner() -> ToolResult:
        """Nested call with a long timeout of its own."""
        con = catalog.connect()
        OPENED.append(con)
        try:
            return ToolResult.ok({"total": con.execute(ENDLESS).fetchall()[0][0]})
        finally:
            con.close()

    @tool(timeout_s=1)
    def outer() -> ToolResult:
        """Outer call that times out first."""
        return inner()

    OPENED.clear()
    assert outer().status == "error"
    assert _wait_until(lambda: not _tool_threads()), [t.name for t in _tool_threads()]
    assert OPENED and all(_is_closed(con) for con in OPENED)


def test_connections_have_limits_and_spill_inside_the_run(run_ctx: RunContext) -> None:
    limits = run_ctx.settings.config.limits
    limits.duckdb_memory_limit, limits.duckdb_threads = "200MB", 2
    con = catalog.connect()
    try:
        setting = lambda name: con.execute(f"SELECT current_setting('{name}')").fetchall()[0][0]  # noqa: E731
        assert setting("threads") == 2
        assert setting("memory_limit").replace(" ", "") in ("190.7MiB", "200.0MB", "200MB")
        assert Path(setting("temp_directory")) == run_ctx.tmp_dir
        # Far more distinct pairs than fit in 200 MB: it must spill to disk, not fail or grow.
        rows = con.execute("SELECT count(*) FROM (SELECT DISTINCT range, hash(range) FROM range(20000000))").fetchall()
        assert rows[0][0] == 20_000_000
        # The limits are locked: a query cannot lift them.
        with pytest.raises(duckdb.Error):
            con.execute("SET memory_limit = '64GB'")
    finally:
        con.close()


def test_ctrl_c_returns_the_prompt(tmp_path: Path) -> None:
    """SIGINT during a long tool call ends the process within seconds."""
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from jellybean.cli import _interruptible
        from jellybean.runtime.context import RunContext, use_run
        from jellybean.runtime.tool_registry import ToolResult, tool
        from jellybean.settings import AppConfig, EnvSettings, Settings
        from jellybean.skills._shared import catalog

        base = Path({str(tmp_path)!r})
        env = EnvSettings(data_root=base / "data", workspace_root=base / "ws", ground_truth_root=base / "gt",
                          config_file=base / "none.yaml")
        (base / "data").mkdir()
        ctx = RunContext.create(Settings(env=env, config=AppConfig()), seed=1, run_id="r")

        @tool(timeout_s=600)
        def endless() -> ToolResult:
            '''Never ends.'''
            con = catalog.connect()
            return ToolResult.ok({{"t": con.execute({ENDLESS!r}).fetchall()}})

        with use_run(ctx), _interruptible(ctx):
            print("started", flush=True)
            endless()
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "started"
    time.sleep(1.0)
    proc.send_signal(signal.SIGINT)
    try:
        _, err = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("the process did not exit after Ctrl+C")
    assert proc.returncode == 130 and "Interrupted. Continue with: jellybean resume r" in err
