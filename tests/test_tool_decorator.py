"""Unit tests for the @tool decorator and ToolResult contract."""

from __future__ import annotations

import json
import time

from jellybean.runtime.context import RunContext
from jellybean.runtime.tool_registry import ToolResult, enforce_limits, get_tool_spec, parse_docstring, tool


@tool
def add(a: int, b: int = 2) -> ToolResult:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number,
            defaults to two.
    """
    return ToolResult.ok({"total": a + b})


@tool
def boom() -> ToolResult:
    """Always fails."""
    raise RuntimeError("kaboom")


@tool
def type_error_inside() -> ToolResult:
    """Raises TypeError inside the body (must not be reported as bad arguments)."""
    return len(5)  # type: ignore[arg-type]


@tool
def many_rows(n: int) -> ToolResult:
    """Return n raw rows."""
    return ToolResult.ok({"rows": [[i, "value"] for i in range(n)]})


@tool
def huge(n: int) -> ToolResult:
    """Return a very large nested summary."""
    return ToolResult.ok({"items": [{"name": f"item{i}", "text": "y" * 300} for i in range(n)], "note": "z" * 9000})


@tool
def returns_dict() -> ToolResult:
    """Return a plain dict, which the decorator coerces."""
    return {"status": "ok", "summary": {"x": 1}}  # type: ignore[return-value]


@tool
def with_seed(seed: int | None = None) -> ToolResult:
    """Echo the seed."""
    return ToolResult.ok({"seed": seed})


@tool(timeout_s=0.2)
def slow() -> ToolResult:
    """Sleep longer than the timeout."""
    time.sleep(2)
    return ToolResult.ok()


def test_schema_from_hints_and_docstring() -> None:
    spec = get_tool_spec(add)
    assert spec is not None
    assert spec.name == "add"
    assert spec.description == "Add two numbers."
    schema = spec.input_schema
    assert schema["required"] == ["a"]
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["description"] == "Second number, defaults to two."


def test_parse_docstring_handles_missing_doc() -> None:
    assert parse_docstring(None) == ("", {})


def test_valid_call_positional_and_keyword() -> None:
    assert add(1).summary == {"total": 3}
    assert add(a=1, b=5).summary == {"total": 6}


def test_validation_errors_become_error_results() -> None:
    assert add(a="not a number").status == "error"
    missing = add()
    assert missing.status == "error" and "Invalid arguments" in missing.summary["error"]
    extra = add(a=1, c=3)
    assert extra.status == "error"


def test_exceptions_never_escape() -> None:
    result = boom()
    assert result.status == "error"
    assert "kaboom" in result.summary["error"]
    inner = type_error_inside()
    assert inner.status == "error" and inner.summary["error"].startswith("TypeError")


def test_dict_results_are_coerced() -> None:
    assert returns_dict().summary == {"x": 1}


def test_raw_rows_are_capped() -> None:
    result = many_rows(100)
    assert result.raw_row_count == 20
    assert result.truncated


def test_output_size_is_capped() -> None:
    result = huge(200)
    assert len(result.to_json()) <= 4000
    assert result.truncated
    assert result.status == "ok"


def test_enforce_limits_last_resort_preview() -> None:
    big = ToolResult.ok({"a": "b" * 50_000})
    limited = enforce_limits(big, max_chars=1000, max_rows=20)
    assert len(limited.to_json()) <= 1000
    assert limited.truncated


def test_seed_injection_outside_and_inside_run(run_ctx: RunContext) -> None:
    assert with_seed().summary["seed"] == run_ctx.seed
    assert with_seed(seed=99).summary["seed"] == 99


def test_timeout_returns_error() -> None:
    started = time.perf_counter()
    result = slow()
    assert result.status == "error"
    assert "timed out" in result.summary["error"]
    assert time.perf_counter() - started < 1.5


def test_calls_are_traced(run_ctx: RunContext) -> None:
    many_rows(30)
    lines = [json.loads(line) for line in run_ctx.trace_path.read_text().splitlines()]
    event = lines[-1]
    assert event["kind"] == "tool_call"
    assert event["tool"] == "many_rows"
    assert event["raw_rows"] == 20
    assert event["output_chars"] > 0
    assert event["arguments"] == {"n": 30}


def test_shortened_lists_are_named_in_a_warning() -> None:
    from jellybean.runtime.tool_registry import ToolResult, enforce_limits

    result = enforce_limits(ToolResult.ok({"datasets": [{"name": f"d{i}" * 40} for i in range(200)]}), 4000, 20)
    assert result.truncated and len(result.to_json()) <= 4000
    assert any("datasets" in w and "of 200" in w and "do not treat them as complete" in w for w in result.warnings)


# ------------------------------------------------------------------ error visibility
def _trace(ctx: RunContext) -> list[dict]:
    return [json.loads(line) for line in ctx.trace_path.read_text(encoding="utf-8").splitlines()]


def test_raising_tool_records_its_error_in_the_trace(run_ctx: RunContext) -> None:
    assert boom().status == "error"
    event = _trace(run_ctx)[-1]
    assert event["tool"] == "boom" and event["status"] == "error"
    assert event["error"] == "RuntimeError: kaboom"
    # The traceback goes to errors.log, never to the trace.
    log = run_ctx.errors_path.read_text(encoding="utf-8")
    assert "Traceback" in log and "kaboom" in log and "tool=boom" in log
    assert "Traceback" not in run_ctx.trace_path.read_text(encoding="utf-8")


def test_error_results_are_traced_and_successes_carry_no_error(run_ctx: RunContext) -> None:
    @tool
    def refuse() -> ToolResult:
        """Returns an error result without raising."""
        return ToolResult.error("nothing to do here")

    refuse()
    add(1)
    events = _trace(run_ctx)
    assert events[-2]["error"] == "nothing to do here"
    assert "error" not in events[-1]
    assert not run_ctx.errors_path.exists()  # no traceback for an expected error


def test_traced_error_is_cut_and_masked(run_ctx: RunContext) -> None:
    @tool
    def leak() -> ToolResult:
        """Fails with a message quoting a personal value."""
        raise ValueError("Could not convert string '+351912345678' to INT32 " + "x" * 2000)

    leak()
    error = _trace(run_ctx)[-1]["error"]
    assert "+351912345678" not in error and "masked:" in error
    assert len(error) < 600 and error.endswith("...<cut>")


def test_console_prints_the_error_of_a_raising_tool(run_ctx: RunContext) -> None:
    import re

    from jellybean.runtime.progress import ConsoleProgress
    from jellybean.runtime.tracing import add_listener, remove_listener

    lines: list[str] = []
    view = ConsoleProgress(echo=lambda line: lines.append(re.sub(r"\x1b\[[0-9;]*m", "", line)))
    add_listener(view)
    try:
        boom()
    finally:
        remove_listener(view)
    assert any("boom()" in line and "✖ error" in line for line in lines)
    assert any("↳ RuntimeError: kaboom" in line for line in lines)  # normal mode, not only --verbose
