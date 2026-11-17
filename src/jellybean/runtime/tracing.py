# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Append-only JSONL tracing (``workspace/<run_id>/logs/trace.jsonl``).

One line per LLM call (tokens, duration) and per tool call (arguments, output
size, raw rows returned, duration, artifact created). Tool outputs are never
written here, only arguments and sizes (privacy requirement, section 11).

Error messages are recorded: a failed tool call carries a short ``error`` text
(cut to ``_MAX_ARG_CHARS``, with values that look like personal data masked), so
the person running the CLI can see why it failed. An exception message can quote
a data value (e.g. a failed cast), which is why it is cut and masked. Full
tracebacks of unexpected exceptions go to ``logs/errors.log``, never to the trace.
"""

from __future__ import annotations

import contextlib
import json
import threading
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jellybean.runtime.context import maybe_current_run

# A single lock is enough: trace lines are tiny and writes are rare.
_LOCK = threading.Lock()

# Arguments longer than this are shortened in the trace to keep lines compact.
_MAX_ARG_CHARS = 500


def _compact(value: Any) -> Any:
    """Shorten long argument values so the trace stays small."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) > _MAX_ARG_CHARS:
        return text[:_MAX_ARG_CHARS] + "...<cut>"
    return value


def safe_error_text(message: Any) -> str:
    """Error text fit for the trace and the console: one line, masked, cut."""
    text = " ".join(str(message).split())
    ctx = maybe_current_run()
    if ctx is not None:
        from jellybean.skills._shared.masking import Masker

        text = Masker(ctx.settings.config.masking, salt=ctx.run_id).mask_text(text)
    return text if len(text) <= _MAX_ARG_CHARS else text[:_MAX_ARG_CHARS] + "...<cut>"


def log_exception(where: str, exc: BaseException) -> None:
    """Append the full traceback of an unexpected exception to ``logs/errors.log``.

    A no-op outside a run. The file stays in the run workspace (never sent to the LLM).
    """
    ctx = maybe_current_run()
    if ctx is None:
        return
    stamp = datetime.now(UTC).isoformat(timespec="milliseconds")
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    with _LOCK, contextlib.suppress(OSError):
        ctx.logs_dir.mkdir(parents=True, exist_ok=True)
        with ctx.errors_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] agent={ctx.agent} {where}\n{text}\n")


# Callables notified of every event (e.g. the console progress view of the CLI).
_LISTENERS: list[Callable[[dict[str, Any]], None]] = []


def add_listener(listener: Callable[[dict[str, Any]], None]) -> None:
    """Register a callable that receives every traced event (as a dict)."""
    _LISTENERS.append(listener)


def remove_listener(listener: Callable[[dict[str, Any]], None]) -> None:
    if listener in _LISTENERS:
        _LISTENERS.remove(listener)


def write_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSON event to ``path``, adding a UTC timestamp, and notify listeners."""
    record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **event}
    line = json.dumps(record, default=str, ensure_ascii=False)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    for listener in list(_LISTENERS):
        with contextlib.suppress(Exception):
            listener(record)


def trace_event(kind: str, **fields: Any) -> None:
    """Write an event to the active run's trace; a no-op outside a run."""
    ctx = maybe_current_run()
    if ctx is None:
        return
    write_event(ctx.trace_path, {"kind": kind, "run_id": ctx.run_id, "agent": ctx.agent, **fields})


def trace_tool_call(
    tool: str,
    arguments: dict[str, Any],
    status: str,
    output_chars: int,
    raw_rows: int,
    duration_s: float,
    artifact_id: str | None,
    truncated: bool,
    error: str | None = None,
) -> None:
    """Record a tool call. ``arguments`` are compacted, never the output.

    ``error`` is the message of a failed call; it is masked and cut before writing.
    """
    extra = {"error": safe_error_text(error)} if error else {}
    trace_event(
        "tool_call",
        tool=tool,
        arguments={k: _compact(v) for k, v in arguments.items()},
        status=status,
        output_chars=output_chars,
        raw_rows=raw_rows,
        duration_s=round(duration_s, 4),
        artifact_id=artifact_id,
        truncated=truncated,
        **extra,
    )
