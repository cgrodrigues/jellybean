"""Live progress of a run in the terminal.

The console view listens to the trace events (see :mod:`tracing`) and prints a
compact, indented line per step, so the user can follow a run while it
happens:

    [00:12] == Phase: discovery ==
    [00:12] ▶ profiler: Profile the datasets in the spec and propose ...
    [00:13]   · profile_table(dataset=traffic)  1.5s
    [00:41] ✔ profiler done — 6 turns, 76k tokens (61% cached), 29.0s
    [00:41]   ↳ outputs:
    [00:41]     - profile_traffic_v1 (json)  /…/workspace/<run_id>/artifacts/profile_traffic_v1.json
    [00:42] ◆ control_1:join[0] → confirmed ... (auto)

It only prints what the trace already records (arguments, sizes and the message
of a failed call, never tool outputs). Failures are always shown with their
message, also without ``--verbose``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

import typer

# Tools whose call is already shown by another line (a delegated agent's start/end).
_HIDDEN_TOOLS = {"delegate", "finish"}
# Longest error message printed on one line.
_MAX_ERROR_CHARS = 400
_STATUS_ICON = {
    "done": ("✔", typer.colors.GREEN),
    "needs_rework": ("↺", typer.colors.YELLOW),
    "needs_human": ("?", typer.colors.YELLOW),
    "failed": ("✖", typer.colors.RED),
}


def _short(value: Any, limit: int = 40) -> str:
    """Compact representation of an argument value for one line."""
    if isinstance(value, dict):
        text = "{" + ", ".join(list(value)[:4]) + ("…" if len(value) > 4 else "") + "}"
    elif isinstance(value, list):
        text = (
            json.dumps(value, default=str) if len(json.dumps(value, default=str)) <= limit else f"[{len(value)} items]"
        )
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def display_path(path: str) -> str:
    """Path relative to the current folder when possible (shorter, still clickable)."""
    from pathlib import Path

    try:
        return str(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return path


def _tokens(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


class ConsoleProgress:
    """Trace listener that prints the progress of a run.

    Args:
        verbose: Also print every LLM turn (tokens, cache reads, duration).
        echo: Output function (defaults to ``typer.echo``); useful for tests.
    """

    def __init__(self, verbose: bool = False, echo: Callable[[str], None] | None = None, max_outputs: int = 12) -> None:
        self.verbose = verbose
        self.max_outputs = max_outputs
        self.echo = echo or (lambda line: typer.echo(line))
        self.started = time.monotonic()
        self._stack: list[tuple[str, float, int, int]] = []  # agent, start, tokens in, cached
        self._outputs: list[list[dict[str, Any]]] = []  # artifacts created, per open agent
        self._lock = threading.Lock()

    # ------------------------------------------------------------- helpers
    def _line(self, text: str, color: str | None = None, indent: int | None = None) -> None:
        elapsed = int(time.monotonic() - self.started)
        prefix = f"[{elapsed // 60:02d}:{elapsed % 60:02d}] "
        pad = "  " * (len(self._stack) if indent is None else indent)
        body = typer.style(text, fg=color) if color else text
        self.echo(prefix + pad + body)

    # -------------------------------------------------------------- events
    def __call__(self, event: dict[str, Any]) -> None:
        with self._lock:
            handler = getattr(self, f"_on_{event.get('kind')}", None)
            if handler:
                handler(event)

    def _on_init_run(self, event: dict[str, Any]) -> None:
        self._line(f"Run initialised (report language: {event.get('report_language')})")

    def _on_spec_update(self, event: dict[str, Any]) -> None:
        if event.get("phase"):
            self._line(f"== Phase: {event['phase']} ==", typer.colors.CYAN, indent=0)

    def _on_agent_start(self, event: dict[str, Any]) -> None:
        task = " ".join(str(event.get("task", "")).split())
        self._line(f"▶ {event['agent']}: {_short(task, 90)}", typer.colors.BLUE)
        self._stack.append((event["agent"], time.monotonic(), 0, 0))
        self._outputs.append([])

    def _on_llm_call(self, event: dict[str, Any]) -> None:
        if self._stack:
            agent, start, tokens, cached = self._stack[-1]
            self._stack[-1] = (
                agent,
                start,
                tokens + event.get("input_tokens", 0) + event.get("output_tokens", 0),
                cached + (event.get("cache_read_tokens") or 0),
            )
        if self.verbose:
            self._line(
                f"… LLM turn {event.get('iteration')}: in {_tokens(event.get('input_tokens', 0))} "
                f"(cached {_tokens(event.get('cache_read_tokens') or 0)}), "
                f"out {_tokens(event.get('output_tokens', 0))}, {event.get('duration_s', 0):.1f}s, "
                f"{event.get('tool_calls', 0)} tool call(s)",
                typer.colors.BRIGHT_BLACK,
            )

    def _on_tool_call(self, event: dict[str, Any]) -> None:
        tool = event.get("tool")
        failed = event.get("status") == "error"
        if tool in _HIDDEN_TOOLS and not failed:
            return
        args = event.get("arguments") or {}
        shown = list(args.items()) if self.verbose else list(args.items())[:2]
        text = f"· {tool}(" + ", ".join(f"{k}={_short(v)}" for k, v in shown) + ")"
        text += f"  {event.get('duration_s', 0):.1f}s"
        if event.get("artifact_id") and self.verbose:
            text += f" → {event['artifact_id']}"
        if failed:
            self._line(text + "  ✖ error", typer.colors.RED)
            message = event.get("error") or "no error message recorded"
            self._line(f"  ↳ {_short(message, _MAX_ERROR_CHARS)}", typer.colors.RED)
        else:
            self._line(text)

    def _on_decision(self, event: dict[str, Any]) -> None:
        self._line(
            f"◆ {event.get('topic')} → {_short(event.get('choice'), 80)} ({event.get('by')})", typer.colors.MAGENTA
        )

    def _on_artifact(self, event: dict[str, Any]) -> None:
        if self._outputs:
            self._outputs[-1].append(event)
        if self.verbose:
            self._line(f"+ {event.get('artifact_id')} ({event.get('type')})", typer.colors.BRIGHT_BLACK)

    def _on_agent_end(self, event: dict[str, Any]) -> None:
        agent, start, tokens, cached = self._stack.pop() if self._stack else (event["agent"], time.monotonic(), 0, 0)
        icon, color = _STATUS_ICON.get(event.get("status"), ("•", None))
        share = f" ({cached / tokens:.0%} cached)" if tokens and cached else ""
        self._line(
            f"{icon} {agent} {event.get('status')} — {event.get('iterations')} turns, "
            f"{_tokens(tokens)} tokens{share}, {time.monotonic() - start:.1f}s",
            color,
        )
        if event.get("status") == "failed":
            reason = event.get("reason") or ", ".join(event.get("issues") or []) or "no reason recorded"
            self._line(f"  ↳ {_short(reason, _MAX_ERROR_CHARS)}", typer.colors.RED)
        outputs = self._outputs.pop() if self._outputs else []
        # Datasets (tables) and plots are listed; documents are summarised unless verbose.
        listed = [o for o in outputs if self.verbose or o.get("type") in ("table", "plot")]
        documents = len(outputs) - len(listed)
        if listed or documents:
            base = len(self._stack) + 1
            self._line("↳ outputs:", indent=base)
            for o in listed[: self.max_outputs]:
                rows = f", {o['rows']:,} rows" if isinstance(o.get("rows"), int) else ""
                self._line(f"- {o['artifact_id']} ({o['type']}{rows})  {display_path(o['path'])}", indent=base + 1)
            if len(listed) > self.max_outputs:
                self._line(f"… and {len(listed) - self.max_outputs} more", indent=base + 1)
            if documents:
                self._line(
                    f"+ {documents} document(s) (json): jellybean outputs <run_id> --type json",
                    typer.colors.BRIGHT_BLACK,
                    indent=base + 1,
                )

    def _on_llm_error(self, event: dict[str, Any]) -> None:
        self._line(f"✖ LLM error: {_short(event.get('error'), _MAX_ERROR_CHARS)}", typer.colors.RED)
