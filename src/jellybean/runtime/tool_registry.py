# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Tool contract: the ``ToolResult`` model and the ``@tool`` decorator.

Every tool in every skill uses :func:`tool`. The decorator is the single place
where the non-negotiable rules of the system are enforced, so no
individual tool can forget them:

* parameters are validated with pydantic before the function runs;
* exceptions become ``status="error"`` results (they never kill the agent);
* a per-tool timeout is applied, and the work is stopped when it is reached;
* ``seed`` is injected from the active run when the caller omits it;
* raw rows (``summary["rows"]``) are capped at ``max_raw_rows``;
* the serialised result is capped at ``max_output_chars`` and marked
  ``truncated`` when shrinking was needed;
* every call is written to the run trace.

The decorator also derives a JSON schema from type hints and the docstring, which
the ToolRegistry (phase 3) hands to the LLM.
"""

from __future__ import annotations

import contextvars
import copy
import inspect
import json
import math
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from decimal import Decimal
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from jellybean.runtime.cancellation import CallScope, current_scope, set_scope
from jellybean.runtime.context import maybe_current_run
from jellybean.runtime.tracing import log_exception, trace_tool_call
from jellybean.settings import get_settings

# Raw data rows must always be returned under this summary key, so the
# decorator (and the contract tests) can count and cap them.
ROWS_KEY = "rows"

# Attribute set on decorated functions so registries can discover them.
TOOL_SPEC_ATTR = "__jellybean_tool__"

_current_tool: contextvars.ContextVar[str | None] = contextvars.ContextVar("jellybean_current_tool", default=None)


def current_tool() -> str | None:
    """Name of the tool currently executing (used for artifact lineage)."""
    return _current_tool.get()


# --------------------------------------------------------------------------- #
# Result model                                                                #
# --------------------------------------------------------------------------- #
class ToolResult(BaseModel):
    """Uniform return value of every tool."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    artifact_id: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    truncated: bool = False

    @classmethod
    def ok(
        cls,
        summary: dict[str, Any] | None = None,
        artifact_id: str | None = None,
        warnings: Iterable[str] = (),
    ) -> ToolResult:
        """Build a successful result."""
        return cls(status="ok", artifact_id=artifact_id, summary=summary or {}, warnings=list(warnings))

    @classmethod
    def error(cls, message: str, **details: Any) -> ToolResult:
        """Build an error result with a helpful message for the LLM."""
        return cls(status="error", summary={"error": message, **details})

    def to_json(self) -> str:
        """Serialisation sent to the LLM; its length is what the limit measures."""
        return json.dumps(self.model_dump(), ensure_ascii=False, separators=(",", ":"), default=str)

    @property
    def raw_row_count(self) -> int:
        """Number of raw data rows carried by this result."""
        rows = self.summary.get(ROWS_KEY)
        return len(rows) if isinstance(rows, list) else 0


# --------------------------------------------------------------------------- #
# Tool specification                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolSpec:
    """Static description of a tool, used to build LLM tool definitions."""

    name: str
    description: str
    input_schema: dict[str, Any]
    params_model: type[BaseModel]
    max_output_chars: int | None
    timeout_s: float | None
    func: Callable[..., Any]


def get_tool_spec(obj: Any) -> ToolSpec | None:
    """Return the :class:`ToolSpec` of a decorated function, else ``None``."""
    return getattr(obj, TOOL_SPEC_ATTR, None)


def iter_module_tools(module: ModuleType) -> list[ToolSpec]:
    """List the tools defined (decorated) in ``module``."""
    specs = [get_tool_spec(v) for v in vars(module).values()]
    return [s for s in specs if s is not None]


# --------------------------------------------------------------------------- #
# Docstring parsing (Google style)                                            #
# --------------------------------------------------------------------------- #
_SECTION_RE = re.compile(r"^\s*(Args|Arguments|Parameters|Returns|Raises|Examples?):\s*$")
_ARG_RE = re.compile(r"^(\s+)(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_ARG_SECTIONS = {"Args", "Arguments", "Parameters"}


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into description and per-argument text."""
    if not doc:
        return "", {}
    description: list[str] = []
    args: dict[str, str] = {}
    section: str | None = None
    last_arg: str | None = None
    arg_indent: int | None = None
    for line in inspect.cleandoc(doc).splitlines():
        header = _SECTION_RE.match(line)
        if header:
            section, last_arg, arg_indent = header.group(1), None, None
            continue
        if section is None:
            description.append(line)
            continue
        if section not in _ARG_SECTIONS:
            continue
        match = _ARG_RE.match(line)
        indent = len(line) - len(line.lstrip())
        if match and (arg_indent is None or indent == arg_indent):
            arg_indent = indent
            last_arg = match.group(2)
            args[last_arg] = match.group(3).strip()
        elif last_arg and line.strip():
            # Continuation line of the previous argument.
            args[last_arg] += " " + line.strip()
    return "\n".join(description).strip(), args


# --------------------------------------------------------------------------- #
# JSON helpers and output limiting                                            #
# --------------------------------------------------------------------------- #
def to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas/datetime values into plain JSON-compatible types."""
    # Imported lazily so the runtime does not require numpy to be loaded.
    import numpy as np

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (datetime, date, dt_time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump())
    # pandas.NA / NaT and anything else unknown.
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def _largest_list(node: Any) -> tuple[Any, Any, list[Any]] | None:
    """Find the (container, key, list) with the largest serialised list."""
    best: tuple[int, tuple[Any, Any, list[Any]]] | None = None

    def visit(container: Any, key: Any, value: Any) -> None:
        nonlocal best
        if isinstance(value, list):
            if len(value) > 1:
                size = len(_dumps(value))
                if best is None or size > best[0]:
                    best = (size, (container, key, value))
            for i, item in enumerate(value):
                visit(value, i, item)
        elif isinstance(value, dict):
            for k, v in value.items():
                visit(value, k, v)

    visit(None, None, node)
    return best[1] if best else None


def _longest_string(node: Any) -> tuple[Any, Any, str] | None:
    """Find the (container, key, string) with the longest string value."""
    best: tuple[int, tuple[Any, Any, str]] | None = None

    def visit(container: Any, key: Any, value: Any) -> None:
        nonlocal best
        if isinstance(value, str) and container is not None:
            if len(value) > 40 and (best is None or len(value) > best[0]):
                best = (len(value), (container, key, value))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(value, i, item)
        elif isinstance(value, dict):
            for k, v in value.items():
                visit(value, k, v)

    visit(None, None, node)
    return best[1] if best else None


def enforce_limits(result: ToolResult, max_chars: int, max_rows: int) -> ToolResult:
    """Cap raw rows and total size of ``result``; mark it truncated if needed.

    Shrinking strategy, applied until the result fits:
      1. cap ``summary["rows"]`` at ``max_rows``;
      2. repeatedly halve the largest list in the summary;
      3. cut the longest strings in half;
      4. as a last resort replace the summary by a short JSON preview.
    """
    summary = to_jsonable(result.summary)
    warnings = [w if len(w) <= 300 else w[:297] + "..." for w in result.warnings[:20]]
    truncated = result.truncated

    rows = summary.get(ROWS_KEY)
    if isinstance(rows, list) and len(rows) > max_rows:
        summary[ROWS_KEY] = rows[:max_rows]
        truncated = True
        warnings.append(f"Raw rows capped at {max_rows}.")

    def build(s: dict[str, Any], t: bool) -> ToolResult:
        return ToolResult(
            status=result.status, artifact_id=result.artifact_id, summary=s, warnings=warnings, truncated=t
        )

    candidate = build(summary, truncated)
    if len(candidate.to_json()) <= max_chars:
        return candidate

    # The output is too large: work on a copy and shrink step by step. Every list
    # that is shortened is named in a warning, so an agent never mistakes a cut
    # list for the full one.
    work = copy.deepcopy(summary)
    shortened_lists: dict[str, tuple[int, int]] = {}

    def with_notice() -> ToolResult:
        if not shortened_lists:
            return build(work, True)
        notice = "Output limit: lists shown partially (do not treat them as complete): " + ", ".join(
            f"{k} {now} of {orig}" for k, (orig, now) in shortened_lists.items()
        )
        return ToolResult(
            status=result.status,
            artifact_id=result.artifact_id,
            summary=work,
            warnings=[*warnings, notice[:300]],
            truncated=True,
        )

    for _ in range(200):
        if len(with_notice().to_json()) <= max_chars:
            return with_notice()
        target = _largest_list(work)
        if target is not None:
            container, key, lst = target
            shortened = lst[: max(1, len(lst) // 2)]
            if container is None:
                break
            original = shortened_lists.get(str(key), (len(lst), 0))[0]
            shortened_lists[str(key)] = (original, len(shortened))
            container[key] = shortened
            continue
        text_target = _longest_string(work)
        if text_target is not None:
            container, key, text = text_target
            container[key] = text[: len(text) // 2] + "..."
            continue
        break

    # Last resort: keep only a JSON preview of the original summary.
    preview = _dumps(summary)
    warnings = warnings[:5]
    n = max_chars
    while n > 0:
        fallback = build({"_truncated_preview": preview[:n]}, True)
        if len(fallback.to_json()) <= max_chars:
            return fallback
        n = int(n * 0.8)
    return build({}, True)


# --------------------------------------------------------------------------- #
# Decorator                                                                   #
# --------------------------------------------------------------------------- #
class _ArgumentError(Exception):
    """Raised when call arguments do not match the tool signature."""


def _format_validation_error(exc: ValidationError, prefix: str = "Invalid arguments", show_input: bool = False) -> str:
    """Render pydantic errors in a short form the LLM can act on (no URLs, one clause per field).

    With ``show_input`` each clause also says what was sent (shortened).
    """
    parts = []
    for err in exc.errors()[:10]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        sent = ""
        if show_input and "input" in err:
            text = repr(err["input"])
            sent = f" (sent: {text if len(text) <= 80 else text[:77] + '...'})"
        parts.append(f"{loc}: {err['msg']}{sent}")
    return f"{prefix}: " + "; ".join(parts)


def _build_params_model(fn: Callable[..., Any], name: str, arg_docs: dict[str, str]) -> type[BaseModel]:
    """Create a pydantic model mirroring the signature of ``fn``."""
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for pname, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"Tool {name!r}: *args/**kwargs are not supported")
        annotation = hints.get(pname, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (annotation, Field(default, description=arg_docs.get(pname)))
    model_name = "".join(part.capitalize() for part in name.split("_")) + "Params"
    return create_model(model_name, __config__=ConfigDict(extra="forbid"), **fields)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop pydantic noise (titles) that only costs tokens."""
    schema = copy.deepcopy(schema)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


# After a timeout the worker gets this long to stop before the caller moves on; a
# reaper thread keeps interrupting it afterwards.
_STOP_GRACE_S = 2.0
_REAP_INTERVAL_S = 0.02
_REAP_MAX_S = 600.0


def _reap(scope: CallScope, thread: threading.Thread, wait_s: float) -> bool:
    """Interrupt the queries of a cancelled call until its thread ends, at most ``wait_s``.

    A tool may catch the interruption of one query and start the next: interrupting
    repeatedly stops those too. Returns whether the thread ended; its connections
    are closed then.
    """
    deadline = time.monotonic() + wait_s
    while thread.is_alive() and time.monotonic() < deadline:
        scope.cancel()
        thread.join(_REAP_INTERVAL_S)
    if thread.is_alive():
        return False
    scope.close_connections()
    scope.detach()
    return True


def _stop(scope: CallScope, thread: threading.Thread) -> None:
    """Cancel a call: stop its queries now, and keep at it in the background if it lingers."""
    scope.cancel()
    # Wait briefly only when there is a query to interrupt: the caller then gets its
    # error with the memory already freed. Pure-Python work gets no grace.
    if not _reap(scope, thread, _STOP_GRACE_S if scope.has_queries() else 0.0):
        # Pure-Python work cannot be interrupted: a daemon thread keeps trying, and
        # neither thread keeps the process alive at exit.
        threading.Thread(target=_reap, args=(scope, thread, _REAP_MAX_S), name="tool-reaper", daemon=True).start()


def _run_with_timeout(fn: Callable[..., Any], kwargs: dict[str, Any], timeout_s: float, name: str = "tool") -> Any:
    """Run ``fn`` in a worker thread, raising ``TimeoutError`` after ``timeout_s``.

    Python threads cannot be killed, so on timeout (or Ctrl+C) the call is
    cancelled instead: its DuckDB queries are interrupted, which ends the work of
    nearly every tool at once and frees its memory, and its connections are closed
    when the thread ends. The worker is a daemon thread: one that is stuck in
    pure-Python work never blocks the exit of the process.
    """
    scope = CallScope(parent=current_scope())
    ctx = contextvars.copy_context()
    ctx.run(set_scope, scope)
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = ctx.run(fn, **kwargs)
        except BaseException as exc:  # handed to the caller below
            box["error"] = exc

    thread = threading.Thread(target=work, name=f"tool-{name}", daemon=True)
    thread.start()
    try:
        thread.join(timeout_s)
    except BaseException:  # KeyboardInterrupt: stop the work before leaving
        _stop(scope, thread)
        raise
    if thread.is_alive():
        _stop(scope, thread)
        raise TimeoutError
    scope.detach()
    if "error" in box:
        raise box["error"]
    return box["value"]


run_with_timeout = _run_with_timeout  # for tools that give a part of their own work a deadline


def _error_message(result: ToolResult) -> str | None:
    """Message of an error result (for the trace), ``None`` for a successful one."""
    if result.status != "error":
        return None
    summary = result.summary
    return str(summary.get("error") or summary.get("_truncated_preview") or "error without a message")


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    max_output_chars: int | None = None,
    timeout_s: float | None = None,
) -> Any:
    """Turn a function into a Jellybean tool.

    Can be used bare (``@tool``) or with options
    (``@tool(max_output_chars=4000, timeout_s=300)``). Options left as ``None``
    fall back to ``config.yaml`` at call time.

    The decorated function keeps its name and can be called from Python with
    positional or keyword arguments; it always returns a :class:`ToolResult`.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., ToolResult]:
        tool_name = name or func.__name__
        description, arg_docs = parse_docstring(func.__doc__)
        params_model = _build_params_model(func, tool_name, arg_docs)
        signature = inspect.signature(func)
        takes_seed = "seed" in signature.parameters

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
            # Limits come from the active run's settings (or the global ones).
            active_run = maybe_current_run()
            settings = active_run.settings if active_run else get_settings()
            limit_chars = max_output_chars or settings.config.tool_output.max_output_chars
            limit_rows = settings.config.tool_output.max_raw_rows
            limits = settings.config.limits
            limit_time = limits.tool_timeouts.get(tool_name) or timeout_s or limits.tool_timeout_s
            started = time.perf_counter()
            call_args: dict[str, Any] = dict(kwargs)

            try:
                # 1) Bind and validate arguments before anything runs.
                try:
                    bound = signature.bind(*args, **kwargs)
                except TypeError as exc:
                    raise _ArgumentError(f"Invalid arguments: {exc}") from exc
                call_args = dict(bound.arguments)
                params = params_model.model_validate(call_args)
                call_kwargs = {k: getattr(params, k) for k in params_model.model_fields}
                # 2) Propagate the run seed to tools with randomness.
                if takes_seed and call_kwargs.get("seed") is None:
                    call_kwargs["seed"] = active_run.seed if active_run else settings.config.run.default_seed
                # 3) Execute with a timeout, remembering the tool for lineage.
                token = _current_tool.set(tool_name)
                try:
                    raw = _run_with_timeout(func, call_kwargs, limit_time, tool_name)
                finally:
                    _current_tool.reset(token)
                if isinstance(raw, ToolResult):
                    result = raw
                elif isinstance(raw, dict):
                    result = ToolResult.model_validate(raw)
                else:
                    result = ToolResult.error(f"Tool returned unsupported type {type(raw).__name__}")
            except _ArgumentError as exc:
                result = ToolResult.error(str(exc))
            except ValidationError as exc:
                result = ToolResult.error(_format_validation_error(exc))
            except TimeoutError:
                # Nothing the call wrote is kept (see ArtifactStore.create): it left no artifact.
                result = ToolResult.error(
                    f"Tool '{tool_name}' timed out after {limit_time:g}s and was stopped; it produced nothing. "
                    "Do not repeat the same call: narrow it (fewer columns, a filter, a smaller sample, "
                    "another method).",
                    timed_out=True,
                )
            except Exception as exc:
                # Unexpected: the message goes to the agent, the trace and the console;
                # the full traceback to logs/errors.log.
                log_exception(f"tool={tool_name}", exc)
                result = ToolResult.error(f"{type(exc).__name__}: {exc}")

            result = enforce_limits(result, limit_chars, limit_rows)
            trace_tool_call(
                tool=tool_name,
                arguments=call_args,
                status=result.status,
                output_chars=len(result.to_json()),
                raw_rows=result.raw_row_count,
                duration_s=time.perf_counter() - started,
                artifact_id=result.artifact_id,
                truncated=result.truncated,
                error=_error_message(result),
            )
            return result

        spec = ToolSpec(
            name=tool_name,
            description=description,
            input_schema=_clean_schema(params_model.model_json_schema()),
            params_model=params_model,
            max_output_chars=max_output_chars,
            timeout_s=timeout_s,
            func=wrapper,
        )
        setattr(wrapper, TOOL_SPEC_ATTR, spec)
        return wrapper

    return decorate(fn) if fn is not None else decorate


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
class ToolRegistryError(LookupError):
    """Raised for unknown tools or duplicate tool names."""


class ToolRegistry:
    """Index of shared tools and per-skill tools.

    Each agent sees only its own tools (from its ``tools.py``) plus the shared
    tools it declares in ``shared_tools``.
    """

    def __init__(self, shared_modules: Iterable[ModuleType]) -> None:
        self.shared: dict[str, ToolSpec] = {}
        for module in shared_modules:
            for spec in iter_module_tools(module):
                if spec.name in self.shared and self.shared[spec.name].func is not spec.func:
                    raise ToolRegistryError(f"Duplicate shared tool '{spec.name}'")
                self.shared[spec.name] = spec
        self._skills: dict[str, dict[str, ToolSpec]] = {}

    def register_skill(self, skill_name: str, module: ModuleType, shared_tools: Iterable[str]) -> None:
        """Register a skill's own tools and the shared tools it declared."""
        own = {spec.name: spec for spec in iter_module_tools(module)}
        tools: dict[str, ToolSpec] = {}
        for name in shared_tools:
            if name not in self.shared:
                raise ToolRegistryError(f"Skill '{skill_name}' declares unknown shared tool '{name}'")
            tools[name] = self.shared[name]
        for name, spec in own.items():
            # A skill module may re-export a shared tool it does not declare;
            # it is still only visible if it is defined or imported there.
            tools[name] = spec
        self._skills[skill_name] = tools

    def tools_for(self, skill_name: str) -> dict[str, ToolSpec]:
        if skill_name not in self._skills:
            raise ToolRegistryError(f"Unknown skill '{skill_name}'")
        return dict(self._skills[skill_name])
