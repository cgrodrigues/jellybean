# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Agent runtime: the tool-use loop, delegation, token budget (spec section 4).

* :class:`Runtime` holds what is shared by every agent of a run: skills, tool
  registry, LLM client, human interface and token budget. It is made available
  to tools (e.g. ``delegate``, ``ask_human``) through :func:`current_runtime`.
* :class:`AgentRunner` runs one agent: it starts from a **clean context** (system
  prompt + spec summary + task), loops over tool calls until the agent calls
  ``finish`` or hits ``max_iterations``, and returns an :class:`AgentResult`.

Between agents only artifact ids and short text travel, never data or
conversation history.
"""

from __future__ import annotations

import contextvars
import importlib
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from jellybean.runtime.context import RunContext, current_run, use_run
from jellybean.runtime.human import AutoApproveHuman, HumanInterface
from jellybean.runtime.llm_client import LLMClient, Message
from jellybean.runtime.skill_loader import SkillDefinition, discover_skills
from jellybean.runtime.tool_registry import ToolRegistry, ToolResult, ToolSpec, get_tool_spec, tool
from jellybean.runtime.tracing import log_exception, safe_error_text, trace_event, trace_tool_call

AgentStatus = Literal["done", "needs_rework", "needs_human", "failed"]
FINISH_TOOL = "finish"


class AgentResult(BaseModel):
    """Structured outcome of one agent execution."""

    agent: str
    status: AgentStatus
    summary: str
    artifact_ids: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    requests: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Artifacts (datasets, documents, plots) this agent created: id, type, rows, path.
    outputs: list[dict[str, Any]] = Field(default_factory=list)


def run_outputs(ctx: RunContext, agent: str | None = None, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    """Artifacts of a run (optionally of one agent, excluding some ids), with paths relative to the run folder."""
    from jellybean.skills._shared.artifacts import ArtifactStore

    if not ctx.manifest_path.exists():
        return []
    return [
        {
            "id": r.id,
            "type": r.type,
            "rows": r.row_count,
            "path": r.path,
            "agent": r.created_by.agent,
            "tool": r.created_by.tool,
        }
        for r in ArtifactStore(ctx).list()
        if (agent is None or r.created_by.agent == agent) and r.id not in (exclude or set())
    ]


@tool
def finish(
    status: AgentStatus,
    summary: str,
    artifact_ids: list[str] | None = None,
    issues: list[str] | None = None,
    requests: list[dict[str, Any]] | None = None,
) -> ToolResult:
    """End your work and report back. Call it exactly once, as your last action.

    Args:
        status: done | needs_rework | needs_human | failed.
        summary: Short summary of what you did and found (no raw data).
        artifact_ids: Artifacts you produced that the caller should use.
        issues: Problems or caveats the caller must know about.
        requests: Requests for other agents, e.g. [{"agent": "preparer", "reason": "..."}].
    """
    # The runner intercepts this tool; the body only exists for the schema.
    return ToolResult.ok({"accepted": True})


FINISH_SPEC: ToolSpec = get_tool_spec(finish)  # type: ignore[assignment]

RUNTIME_RULES = """

## Runtime rules (apply to every agent)
- You act only through tools. Data never enters this conversation: tools return
  compact summaries and artifact ids. Do not ask for raw data you do not need.
- Never compute numbers yourself; every number must come from a tool result.
- Tool errors are returned as results with status "error": read the message and
  adapt instead of repeating the same call.
- When your work is complete, call `finish` exactly once with a short summary,
  the relevant artifact ids, any issues, and requests for other agents.
"""


def budget_tokens(response: Any) -> int:
    """Tokens charged to the run budget: cache reads count at 10% (their relative price)."""
    cached = getattr(response, "cache_read_tokens", 0) or 0
    return int(response.input_tokens - cached + 0.1 * cached + response.output_tokens)


class TokenBudget:
    """Run-wide token budget shared by all agents (thread-safe)."""

    def __init__(self, limit: int, used: int = 0) -> None:
        self.limit = limit
        self.used = used
        self._lock = threading.Lock()

    def add(self, tokens: int) -> None:
        with self._lock:
            self.used += tokens

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit


_current_runtime: contextvars.ContextVar[Runtime | None] = contextvars.ContextVar(
    "jellybean_current_runtime", default=None
)


def current_runtime() -> Runtime:
    rt = _current_runtime.get()
    if rt is None:
        raise RuntimeError("No active runtime (tools that delegate or ask humans need one).")
    return rt


@dataclass
class Runtime:
    """Everything shared by the agents of one run."""

    ctx: RunContext
    llm: LLMClient
    human: HumanInterface = field(default_factory=AutoApproveHuman)
    skills: dict[str, SkillDefinition] = field(default_factory=discover_skills)
    options: dict[str, Any] = field(default_factory=dict)
    # Control-point questions prepared by tools, keyed by topic: ask_human must
    # use them unchanged (the LLM cannot alter options or the recommendation).
    pending_questions: dict[str, dict[str, Any]] = field(default_factory=dict)
    registry: ToolRegistry = field(init=False)
    budget: TokenBudget = field(init=False)

    def __post_init__(self) -> None:
        from jellybean.skills._shared import data_tools, findings, spec_tools

        self.registry = ToolRegistry([data_tools, spec_tools, findings])
        for skill in self.skills.values():
            self.registry.register_skill(skill.name, importlib.import_module(skill.module), skill.shared_tools)
        self.budget = TokenBudget(self.ctx.settings.config.limits.token_budget_per_run)

    @contextmanager
    def active(self) -> Iterator[Runtime]:
        """Activate this runtime and its run context."""
        token = _current_runtime.set(self)
        try:
            with use_run(self.ctx):
                yield self
        finally:
            _current_runtime.reset(token)

    def run_agent(self, agent: str, task: str, artifact_ids: list[str] | None = None) -> AgentResult:
        """Run one agent with a fresh context and return its result."""
        if agent not in self.skills:
            raise KeyError(f"Unknown agent '{agent}'. Available: {sorted(self.skills)}")
        with self.active():
            return AgentRunner(self, self.skills[agent]).run(task, artifact_ids or [])


class AgentRunner:
    """Runs a single agent until it finishes, fails or hits its limits."""

    def __init__(self, runtime: Runtime, skill: SkillDefinition) -> None:
        self.runtime = runtime
        self.skill = skill
        limits = runtime.ctx.settings.config.limits
        self.max_iterations = skill.max_iterations or limits.max_iterations_per_agent
        self.tools = runtime.registry.tools_for(skill.name)
        self.tools[FINISH_TOOL] = FINISH_SPEC
        # Calls (tool + arguments) that timed out in this agent run: never executed again.
        self._timed_out: set[tuple[str, str]] = set()

    def _system_prompt(self) -> str:
        return self.skill.system_prompt.rstrip() + RUNTIME_RULES

    def _first_message(self, task: str, artifact_ids: list[str]) -> str:
        ctx = current_run()
        spec_text = "No spec.yaml yet."
        if ctx.spec_path.exists():
            from jellybean.spec import load_spec

            spec_text = json.dumps(load_spec(ctx.spec_path).compact_summary(), ensure_ascii=False, default=str)
        ids = ", ".join(artifact_ids) if artifact_ids else "none"
        text = f"## Current spec (summary)\n{spec_text}\n\n## Task\n{task}\n\n## Artifacts\n{ids}"
        if "delegate" in self.tools:
            agents = [f"- {s.name}: {s.description}" for s in self.runtime.skills.values() if s.name != self.skill.name]
            text += "\n\n## Agents you can delegate to\n" + ("\n".join(agents) or "none")
        return text

    def _execute(self, name: str, arguments: Any) -> ToolResult:
        spec = self.tools.get(name)
        if spec is None:
            return self._rejected(name, {}, f"Unknown tool '{name}'. Available: {sorted(self.tools)}")
        if not isinstance(arguments, dict):
            return self._rejected(name, {}, "Tool arguments must be a JSON object.")
        key = (name, json.dumps(arguments, sort_keys=True, default=str))
        if key in self._timed_out:
            return self._rejected(
                name,
                arguments,
                f"Not run: this exact call to '{name}' already timed out in this task and would time out again. "
                "Change the arguments (a smaller sample or fewer rows through the tool's parameters, fewer "
                "columns, another method), or continue without it and report it as not run.",
            )
        result = spec.func(**arguments)
        if result.status == "error" and result.summary.get("timed_out"):
            self._timed_out.add(key)
        return result

    @staticmethod
    def _rejected(name: str, arguments: dict[str, Any], message: str) -> ToolResult:
        """Error result for a call that never reached a tool; traced so it is not silent."""
        result = ToolResult.error(message)
        trace_tool_call(name, arguments, "error", len(result.to_json()), 0, 0.0, None, False, error=message)
        return result

    def run(self, task: str, artifact_ids: list[str]) -> AgentResult:
        ctx = current_run().for_agent(self.skill.name)
        before = {o["id"] for o in run_outputs(ctx)}
        with use_run(ctx):
            result = self._loop(task, artifact_ids)
        result.outputs = [
            {k: v for k, v in o.items() if k != "agent"}
            for o in run_outputs(ctx, agent=self.skill.name, exclude=before)
        ]
        return result

    def _result(
        self, status: AgentStatus, summary: str, iterations: int, tokens: tuple[int, int], **extra: Any
    ) -> AgentResult:
        result = AgentResult(
            agent=self.skill.name,
            status=status,
            summary=summary,
            iterations=iterations,
            input_tokens=tokens[0],
            output_tokens=tokens[1],
            **extra,
        )
        # A failure always says why (shown in the console): runtime reasons are short codes
        # and messages, never data.
        why = {"reason": safe_error_text(summary), "issues": result.issues} if status == "failed" else {}
        trace_event(
            "agent_end", status=status, iterations=iterations, input_tokens=tokens[0], output_tokens=tokens[1], **why
        )
        return result

    def _loop(self, task: str, artifact_ids: list[str]) -> AgentResult:
        system = self._system_prompt()
        messages: list[Message] = [Message(role="user", content=self._first_message(task, artifact_ids))]
        specs = list(self.tools.values())
        tokens_in = tokens_out = 0
        nudged = False
        trace_event(
            "agent_start", task_chars=len(task), task=task[:160], artifact_ids=artifact_ids, tools=sorted(self.tools)
        )

        for iteration in range(1, self.max_iterations + 1):
            if self.runtime.budget.exhausted:
                return self._result(
                    "failed",
                    "Token budget for the run is exhausted; stopped cleanly.",
                    iteration - 1,
                    (tokens_in, tokens_out),
                    issues=["token_budget_exhausted"],
                )
            started = time.perf_counter()
            try:
                response = self.runtime.llm.complete(self.skill.name, system, messages, specs)
            except Exception as exc:
                log_exception("llm_call", exc)
                trace_event("llm_error", error=safe_error_text(f"{type(exc).__name__}: {exc}"))
                return self._result(
                    "failed",
                    f"LLM call failed: {type(exc).__name__}: {exc}"[:500],
                    iteration - 1,
                    (tokens_in, tokens_out),
                    issues=["llm_error"],
                )
            tokens_in += response.input_tokens
            tokens_out += response.output_tokens
            self.runtime.budget.add(budget_tokens(response))
            trace_event(
                "llm_call",
                model=self.runtime.llm.model_name,
                iteration=iteration,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                duration_s=round(time.perf_counter() - started, 3),
                tool_calls=len(response.tool_calls),
                stop_reason=response.stop_reason,
                cache_read_tokens=response.cache_read_tokens,
                cache_write_tokens=response.cache_write_tokens,
            )
            messages.append(response.as_message())

            if not response.tool_calls:
                if nudged:
                    return self._result(
                        "done",
                        response.text[:2000] or "(no summary)",
                        iteration,
                        (tokens_in, tokens_out),
                        issues=["agent ended without calling finish"],
                    )
                nudged = True
                messages.append(Message(role="user", content="Call the `finish` tool to report your result."))
                continue

            finish_args: dict[str, Any] | None = None
            finish_index = -1
            for call in response.tool_calls:
                if call.name == FINISH_TOOL:
                    finish_args = call.arguments if isinstance(call.arguments, dict) else {}
                    finish_index = len(messages)
                    content = json.dumps({"status": "ok", "summary": {"accepted": True}})
                else:
                    content = self._execute(call.name, call.arguments).to_json()
                messages.append(Message(role="tool", content=content, tool_call_id=call.id, name=call.name))

            if finish_args is not None:
                try:
                    args = FINISH_SPEC.params_model.model_validate(finish_args)
                except ValidationError as exc:
                    # Replace the optimistic acknowledgement with the validation error.
                    error = self._rejected(FINISH_TOOL, {}, f"Invalid finish arguments: {exc}").to_json()
                    messages[finish_index] = Message(
                        role="tool", content=error, tool_call_id=messages[finish_index].tool_call_id, name=FINISH_TOOL
                    )
                    continue
                return self._result(
                    args.status,
                    args.summary,
                    iteration,
                    (tokens_in, tokens_out),
                    artifact_ids=args.artifact_ids or [],
                    issues=args.issues or [],
                    requests=args.requests or [],
                )

        return self._result(
            "failed",
            f"Stopped after max_iterations={self.max_iterations} without finishing.",
            self.max_iterations,
            (tokens_in, tokens_out),
            issues=["max_iterations"],
        )
