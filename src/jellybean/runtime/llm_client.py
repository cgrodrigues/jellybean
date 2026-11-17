# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""LLM client interface, the LangChain implementation and a scripted FakeLLM.

The runtime talks to models only through :class:`LLMClient`, using small
provider-neutral message types. :class:`LangChainLLMClient` maps them onto
LangChain chat models for the three supported providers:

* ``anthropic`` -> ``langchain_anthropic.ChatAnthropic``
* ``openai``    -> ``langchain_openai.ChatOpenAI``
* ``bedrock``   -> ``langchain_aws.ChatBedrockConverse`` (standard AWS credential chain)

Provider, model and key come from the environment (``.env``); nothing is
hard-coded. :class:`FakeLLM` replays pre-programmed responses so the runtime
can be tested without network calls.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from jellybean.runtime.tool_registry import ToolSpec
from jellybean.settings import Settings


# --------------------------------------------------------------------------- #
# Provider-neutral message model                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """One conversation turn.

    ``role="assistant"`` messages may carry tool calls; ``role="tool"``
    messages carry the result of one tool call (``tool_call_id``).
    """

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    """What the model returned for one call."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    # Part of input_tokens read from / written to the prompt cache.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def as_message(self) -> Message:
        return Message(role="assistant", content=self.text, tool_calls=self.tool_calls)


class LLMClient(Protocol):
    """Interface used by the AgentRunner."""

    model_name: str

    def complete(self, agent: str, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        """Return the next assistant turn for ``messages`` with ``tools`` available."""
        ...


class LLMConfigurationError(RuntimeError):
    """Raised when provider/model/credentials are missing or invalid."""


def tool_definition(spec: ToolSpec) -> dict[str, Any]:
    """OpenAI-style function definition; LangChain converts it for each provider."""
    return {
        "type": "function",
        "function": {"name": spec.name, "description": spec.description, "parameters": spec.input_schema},
    }


# --------------------------------------------------------------------------- #
# LangChain implementation                                                    #
# --------------------------------------------------------------------------- #
def _to_langchain(system: str, messages: Sequence[Message]) -> list[Any]:
    """Convert neutral messages into LangChain message objects."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    converted: list[Any] = [SystemMessage(content=system)]
    for m in messages:
        if m.role == "user":
            converted.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            converted.append(
                AIMessage(
                    content=m.content,
                    tool_calls=[{"name": c.name, "args": c.arguments, "id": c.id} for c in m.tool_calls],
                )
            )
        else:
            converted.append(ToolMessage(content=m.content, tool_call_id=m.tool_call_id or "", name=m.name))
    return converted


def _from_langchain(message: Any) -> LLMResponse:
    """Convert a LangChain ``AIMessage`` into an :class:`LLMResponse`."""
    usage = getattr(message, "usage_metadata", None) or {}
    calls = tuple(
        ToolCall(id=c.get("id") or f"call_{i}", name=c["name"], arguments=dict(c.get("args") or {}))
        for i, c in enumerate(getattr(message, "tool_calls", None) or [])
    )
    details = usage.get("input_token_details") or {}
    metadata = getattr(message, "response_metadata", None) or {}
    stop = metadata.get("stop_reason") or metadata.get("finish_reason") or metadata.get("stopReason")
    return LLMResponse(
        text=message.text or "",
        tool_calls=calls,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        stop_reason=stop,
        cache_read_tokens=int(details.get("cache_read") or 0),
        cache_write_tokens=int(details.get("cache_creation") or 0),
    )


def build_chat_model(settings: Settings) -> Any:
    """Instantiate the LangChain chat model selected in the environment."""
    env, llm = settings.env, settings.config.llm
    if not env.model:
        raise LLMConfigurationError("JELLYBEAN_MODEL is not set (see .env.example).")
    common: dict[str, Any] = {"model": env.model, "max_tokens": llm.max_tokens, "max_retries": llm.max_retries}
    if llm.temperature is not None:
        common["temperature"] = llm.temperature
    key = env.api_key.get_secret_value() if env.api_key else None

    if env.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not key:
            raise LLMConfigurationError("JELLYBEAN_API_KEY is required for provider 'anthropic'.")
        return ChatAnthropic(api_key=key, timeout=llm.request_timeout_s, **common)
    if env.provider == "openai":
        from langchain_openai import ChatOpenAI

        if not key:
            raise LLMConfigurationError("JELLYBEAN_API_KEY is required for provider 'openai'.")
        return ChatOpenAI(api_key=key, timeout=llm.request_timeout_s, **common)
    if env.provider == "bedrock":
        # Credentials come from the standard AWS chain (profile, env vars,
        # AWS_BEARER_TOKEN_BEDROCK, SSO or instance role).
        import os

        from langchain_aws import ChatBedrockConverse

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        return ChatBedrockConverse(region_name=region, **common)
    raise LLMConfigurationError(f"Unsupported provider {env.provider!r}")


class LangChainLLMClient:
    """:class:`LLMClient` backed by a LangChain chat model."""

    def __init__(self, settings: Settings, chat_model: Any | None = None) -> None:
        self.settings = settings
        self.chat_model = chat_model or build_chat_model(settings)
        self.model_name = settings.env.model or "unknown"
        # Bound models are cached per tool set (agents reuse the same tools).
        self._bound: dict[tuple[str, ...], Any] = {}

    def _bound_model(self, tools: Sequence[ToolSpec]) -> Any:
        key = tuple(t.name for t in tools)
        if key not in self._bound:
            self._bound[key] = (
                self.chat_model.bind_tools([tool_definition(t) for t in tools]) if tools else (self.chat_model)
            )
        return self._bound[key]

    def _invoke_kwargs(self) -> dict[str, Any]:
        """Provider-specific call options (prompt caching for anthropic and bedrock)."""
        if self.settings.config.llm.prompt_caching and self.settings.env.provider in ("anthropic", "bedrock"):
            # ChatAnthropic sends it as automatic caching; ChatBedrockConverse turns it
            # into cachePoint blocks on the system prompt, tools and last message.
            return {"cache_control": {"type": "ephemeral"}}
        return {}

    def complete(self, agent: str, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        reply = self._bound_model(tools).invoke(_to_langchain(system, messages), **self._invoke_kwargs())
        # print(f"LLMClient.complete: {agent=} {system=} {len(messages)=} {len(tools)=} -> {len(reply.text)} chars")
        return _from_langchain(reply)


def build_llm_client(settings: Settings) -> LLMClient:
    """Factory used by the CLI."""
    return LangChainLLMClient(settings)


# --------------------------------------------------------------------------- #
# FakeLLM                                                                     #
# --------------------------------------------------------------------------- #
# A scripted step is either a ready response or a function of the conversation.
Step = LLMResponse | Callable[[Sequence[Message]], LLMResponse]


@dataclass
class RecordedCall:
    """What the FakeLLM received (used to assert context isolation)."""

    agent: str
    system: str
    messages: list[Message]
    tool_names: list[str]


@dataclass
class FakeLLM:
    """LLM double with pre-programmed responses per agent.

    Args:
        script: ``{agent_name: [step, step, ...]}``. Each call by an agent
            consumes its next step. A step may be a callable receiving the
            conversation so far (e.g. to reuse an artifact id from a tool result).
        tokens_per_call: Token usage reported for each call (input, output).
    """

    script: dict[str, list[Step]]
    tokens_per_call: tuple[int, int] = (100, 20)
    model_name: str = "fake-llm"
    calls: list[RecordedCall] = field(default_factory=list)

    def complete(self, agent: str, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        self.calls.append(RecordedCall(agent, system, list(messages), [t.name for t in tools]))
        steps = self.script.get(agent)
        if not steps:
            raise AssertionError(f"FakeLLM has no scripted response left for agent '{agent}'")
        step = steps.pop(0)
        response = step(messages) if callable(step) else step
        return LLMResponse(
            text=response.text,
            tool_calls=response.tool_calls,
            input_tokens=response.input_tokens or self.tokens_per_call[0],
            output_tokens=response.output_tokens or self.tokens_per_call[1],
            stop_reason=response.stop_reason,
        )


_CALL_IDS = itertools.count(1)


def call(tool_name: str, /, **arguments: Any) -> LLMResponse:
    """Helper for scripts: a response that calls one tool (positional-only name, so
    tools with a ``name`` argument work)."""
    return LLMResponse(text="", tool_calls=(ToolCall(f"call_{next(_CALL_IDS)}", tool_name, arguments),))


def last_tool_result(messages: Sequence[Message]) -> dict[str, Any]:
    """Helper for scripts: the parsed JSON of the latest tool result."""
    for m in reversed(messages):
        if m.role == "tool":
            return json.loads(m.content)
    raise AssertionError("no tool result in conversation")
