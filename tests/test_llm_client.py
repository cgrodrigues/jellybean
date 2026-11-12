"""LLMClient tests: message conversion, provider construction, FakeLLM. No network calls."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import SecretStr

from jellybean.runtime.agent_runner import FINISH_SPEC
from jellybean.runtime.llm_client import (
    FakeLLM,
    LangChainLLMClient,
    LLMConfigurationError,
    Message,
    ToolCall,
    _from_langchain,
    _to_langchain,
    build_chat_model,
    call,
    tool_definition,
)
from jellybean.settings import AppConfig, EnvSettings, Settings


def _settings(**env) -> Settings:
    return Settings(env=EnvSettings(**env), config=AppConfig())


def test_neutral_messages_convert_to_langchain() -> None:
    msgs = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="", tool_calls=(ToolCall("t1", "get_spec", {}),)),
        Message(role="tool", content='{"status":"ok"}', tool_call_id="t1", name="get_spec"),
    ]
    converted = _to_langchain("system prompt", msgs)
    assert [type(m) for m in converted] == [SystemMessage, HumanMessage, AIMessage, ToolMessage]
    assert converted[2].tool_calls[0]["name"] == "get_spec"
    assert converted[3].tool_call_id == "t1"


def test_langchain_reply_converts_back() -> None:
    reply = AIMessage(
        content=[{"type": "text", "text": "let me check"}],
        tool_calls=[{"name": "list_datasets", "args": {}, "id": "abc"}],
        usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        response_metadata={"stop_reason": "tool_use"},
    )
    response = _from_langchain(reply)
    assert response.text == "let me check"
    assert response.tool_calls == (ToolCall("abc", "list_datasets", {}),)
    assert (response.input_tokens, response.output_tokens, response.stop_reason) == (11, 7, "tool_use")


def test_tool_definition_shape() -> None:
    definition = tool_definition(FINISH_SPEC)
    assert definition["function"]["name"] == "finish"
    assert "status" in definition["function"]["parameters"]["properties"]


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"provider": "anthropic", "model": None, "api_key": SecretStr("k")}, "JELLYBEAN_MODEL"),
        ({"provider": "anthropic", "model": "m", "api_key": None}, "JELLYBEAN_API_KEY"),
        ({"provider": "openai", "model": "m", "api_key": None}, "JELLYBEAN_API_KEY"),
    ],
)
def test_configuration_errors(env: dict, message: str) -> None:
    with pytest.raises(LLMConfigurationError, match=message):
        build_chat_model(_settings(**env))


def test_builds_each_provider_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_anthropic import ChatAnthropic
    from langchain_aws import ChatBedrockConverse
    from langchain_openai import ChatOpenAI

    assert isinstance(
        build_chat_model(_settings(provider="anthropic", model="m", api_key=SecretStr("k"))), ChatAnthropic
    )
    assert isinstance(build_chat_model(_settings(provider="openai", model="m", api_key=SecretStr("k"))), ChatOpenAI)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    bedrock = build_chat_model(_settings(provider="bedrock", model="some.model-id", api_key=None))
    assert isinstance(bedrock, ChatBedrockConverse) and bedrock.region_name == "eu-west-1"


def test_langchain_client_binds_tools_once() -> None:
    class StubChat:
        def __init__(self) -> None:
            self.bound = 0

        def bind_tools(self, tools):
            self.bound += 1
            self.tools = tools
            return self

        def invoke(self, messages, **kwargs):
            self.kwargs = kwargs
            return AIMessage(content="done", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    stub = StubChat()
    client = LangChainLLMClient(_settings(model="m"), chat_model=stub)
    for _ in range(3):
        assert client.complete("a", "sys", [Message(role="user", content="x")], [FINISH_SPEC]).text == "done"
    assert stub.bound == 1 and stub.tools[0]["function"]["name"] == "finish"
    assert stub.kwargs == {"cache_control": {"type": "ephemeral"}}  # prompt caching (anthropic default)


def test_fake_llm_records_and_runs_out() -> None:
    llm = FakeLLM({"a": [call("finish", status="done", summary="s")]})
    response = llm.complete("a", "sys", [], [])
    assert response.tool_calls[0].name == "finish" and response.input_tokens == 100
    with pytest.raises(AssertionError, match="no scripted response"):
        llm.complete("a", "sys", [], [])
