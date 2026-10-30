"""Wire-format translation is tested without the SDKs installed."""

from __future__ import annotations

import pytest

from gauntlet import ConfigError
from gauntlet.models import Message, ToolCall, ToolSpec
from gauntlet.providers import create_provider
from gauntlet.providers.anthropic import to_anthropic_messages, to_anthropic_tools
from gauntlet.providers.openai import parse_arguments, to_openai_messages, to_openai_tools

CONVERSATION = [
    Message(role="user", content="cancel A1"),
    Message(
        role="assistant",
        content="Checking.",
        tool_calls=[
            ToolCall(id="t1", name="get_order", arguments={"order_id": "A1"}),
            ToolCall(id="t2", name="get_user", arguments={}),
        ],
    ),
    Message(role="tool", tool_call_id="t1", content='{"status": "open"}'),
    Message(role="tool", tool_call_id="t2", content='{"error": "x"}', is_error=True),
    Message(role="user", content="thanks"),
]
TOOLS = [ToolSpec(name="get_order", description="d", parameters={"type": "object"})]


def test_anthropic_merges_tool_results_and_follow_up_into_one_user_turn() -> None:
    out = to_anthropic_messages(CONVERSATION)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    merged = out[2]["content"]
    assert [b["type"] for b in merged] == ["tool_result", "tool_result", "text"]
    assert merged[1]["is_error"] is True and merged[0]["tool_use_id"] == "t1"
    assert to_anthropic_tools(TOOLS)[0]["input_schema"] == {"type": "object"}


def test_anthropic_skips_empty_assistant_messages() -> None:
    out = to_anthropic_messages([Message(role="user", content="a"), Message(role="assistant")])
    assert len(out) == 1


def test_openai_format() -> None:
    out = to_openai_messages("sys", CONVERSATION)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool", "user"]
    assert out[2]["tool_calls"][0]["function"]["arguments"] == '{"order_id": "A1"}'
    assert out[3]["tool_call_id"] == "t1"
    assert to_openai_tools(TOOLS)[0]["function"]["name"] == "get_order"


def test_openai_argument_parsing_is_defensive() -> None:
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("") == {}
    assert "__unparseable_arguments__" in parse_arguments("{not json")
    assert "__non_object_arguments__" in parse_arguments("[1, 2]")


@pytest.mark.parametrize("spec", ["nonsense", "foo:bar", "python:"])
def test_bad_provider_specs(spec: str) -> None:
    with pytest.raises(ConfigError):
        create_provider(spec)


def test_python_provider_loading(tmp_path) -> None:
    agent = tmp_path / "agent.py"
    agent.write_text(
        "from gauntlet import AgentStep\n"
        "class A:\n"
        "    name = 'mine'\n"
        "    async def complete(self, *, system, messages, tools):\n"
        "        return AgentStep(text='hi')\n"
        "def make():\n"
        "    return A()\n"
        "def create_provider():\n"
        "    return A()\n"
    )
    assert create_provider(f"python:{agent}").name == "mine"
    assert create_provider(f"python:{agent}:make").name == "mine"
    with pytest.raises(ConfigError):
        create_provider(f"python:{agent}:missing")
