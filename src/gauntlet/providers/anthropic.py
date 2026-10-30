"""Anthropic Messages API provider (``anthropic:<model>``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from gauntlet.errors import ConfigError, ProviderError
from gauntlet.models import AgentStep, Message, ToolCall, ToolSpec, Usage

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Translate neutral messages to the Anthropic wire format.

    Tool results travel as ``tool_result`` blocks inside a *user* message, and
    consecutive user-side content is merged into one message so the
    alternating-role contract of the API is always respected.
    """
    out: list[dict[str, Any]] = []

    def push_user_block(block: dict[str, Any]) -> None:
        if out and out[-1]["role"] == "user":
            out[-1]["content"].append(block)
        else:
            out.append({"role": "user", "content": [block]})

    for m in messages:
        if m.role == "user":
            push_user_block({"type": "text", "text": m.content})
        elif m.role == "tool":
            push_user_block(
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                    "is_error": m.is_error,
                }
            )
        else:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in m.tool_calls
            )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
    return out


def to_anthropic_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
    ]


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ConfigError(
                "the 'anthropic' package is not installed: pip install 'agent-gauntlet[anthropic]'"
            ) from exc
        self._sdk = anthropic
        self._client = client or anthropic.AsyncAnthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = f"anthropic:{model}"

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> AgentStep:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": to_anthropic_messages(messages),
        }
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        try:
            response = await self._client.messages.create(**kwargs)
        except (self._sdk.APIConnectionError, self._sdk.APITimeoutError) as exc:
            raise ProviderError(f"anthropic connection error: {exc}", retryable=True) from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(
                f"anthropic API error {exc.status_code}: {exc.message}",
                retryable=exc.status_code in RETRYABLE_STATUS,
            ) from exc

        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}))
            for b in response.content
            if b.type == "tool_use"
        ]
        usage = Usage(
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
        )
        return AgentStep(text=text, tool_calls=calls, usage=usage)
