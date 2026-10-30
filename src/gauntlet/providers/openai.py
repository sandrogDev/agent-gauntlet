"""OpenAI-compatible Chat Completions provider (``openai:<model>``).

Works with any server that speaks the Chat Completions tool-calling protocol
(vLLM, Ollama, LiteLLM, Together, ...) via the standard ``OPENAI_BASE_URL``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from gauntlet.errors import ConfigError, ProviderError
from gauntlet.models import AgentStep, Message, ToolCall, ToolSpec, Usage
from gauntlet.providers.anthropic import RETRYABLE_STATUS


def to_openai_messages(system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        else:
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in m.tool_calls
                ]
            out.append(msg)
    return out


def to_openai_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Models occasionally emit malformed JSON; surface it instead of crashing.

    A non-object payload is wrapped so the schema check flags it as invalid.
    """
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"__unparseable_arguments__": raw}
    return value if isinstance(value, dict) else {"__non_object_arguments__": value}


class OpenAIProvider:
    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ConfigError(
                "the 'openai' package is not installed: pip install 'agent-gauntlet[openai]'"
            ) from exc
        self._sdk = openai
        self._client = client or openai.AsyncOpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = f"openai:{model}"

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
            "messages": to_openai_messages(system, messages),
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except (self._sdk.APIConnectionError, self._sdk.APITimeoutError) as exc:
            raise ProviderError(f"openai connection error: {exc}", retryable=True) from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(
                f"openai API error {exc.status_code}: {exc.message}",
                retryable=exc.status_code in RETRYABLE_STATUS,
            ) from exc

        message = response.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=parse_arguments(c.function.arguments))
            for c in (message.tool_calls or [])
        ]
        usage = Usage()
        if response.usage is not None:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return AgentStep(text=message.content or "", tool_calls=calls, usage=usage)
