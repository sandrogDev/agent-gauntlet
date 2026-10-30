"""The single interface every model backend implements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from gauntlet.models import AgentStep, Message, ToolSpec


@runtime_checkable
class Provider(Protocol):
    """A model that can take one step in a tool-calling conversation.

    Implementations must be stateless with respect to the conversation: the
    runner always passes the full history. That keeps scenarios independent and
    lets one provider instance serve many concurrent scenarios.
    """

    name: str

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> AgentStep: ...
