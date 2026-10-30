"""Shared fixtures.

Async tests run through a tiny hook instead of pytest-asyncio, keeping the dev
dependency set minimal.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

from gauntlet import AgentStep, Message, ProviderError, ToolCall, ToolError, ToolRegistry, ToolSpec
from gauntlet.models import Scenario, Suite

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return None


# --------------------------------------------------------------------------- #
# Scripted provider
# --------------------------------------------------------------------------- #

Step = AgentStep | Exception | Callable[[Sequence[Message]], AgentStep]


def call(name: str, **arguments: Any) -> AgentStep:
    """Shorthand for a step that makes a single tool call."""
    return AgentStep(
        tool_calls=[ToolCall(id=f"{name}-{next(_ids)}", name=name, arguments=arguments)]
    )


def say(text: str) -> AgentStep:
    return AgentStep(text=text)


_ids = itertools.count(1)


class ScriptedProvider:
    """Replays a fixed list of steps; records every request it receives."""

    def __init__(self, steps: Iterable[Step], name: str = "scripted") -> None:
        self._steps = list(steps)
        self.name = name
        self.requests: list[list[Message]] = []

    async def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> AgentStep:
        self.requests.append(list(messages))
        if not self._steps:
            return say("(script exhausted)")
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(messages)
        return step


class FlakyProvider:
    """Fails ``failures`` times with the given error, then succeeds."""

    def __init__(self, failures: int, error: Exception | None = None) -> None:
        self.failures = failures
        self.error = error or ProviderError("rate limited", retryable=True)
        self.calls = 0
        self.name = "flaky"

    async def complete(self, **_: Any) -> AgentStep:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return say("ok")


# --------------------------------------------------------------------------- #
# Tiny tool world
# --------------------------------------------------------------------------- #


def build_mini_registry() -> ToolRegistry:
    reg = ToolRegistry()
    state = {"orders": {"A1": {"status": "open", "total": 40.0}}}

    @reg.tool(
        description="Fetch an order",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string", "pattern": "^[A-Z][0-9]$"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    )
    def get_order(order_id: str) -> dict[str, Any]:
        if order_id not in state["orders"]:
            raise ToolError("order not found")
        return {"order_id": order_id, **state["orders"][order_id]}

    @reg.tool(
        description="Cancel an order",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        side_effect=True,
    )
    async def cancel_order(order_id: str) -> dict[str, Any]:
        state["orders"][order_id]["status"] = "cancelled"
        return {"order_id": order_id, "status": "cancelled"}

    @reg.tool(description="Always crashes", parameters={"type": "object"})
    def explode() -> None:
        raise RuntimeError("boom")

    @reg.rule("cancel_after_lookup")
    def cancel_after_lookup(trace: Any) -> tuple[bool, str]:
        names = [r.call.name for r in trace.tool_calls if r.succeeded]
        if "cancel_order" in names and "get_order" not in names[: names.index("cancel_order")]:
            return False, "cancelled without lookup"
        return True, "ok"

    return reg


@pytest.fixture
def registry_factory() -> Callable[[], ToolRegistry]:
    return build_mini_registry


def make_suite(scenarios: list[Scenario] | None = None, **overrides: Any) -> Suite:
    data: dict[str, Any] = {
        "name": "mini",
        "system_prompt": "You are a shop assistant.",
        "tools_module": "unused.py",
        "max_steps": 4,
        "scenarios": scenarios or [Scenario(id="s1", turns=["cancel order A1"])],
    }
    data.update(overrides)
    return Suite.model_validate(data)
