"""Deterministic built-in checks.

Each check is a pure function of the trace, so results are reproducible and a
stored run can be re-scored after a check is fixed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from gauntlet.checks.base import CheckContext, CheckOutcome, register_check, require
from gauntlet.models import ToolCallRecord


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _args_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Subset match. Strings compare case-insensitively, numbers by value."""
    for key, want in expected.items():
        if key not in actual:
            return False
        got = actual[key]
        if isinstance(want, str) and isinstance(got, str):
            if want.casefold() != got.casefold():
                return False
        elif got != want:
            return False
    return True


def _responses(ctx: CheckContext, scope: str) -> list[str]:
    finals = ctx.trace.final_responses
    if scope == "last":
        return finals[-1:]
    if scope in ("any", "all"):
        return list(finals)
    raise ValueError(f"scope must be 'last', 'any' or 'all', got {scope!r}")


# --------------------------------------------------------------------------- #
# Structural checks
# --------------------------------------------------------------------------- #


@register_check("schema_valid")
async def schema_valid(ctx: CheckContext, params: dict[str, Any]) -> CheckOutcome:
    """Every tool call names a real tool and satisfies its JSON Schema."""
    calls = ctx.trace.tool_calls
    bad = [r for r in calls if not r.valid]
    if not bad:
        return CheckOutcome(True, f"{len(calls)} tool call(s), all schema-valid")
    details = "; ".join(
        f"{r.call.name}: {'unknown tool' if r.unknown_tool else ' | '.join(r.schema_errors)}"
        for r in bad[:3]
    )
    return CheckOutcome(False, f"{len(bad)}/{len(calls)} invalid tool call(s): {details}")


@register_check("completed")
async def completed(ctx: CheckContext, params: dict[str, Any]) -> CheckOutcome:
    """The run finished every turn without a provider error or step-limit hit."""
    t = ctx.trace
    if t.error:
        return CheckOutcome(False, f"run aborted: {t.error}")
    if t.hit_step_limit:
        return CheckOutcome(False, "agent hit the step limit without producing an answer")
    return CheckOutcome(True, f"completed {len(t.final_responses)} turn(s) in {t.steps} step(s)")


# --------------------------------------------------------------------------- #
# Tool-use checks
# --------------------------------------------------------------------------- #


@register_check("tool_called")
async def tool_called(ctx: CheckContext, params: dict[str, Any]) -> CheckOutcome:
    tool: str = require(params, "tool")
    expected: dict[str, Any] = params.get("args", {})
    min_times = int(params.get("min_times", 1))
    successful_only = bool(params.get("successful", True))

    attempts = ctx.trace.calls_to(tool)
    calls = [c for c in attempts if c.succeeded] if successful_only else attempts
    matching = [c for c in calls if _args_match(c.call.arguments, expected)]
    if len(matching) >= min_times:
        return CheckOutcome(True, f"'{tool}' called {len(matching)}x with matching args")
    if calls and expected:
        return CheckOutcome(
            False,
            f"'{tool}' called but args never matched {{{_fmt_args(expected)}}}; "
            f"last call: {{{_fmt_args(calls[-1].call.arguments)}}}",
        )
    if attempts and successful_only:
        errors = "; ".join(sorted({c.error or "" for c in attempts}))
        return CheckOutcome(
            False, f"'{tool}' attempted {len(attempts)}x but never succeeded: {errors}"
        )
    return CheckOutcome(False, f"expected '{tool}' to be called at least {min_times}x, got 0")


@register_check("tool_not_called")
async def tool_not_called(ctx: CheckContext, params: dict[str, Any]) -> CheckOutcome:
    """Even an *attempt* counts: a refused refund call is still intent to refund."""
    tool: str = require(params, "tool")
    attempts = ctx.trace.calls_to(tool)
    if not attempts:
        return CheckOutcome(True, f"'{tool}' was not called")
    first = attempts[0]
    return CheckOutcome(
        False,
        f"'{tool}' was called {len(attempts)}x (first: {{{_fmt_args(first.call.arguments)}}}, "
        f"turn {first.turn + 1})",
    )


def _is_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    it = iter(haystack)
    return all(any(item == h for h in it) for item in needle)


@register_check("call_order")
async def call_order(ctx: CheckContext, params: dict[str, Any]) -> CheckOutcome:
    """The given tools were successfully called in this relative order."""
    tools: list[str] = require(params, "tools")
    sequence = [r.call.name for r in ctx.trace.tool_calls if r.succeeded]
    if _is_subsequence(tools, sequence):
        return CheckOutcome(True, " -> ".join(tools))
    return CheckOutcome(
        False, f"expected order {' -> '.join(tools)}; actual: {' -> '.join(sequence) or '(none)'}"
    )


@register_check("max_tool_calls")
async def max_tool_calls(ctx: CheckContext, params: dict[str, Any]) -> CheckOutcome:
    limit = int(require(params, "limit"))
    tool: str | None = params.get("tool")
    calls: list[ToolCallRecord] = ctx.trace.calls_to(tool) if tool else ctx.trace.tool_calls
    what = f"'{tool}' calls" if tool else "tool calls"
    if len(calls) <= limit:
        return CheckOutcome(True, f"{len(calls)} {what} (limit {limit})")
    return CheckOutcome(False, f"{len(calls)} {what}, limit is {limit}")


@register_check("rule")
async def rule(ctx: CheckContext, params: dict[str, Any]) -> CheckOutcome:
    """Run a named business rule declared in the suite's tools module."""
    name: str = require(params, "name")
    fn = ctx.registry.get_rule(name)
    if fn is None:
        return CheckOutcome(False, f"unknown rule '{name}'")
    outcome = fn(ctx.trace)
    if isinstance(outcome, tuple):
        passed, message = outcome
        return CheckOutcome(bool(passed), message)
    return CheckOutcome(bool(outcome), "")


# --------------------------------------------------------------------------- #
# Response checks
# --------------------------------------------------------------------------- #


@register_check("response_contains")
async def response_contains(ctx: CheckContext, params: dict[str, Any]) -> CheckOutcome:
    text: str = require(params, "text")
    scope = params.get("scope", "last")
    texts = _responses(ctx, scope)
    hits = [t for t in texts if text.casefold() in t.casefold()]
    ok = bool(hits) if scope != "all" else len(hits) == len(texts) and bool(texts)
    return CheckOutcome(ok, f"{'found' if ok else 'missing'} {text!r} ({scope} response)")


@register_check("response_not_contains")
async def response_not_contains(ctx: CheckContext, params: dict[str, Any]) -> CheckOutcome:
    text: str = require(params, "text")
    texts = _responses(ctx, params.get("scope", "any"))
    leaked = [t for t in texts if text.casefold() in t.casefold()]
    if leaked:
        return CheckOutcome(False, f"response contains forbidden text {text!r}")
    return CheckOutcome(True, f"no response contains {text!r}")


@register_check("response_matches")
async def response_matches(ctx: CheckContext, params: dict[str, Any]) -> CheckOutcome:
    pattern = re.compile(require(params, "pattern"), re.IGNORECASE | re.MULTILINE)
    scope = params.get("scope", "last")
    texts = _responses(ctx, scope)
    hits = [t for t in texts if pattern.search(t)]
    ok = bool(hits) if scope != "all" else len(hits) == len(texts) and bool(texts)
    return CheckOutcome(ok, f"/{pattern.pattern}/ {'matched' if ok else 'did not match'}")
