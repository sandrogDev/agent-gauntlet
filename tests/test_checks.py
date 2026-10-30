from __future__ import annotations

from typing import Any

import pytest

from conftest import ScriptedProvider, build_mini_registry, say
from gauntlet.checks import CheckContext, run_checks
from gauntlet.checks.judge import parse_verdict, render_transcript
from gauntlet.models import (
    CheckSpec,
    Message,
    Scenario,
    ToolCall,
    ToolCallRecord,
    Trace,
)


def _record(name: str, step: int = 0, error: str | None = None, **args: Any) -> ToolCallRecord:
    return ToolCallRecord(
        turn=0, step=step, call=ToolCall(id=f"{name}{step}", name=name, arguments=args), error=error
    )


def _ctx(*records: ToolCallRecord, responses: list[str] | None = None, judge=None) -> CheckContext:
    trace = Trace(
        scenario_id="t", tool_calls=list(records), final_responses=responses or ["All done."]
    )
    return CheckContext(
        scenario=Scenario(id="t", turns=["x"]),
        trace=trace,
        registry=build_mini_registry(),
        judge=judge,
    )


async def _one(ctx: CheckContext, **spec: Any):
    (result,) = await run_checks([CheckSpec.model_validate(spec)], ctx)
    return result


async def test_tool_called_subset_and_case_insensitive() -> None:
    ctx = _ctx(_record("get_order", order_id="A1", verbose=True))
    assert (await _one(ctx, check="tool_called", tool="get_order", args={"order_id": "a1"})).passed
    assert not (
        await _one(ctx, check="tool_called", tool="get_order", args={"order_id": "B2"})
    ).passed


async def test_tool_called_requires_success_by_default() -> None:
    ctx = _ctx(_record("get_order", error="timeout", order_id="A1"))
    result = await _one(ctx, check="tool_called", tool="get_order")
    assert not result.passed and "never succeeded" in result.message
    assert (await _one(ctx, check="tool_called", tool="get_order", successful=False)).passed


async def test_tool_not_called_counts_failed_attempts() -> None:
    ctx = _ctx(_record("cancel_order", error="denied", order_id="A1"))
    assert not (await _one(ctx, check="tool_not_called", tool="cancel_order")).passed
    assert (await _one(ctx, check="tool_not_called", tool="explode")).passed


async def test_call_order_is_subsequence() -> None:
    ctx = _ctx(_record("get_order", 0), _record("explode", 1), _record("cancel_order", 2))
    assert (await _one(ctx, check="call_order", tools=["get_order", "cancel_order"])).passed
    assert not (await _one(ctx, check="call_order", tools=["cancel_order", "get_order"])).passed


async def test_max_tool_calls_overall_and_per_tool() -> None:
    ctx = _ctx(_record("get_order", 0), _record("get_order", 1), _record("cancel_order", 2))
    assert not (await _one(ctx, check="max_tool_calls", limit=2)).passed
    assert (await _one(ctx, check="max_tool_calls", limit=2, tool="get_order")).passed


async def test_schema_valid_reports_details() -> None:
    bad = _record("get_order", order_id="zz")
    bad.schema_errors = ["order_id: 'zz' does not match '^[A-Z][0-9]$'"]
    result = await _one(_ctx(bad), check="schema_valid")
    assert not result.passed and "does not match" in result.message


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"check": "response_contains", "text": "DONE"}, True),
        ({"check": "response_not_contains", "text": "secret"}, True),
        ({"check": "response_matches", "pattern": r"all\s+done"}, True),
        ({"check": "response_matches", "pattern": "nope"}, False),
    ],
)
async def test_response_checks(spec: dict[str, Any], expected: bool) -> None:
    assert (await _one(_ctx(), **spec)).passed is expected


async def test_response_scope_all() -> None:
    ctx = _ctx(responses=["hello there", "bye"])
    assert not (await _one(ctx, check="response_contains", text="hello", scope="all")).passed
    assert (await _one(ctx, check="response_contains", text="hello", scope="any")).passed


async def test_rule_check_uses_registry() -> None:
    ctx = _ctx(_record("cancel_order", 0, order_id="A1"))
    result = await _one(ctx, check="rule", name="cancel_after_lookup")
    assert not result.passed and result.message == "cancelled without lookup"
    assert not (await _one(ctx, check="rule", name="nope")).passed


async def test_misconfigured_and_unknown_checks_fail_gracefully() -> None:
    assert "missing required parameter" in (await _one(_ctx(), check="tool_called")).message
    assert "unknown check" in (await _one(_ctx(), check="does_not_exist")).message


async def test_judge_skipped_without_provider() -> None:
    result = await _one(_ctx(), check="judge", rubric="be nice")
    assert result.skipped


async def test_judge_parses_verdict() -> None:
    judge = ScriptedProvider([say('Sure! {"verdict": "fail", "reason": "it was rude"}')])
    result = await _one(_ctx(judge=judge), check="judge", rubric="be nice")
    assert not result.passed and "rude" in result.message
    prompt = judge.requests[0][0].content
    assert "<rubric>" in prompt and "be nice" in prompt


def test_parse_verdict_rejects_garbage() -> None:
    assert parse_verdict("no json here") is None
    assert parse_verdict('{"verdict": "maybe"}') is None
    assert parse_verdict('{"verdict": "PASS", "reason": "fine"}') == (True, "fine")


def test_render_transcript_includes_tool_traffic() -> None:
    trace = Trace(
        scenario_id="t",
        messages=[
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="1", name="get_order", arguments={"order_id": "A1"})],
            ),
            Message(role="tool", content='{"error": "x"}', is_error=True, tool_call_id="1"),
        ],
    )
    text = render_transcript(trace)
    assert (
        "USER: hi" in text and 'TOOL get_order({"order_id": "A1"})' in text and "TOOL ERROR" in text
    )
