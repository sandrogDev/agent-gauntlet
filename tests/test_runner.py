from __future__ import annotations

from conftest import ScriptedProvider, call, make_suite, say
from gauntlet import ProviderError
from gauntlet.models import CheckSpec, Scenario, Severity
from gauntlet.runner import Runner


def _scenario(**kwargs: object) -> Scenario:
    return Scenario.model_validate({"id": "s1", "turns": ["cancel order A1"], **kwargs})


async def test_happy_path_records_full_trace(registry_factory) -> None:
    provider = ScriptedProvider(
        [call("get_order", order_id="A1"), call("cancel_order", order_id="A1"), say("Cancelled.")]
    )
    runner = Runner(make_suite(), provider, registry_factory)
    result = await runner.run_scenario(_scenario())

    trace = result.trace
    assert [r.call.name for r in trace.tool_calls] == ["get_order", "cancel_order"]
    assert all(r.succeeded and r.valid for r in trace.tool_calls)
    assert trace.tool_calls[1].result == {"order_id": "A1", "status": "cancelled"}
    assert trace.final_responses == ["Cancelled."]
    assert trace.steps == 3
    assert [m.role for m in trace.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.passed and result.score == 1.0


async def test_invalid_arguments_are_rejected_before_execution(registry_factory) -> None:
    provider = ScriptedProvider([call("get_order", order_id="not-an-id"), say("hm")])
    result = await Runner(make_suite(), provider, registry_factory).run_scenario(_scenario())

    record = result.trace.tool_calls[0]
    assert not record.valid
    assert record.result is None
    assert "does not match" in record.schema_errors[0]
    # schema_valid is a critical global check
    assert not result.passed


async def test_unknown_tool_is_flagged(registry_factory) -> None:
    provider = ScriptedProvider([call("delete_everything"), say("done")])
    result = await Runner(make_suite(), provider, registry_factory).run_scenario(_scenario())
    assert result.trace.tool_calls[0].unknown_tool
    assert not result.passed


async def test_tool_errors_are_returned_to_the_agent(registry_factory) -> None:
    provider = ScriptedProvider([call("get_order", order_id="Z9"), call("explode"), say("sorry")])
    result = await Runner(make_suite(), provider, registry_factory).run_scenario(_scenario())

    business, crash = result.trace.tool_calls
    assert business.error == "order not found"
    assert crash.error is not None and "RuntimeError" in crash.error
    tool_msgs = [m for m in result.trace.messages if m.role == "tool"]
    assert all(m.is_error for m in tool_msgs)


async def test_fault_injection_hits_only_first_call(registry_factory) -> None:
    provider = ScriptedProvider(
        [call("get_order", order_id="A1"), call("get_order", order_id="A1"), say("ok")]
    )
    scenario = _scenario(tool_faults={"get_order": "503 retry"})
    result = await Runner(make_suite(), provider, registry_factory).run_scenario(scenario)

    first, second = result.trace.tool_calls
    assert first.injected_fault and first.error == "503 retry"
    assert second.succeeded and not second.injected_fault


async def test_step_limit_is_enforced(registry_factory) -> None:
    provider = ScriptedProvider([call("get_order", order_id="A1")] * 10)
    result = await Runner(make_suite(max_steps=3), provider, registry_factory).run_scenario(
        _scenario()
    )
    assert result.trace.hit_step_limit
    assert result.trace.steps == 3
    assert not result.passed


async def test_provider_failure_is_captured_not_raised(registry_factory) -> None:
    provider = ScriptedProvider([ProviderError("auth failed")])
    result = await Runner(make_suite(), provider, registry_factory).run_scenario(_scenario())
    assert result.trace.error == "auth failed"
    assert result.score == 0.0 and not result.passed


async def test_multi_turn_keeps_history(registry_factory) -> None:
    provider = ScriptedProvider(
        [say("Which order?"), call("get_order", order_id="A1"), say("Found it")]
    )
    scenario = _scenario(turns=["help", "order A1"])
    result = await Runner(make_suite(), provider, registry_factory).run_scenario(scenario)
    assert result.trace.final_responses == ["Which order?", "Found it"]
    last_request = provider.requests[-1]
    assert [m.content for m in last_request if m.role == "user"] == ["help", "order A1"]


async def test_each_scenario_gets_fresh_tool_state(registry_factory) -> None:
    script = [
        call("cancel_order", order_id="A1"),
        say("x"),
        call("get_order", order_id="A1"),
        say("y"),
    ]
    provider = ScriptedProvider(script)
    runner = Runner(make_suite(), provider, registry_factory, concurrency=1)
    run = await runner.run([_scenario(id="first"), _scenario(id="second")])
    lookup = run.results[1].trace.tool_calls[0]
    assert lookup.result["status"] == "open"  # not leaked from the first scenario


async def test_run_aggregates_metrics_and_keeps_order(registry_factory) -> None:
    provider = ScriptedProvider([say("a"), say("b"), say("c")])
    expect = [
        CheckSpec.model_validate(
            {"check": "response_contains", "text": "b", "severity": Severity.CRITICAL}
        )
    ]
    scenarios = [_scenario(id=f"s{i}", expect=expect) for i in range(3)]
    seen: list[str] = []
    runner = Runner(
        make_suite(),
        provider,
        registry_factory,
        concurrency=1,
        on_result=lambda r: seen.append(r.scenario.id),
    )
    run = await runner.run(scenarios)

    assert [r.scenario.id for r in run.results] == ["s0", "s1", "s2"]
    assert seen == ["s0", "s1", "s2"]
    assert run.metrics.scenarios == 3 and run.metrics.passed == 1
    assert run.metrics.failures_by_check == {"response_contains": 2}
