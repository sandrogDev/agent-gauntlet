"""Drive an agent through scenarios and record everything it does.

The runner owns the agent loop instead of delegating it to an SDK helper, which
is what makes the interesting interventions possible: arguments are validated
*before* a tool executes (an invalid call never reaches the sandbox), faults
can be injected into specific tools, and every step is captured in a
provider-neutral trace that checks can reason about.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from gauntlet import __version__
from gauntlet.checks import CheckContext, run_checks
from gauntlet.errors import ProviderError
from gauntlet.models import (
    AgentStep,
    Message,
    RunResult,
    Scenario,
    ScenarioResult,
    Suite,
    ToolCall,
    ToolCallRecord,
    Trace,
)
from gauntlet.providers.base import Provider
from gauntlet.scoring import compute_metrics, score_scenario
from gauntlet.tools import RegistryFactory, ToolError, ToolRegistry

ProgressFn = Callable[[ScenarioResult], None]
MAX_TOOL_RESULT_CHARS = 8_000


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def render_tool_result(record: ToolCallRecord) -> str:
    if record.error is not None:
        return json.dumps({"error": record.error})
    text = json.dumps(record.result, default=str, ensure_ascii=False)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "...<truncated>"
    return text


class Runner:
    def __init__(
        self,
        suite: Suite,
        provider: Provider,
        registry_factory: RegistryFactory,
        *,
        judge: Provider | None = None,
        concurrency: int = 4,
        seed: int = 0,
        on_result: ProgressFn | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.suite = suite
        self.provider = provider
        self.registry_factory = registry_factory
        self.judge = judge
        self.concurrency = concurrency
        self.seed = seed
        self.on_result = on_result

    async def run(self, scenarios: Sequence[Scenario]) -> RunResult:
        started = datetime.now(UTC)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def guarded(scenario: Scenario) -> ScenarioResult:
            async with semaphore:
                result = await self.run_scenario(scenario)
            if self.on_result:
                self.on_result(result)
            return result

        # gather() preserves input order, so reports are stable across runs.
        results = list(await asyncio.gather(*(guarded(s) for s in scenarios)))
        return RunResult(
            run_id=new_run_id(),
            suite=self.suite.name,
            provider=self.provider.name,
            seed=self.seed,
            started_at=started,
            finished_at=datetime.now(UTC),
            gauntlet_version=__version__,
            results=results,
            metrics=compute_metrics(results),
        )

    async def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        registry = self.registry_factory()
        trace = await self._execute(scenario, registry)
        ctx = CheckContext(scenario=scenario, trace=trace, registry=registry, judge=self.judge)
        checks = await run_checks([*self.suite.global_checks, *scenario.expect], ctx)
        score, passed = score_scenario(checks, trace, self.suite.pass_threshold)
        return ScenarioResult(
            scenario=scenario, trace=trace, checks=checks, score=score, passed=passed
        )

    # ------------------------------------------------------------------ #

    async def _execute(self, scenario: Scenario, registry: ToolRegistry) -> Trace:
        trace = Trace(scenario_id=scenario.id)
        pending_faults = dict(scenario.tool_faults)
        tools = registry.specs()
        start = time.perf_counter()
        try:
            for turn, user_text in enumerate(scenario.turns):
                trace.messages.append(Message(role="user", content=user_text))
                step: AgentStep | None = None
                for step_index in range(self.suite.max_steps):
                    step = await self.provider.complete(
                        system=self.suite.system_prompt, messages=list(trace.messages), tools=tools
                    )
                    trace.steps += 1
                    trace.usage = trace.usage + step.usage
                    trace.messages.append(
                        Message(role="assistant", content=step.text, tool_calls=step.tool_calls)
                    )
                    if not step.tool_calls:
                        break
                    for call in step.tool_calls:
                        record = await self._invoke(
                            registry, call, turn, step_index, pending_faults
                        )
                        trace.tool_calls.append(record)
                        trace.messages.append(
                            Message(
                                role="tool",
                                tool_call_id=call.id,
                                content=render_tool_result(record),
                                is_error=record.error is not None,
                            )
                        )
                else:
                    trace.hit_step_limit = True
                trace.final_responses.append(step.text if step else "")
        except ProviderError as exc:
            trace.error = str(exc)
        trace.latency_s = round(time.perf_counter() - start, 4)
        return trace

    @staticmethod
    async def _invoke(
        registry: ToolRegistry,
        call: ToolCall,
        turn: int,
        step: int,
        pending_faults: dict[str, str],
    ) -> ToolCallRecord:
        record = ToolCallRecord(turn=turn, step=step, call=call)
        if call.name not in registry:
            record.unknown_tool = True
            record.error = f"unknown tool '{call.name}'"
            return record

        record.schema_errors = registry.validate(call.name, call.arguments)
        if record.schema_errors:
            # Mirror a real API gateway: reject before the handler runs.
            record.error = "invalid arguments: " + "; ".join(record.schema_errors)
            return record

        if call.name in pending_faults:
            record.error = pending_faults.pop(call.name)
            record.injected_fault = True
            return record

        try:
            record.result = await registry.execute(call.name, call.arguments)
        except ToolError as exc:
            record.error = str(exc)
        except Exception as exc:  # a crashing tool must not crash the run
            record.error = f"internal tool error ({type(exc).__name__}): {exc}"
        return record
