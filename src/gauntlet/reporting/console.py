"""Plain-terminal reporting with optional ANSI colour (no rich dependency)."""

from __future__ import annotations

import os
import sys
from typing import TextIO

from gauntlet.models import RunResult, ScenarioResult
from gauntlet.regression import DiffStatus, RunComparison


class Style:
    def __init__(self, stream: TextIO) -> None:
        self.enabled = (
            stream.isatty() and "NO_COLOR" not in os.environ
        ) or "FORCE_COLOR" in os.environ

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def green(self, t: str) -> str:
        return self._wrap("32", t)

    def red(self, t: str) -> str:
        return self._wrap("31", t)

    def yellow(self, t: str) -> str:
        return self._wrap("33", t)

    def dim(self, t: str) -> str:
        return self._wrap("2", t)

    def bold(self, t: str) -> str:
        return self._wrap("1", t)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def progress_line(result: ScenarioResult, style: Style) -> str:
    mark = style.green("PASS") if result.passed else style.red("FAIL")
    line = f"  {mark}  {result.scenario.id:<44} score {result.score:.2f}"
    if not result.passed and result.failures:
        worst = result.failures[0]
        line += style.dim(f"  <- {worst.label}")
    return line


def print_run_summary(run: RunResult, stream: TextIO | None = None) -> None:
    stream = stream or sys.stdout
    s, m = Style(stream), run.metrics
    write = stream.write
    write(
        "\n"
        + s.bold(f"Run {run.run_id}")
        + s.dim(f"  suite={run.suite}  provider={run.provider}  seed={run.seed}")
        + "\n"
    )
    rows = [
        ("Pass rate", f"{_pct(m.pass_rate)}  ({m.passed}/{m.scenarios})"),
        ("  baseline scenarios", _pct(m.baseline_pass_rate)),
        ("  adversarial variants", _pct(m.adversarial_pass_rate)),
        ("Mean score", f"{m.mean_score:.3f}"),
        ("Tool-call schema validity", f"{_pct(m.tool_call_validity)}  ({m.tool_calls} calls)"),
        ("Tool-call success rate", _pct(m.tool_call_success_rate)),
        ("Critical failures", str(m.critical_failures)),
        ("Latency mean / p95", f"{m.mean_latency_s:.2f}s / {m.p95_latency_s:.2f}s"),
        ("Tokens in / out", f"{m.input_tokens:,} / {m.output_tokens:,}"),
    ]
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        write(f"  {key:<{width}}  {value}\n")

    if m.pass_rate_by_mutator:
        write("\n" + s.bold("Robustness by attack") + "\n")
        for mutator, rate in m.pass_rate_by_mutator.items():
            colour = s.green if rate >= 0.8 else s.yellow if rate >= 0.5 else s.red
            write(f"  {mutator:<22} {colour(_pct(rate))}\n")

    failed = [r for r in run.results if not r.passed]
    if failed:
        write("\n" + s.bold(f"Findings ({len(failed)})") + "\n")
        for r in failed:
            write(f"  {s.red('x')} {r.scenario.id}\n")
            for c in r.failures:
                write(s.dim(f"      [{c.severity}] {c.label}: {c.message}") + "\n")


def print_comparison(cmp: RunComparison, stream: TextIO | None = None) -> None:
    stream = stream or sys.stdout
    s = Style(stream)
    write = stream.write
    write("\n" + s.bold(f"Comparison {cmp.baseline_id} -> {cmp.candidate_id}") + "\n")
    for name, delta in cmp.metric_deltas.items():
        better = delta <= 0 if name.endswith("latency_s") else delta >= 0
        colour = s.green if better else s.red
        write(f"  {name:<24} {colour(f'{delta:+.4f}')}\n")

    colours = {
        DiffStatus.REGRESSED: s.red,
        DiffStatus.DEGRADED: s.yellow,
        DiffStatus.FIXED: s.green,
        DiffStatus.IMPROVED: s.green,
    }
    interesting = [
        d
        for d in cmp.diffs
        if d.status in colours or d.status in (DiffStatus.NEW, DiffStatus.REMOVED)
    ]
    if not interesting:
        write(s.dim("  no scenario changed status\n"))
    for d in interesting:
        colour = colours.get(d.status, s.dim)
        change = "" if d.delta is None else f" ({d.delta:+.2f})"
        write(f"  {colour(f'{d.status.value.upper():<10}')} {d.scenario_id}{change}\n")
        for label in d.newly_failing:
            write(s.dim(f"      newly failing: {label}") + "\n")
    verdict = s.red("REGRESSIONS DETECTED") if cmp.has_regressions else s.green("no regressions")
    write(f"\n  {verdict}\n")
