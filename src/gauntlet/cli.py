"""Command-line interface.

Exit codes (stable, for CI):
  0  success
  1  quality gate failed (regressions, pass rate below --min-pass-rate)
  2  configuration / usage error
  3  unexpected internal error
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from gauntlet import __version__
from gauntlet.errors import ConfigError, GauntletError
from gauntlet.loader import lint_against_registry, load_suite
from gauntlet.models import RunResult, Suite
from gauntlet.mutators import expand
from gauntlet.providers import create_provider
from gauntlet.regression import RunComparison, compare
from gauntlet.reporting import print_comparison, print_run_summary, progress_line, write_html
from gauntlet.reporting.console import Style
from gauntlet.resilience import CircuitBreaker, ResilientProvider, RetryPolicy
from gauntlet.runner import Runner
from gauntlet.storage import list_runs, resolve_run, save_run
from gauntlet.tools import RegistryFactory, load_registry_factory

EXIT_OK, EXIT_GATE, EXIT_CONFIG, EXIT_INTERNAL = 0, 1, 2, 3
DEFAULT_PROVIDER = "anthropic:claude-sonnet-5-5"


def _load(suite_path: Path) -> tuple[Suite, RegistryFactory]:
    suite = load_suite(suite_path)
    factory = load_registry_factory(suite.tools_path)
    problems = lint_against_registry(suite, factory())
    if problems:
        raise ConfigError(f"{suite_path}:\n  - " + "\n  - ".join(problems))
    return suite, factory


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_run(args: argparse.Namespace) -> int:
    suite, factory = _load(args.suite)
    scenarios = expand(suite, seed=args.seed, adversarial=args.adversarial)
    if args.tag:
        wanted = set(args.tag)
        scenarios = [s for s in scenarios if wanted & set(s.tags)]
    if args.scenario:
        prefixes = tuple(args.scenario)
        scenarios = [s for s in scenarios if s.id.startswith(prefixes)]
    if not scenarios:
        raise ConfigError("no scenarios selected (check --tag / --scenario filters)")

    policy = RetryPolicy(max_attempts=args.max_attempts, timeout_s=args.timeout)
    breaker = CircuitBreaker()
    provider = ResilientProvider(
        create_provider(args.provider), policy=policy, breaker=breaker, seed=args.seed
    )
    judge = (
        ResilientProvider(create_provider(args.judge), policy=policy, seed=args.seed)
        if args.judge
        else None
    )

    baseline: RunResult | None = None
    if args.baseline:
        baseline = resolve_run(args.baseline, args.out_dir, suite=suite.name)

    style = Style(sys.stdout)
    print(
        style.bold(f"gauntlet {__version__}")
        + f"  {suite.name}: {len(scenarios)} scenarios against {provider.name}"
        + (f" (judge: {judge.name})" if judge else "")
    )
    runner = Runner(
        suite,
        provider,
        factory,
        judge=judge,
        concurrency=args.concurrency,
        seed=args.seed,
        on_result=lambda r: print(progress_line(r, style), flush=True),
    )
    run = asyncio.run(runner.run(scenarios))

    run_path = save_run(run, args.out_dir)
    print_run_summary(run)
    if provider.retries:
        print(style.dim(f"\n  provider retries: {provider.retries}"))

    comparison: RunComparison | None = None
    if baseline is not None:
        comparison = compare(baseline, run, tolerance=args.tolerance)
        print_comparison(comparison)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    if args.html:
        write_html(run, args.html, comparison)
        print(f"\n  HTML report: {args.html}")
    print(style.dim(f"  run saved:   {run_path}"))

    return _gate(run, comparison, args)


def _gate(run: RunResult, comparison: RunComparison | None, args: argparse.Namespace) -> int:
    reasons: list[str] = []
    if run.metrics.pass_rate < args.min_pass_rate:
        reasons.append(f"pass rate {run.metrics.pass_rate:.1%} < required {args.min_pass_rate:.1%}")
    if args.fail_on_regression and comparison is not None and comparison.has_regressions:
        reasons.append("regressions vs baseline")
    if args.fail_on_critical and run.metrics.critical_failures:
        reasons.append(f"{run.metrics.critical_failures} scenario(s) with critical failures")
    if reasons:
        print(Style(sys.stdout).red("\n  QUALITY GATE FAILED: " + "; ".join(reasons)))
        return EXIT_GATE
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = resolve_run(args.baseline, args.out_dir)
    candidate = resolve_run(args.candidate, args.out_dir)
    comparison = compare(baseline, candidate, tolerance=args.tolerance)
    print_comparison(comparison)
    if args.html:
        write_html(candidate, args.html, comparison)
        print(f"\n  HTML report: {args.html}")
    return EXIT_GATE if args.fail_on_regression and comparison.has_regressions else EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    run = resolve_run(args.run, args.out_dir)
    print_run_summary(run)
    if args.html:
        write_html(run, args.html)
        print(f"\n  HTML report: {args.html}")
    return EXIT_OK


def cmd_scenarios(args: argparse.Namespace) -> int:
    suite, _ = _load(args.suite)
    for s in expand(suite, seed=args.seed, adversarial=args.adversarial):
        origin = f"  <- {s.variant_of} via {s.mutator}" if s.variant_of else ""
        print(f"{s.id:<48} {len(s.turns)} turn(s), {len(s.expect)} check(s){origin}")
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    suite, factory = _load(args.suite)
    registry = factory()
    variants = len(expand(suite)) - len(suite.scenarios)
    print(
        f"OK  {suite.name}: {len(suite.scenarios)} scenarios (+{variants} adversarial variants), "
        f"{len(registry.tool_names)} tools, {len(registry.rule_names)} rules"
    )
    return EXIT_OK


def cmd_history(args: argparse.Namespace) -> int:
    from gauntlet.storage import load_run_file

    for path in list_runs(args.out_dir):
        run = load_run_file(path)
        m = run.metrics
        print(
            f"{run.run_id}  {run.suite:<24} {run.provider:<36} "
            f"pass {m.pass_rate:6.1%}  score {m.mean_score:.3f}"
        )
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _unit_interval(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gauntlet",
        description="Adversarial evaluation harness for tool-calling LLM agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".gauntlet"),
        help="where runs are stored (default: .gauntlet)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite against a provider")
    run.add_argument("suite", type=Path)
    run.add_argument(
        "-p",
        "--provider",
        default=DEFAULT_PROVIDER,
        help="anthropic:<model> | openai:<model> | python:<file.py>[:<factory>]",
    )
    run.add_argument("--judge", help="provider spec for LLM-as-judge checks (optional)")
    run.add_argument("-c", "--concurrency", type=int, default=4)
    run.add_argument("--seed", type=int, default=0, help="seed for adversarial variants and jitter")
    run.add_argument(
        "--no-adversarial",
        dest="adversarial",
        action="store_false",
        help="run only the hand-written scenarios",
    )
    run.add_argument(
        "-t", "--tag", action="append", help="only scenarios with this tag (repeatable)"
    )
    run.add_argument(
        "-s", "--scenario", action="append", help="only scenario ids with this prefix (repeatable)"
    )
    run.add_argument("--html", type=Path, help="write a self-contained HTML report")
    run.add_argument("--json", type=Path, help="also write the run JSON here")
    run.add_argument(
        "--baseline", help="'latest', 'previous', a run id or a run file to compare against"
    )
    run.add_argument(
        "--tolerance", type=float, default=0.05, help="score drop treated as degradation"
    )
    run.add_argument("--fail-on-regression", action="store_true")
    run.add_argument("--fail-on-critical", action="store_true")
    run.add_argument("--min-pass-rate", type=_unit_interval, default=0.0)
    run.add_argument("--max-attempts", type=int, default=4, help="provider attempts per call")
    run.add_argument("--timeout", type=float, default=60.0, help="seconds per provider call")
    run.set_defaults(func=cmd_run)

    cmp = sub.add_parser("compare", help="diff two stored runs")
    cmp.add_argument("baseline")
    cmp.add_argument("candidate", nargs="?", default="latest")
    cmp.add_argument("--tolerance", type=float, default=0.05)
    cmp.add_argument("--fail-on-regression", action="store_true")
    cmp.add_argument("--html", type=Path)
    cmp.set_defaults(func=cmd_compare)

    rep = sub.add_parser("report", help="re-render a stored run")
    rep.add_argument("run", nargs="?", default="latest")
    rep.add_argument("--html", type=Path)
    rep.set_defaults(func=cmd_report)

    sc = sub.add_parser("scenarios", help="list scenarios including adversarial variants")
    sc.add_argument("suite", type=Path)
    sc.add_argument("--seed", type=int, default=0)
    sc.add_argument("--no-adversarial", dest="adversarial", action="store_false")
    sc.set_defaults(func=cmd_scenarios)

    val = sub.add_parser(
        "validate", help="lint a suite and its tools module without calling a model"
    )
    val.add_argument("suite", type=Path)
    val.set_defaults(func=cmd_validate)

    hist = sub.add_parser("history", help="list stored runs")
    hist.set_defaults(func=cmd_history)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except GauntletError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
