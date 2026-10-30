"""Load suites from YAML and lint them against their tool registry.

Catching a typo'd tool name at load time is far cheaper than discovering it as
a mysterious 0% pass rate after spending an hour of API budget.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from gauntlet.checks import CHECKS
from gauntlet.errors import ConfigError
from gauntlet.models import CheckSpec, Suite
from gauntlet.mutators import MUTATORS
from gauntlet.tools import ToolRegistry

_TOOL_PARAM_CHECKS = {"tool_called", "tool_not_called", "max_tool_calls"}


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"suite file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    try:
        suite = Suite.model_validate({**data, "base_dir": path.parent})
    except ValidationError as exc:
        raise ConfigError(f"{path}: invalid suite definition\n{exc}") from exc

    problems = _static_problems(suite)
    if problems:
        raise ConfigError(f"{path}:\n  - " + "\n  - ".join(problems))
    return suite


def _all_specs(suite: Suite) -> list[tuple[str, CheckSpec]]:
    specs = [("global_checks", c) for c in suite.global_checks]
    specs += [(f"scenario '{s.id}'", c) for s in suite.scenarios for c in s.expect]
    specs += [(f"mutator '{m.mutator}'", c) for m in suite.adversarial for c in m.extra_expect]
    return specs


def _static_problems(suite: Suite) -> list[str]:
    problems = [
        f"{where}: unknown check '{c.check}' (available: {', '.join(sorted(CHECKS))})"
        for where, c in _all_specs(suite)
        if c.check not in CHECKS
    ]
    problems += [
        f"adversarial: unknown mutator '{m.mutator}' (available: {', '.join(sorted(MUTATORS))})"
        for m in suite.adversarial
        if m.mutator not in MUTATORS
    ]
    ids = {s.id for s in suite.scenarios}
    tags = {t for s in suite.scenarios for t in s.tags}
    for m in suite.adversarial:
        problems += [
            f"mutator '{m.mutator}': unknown scenario '{x}'" for x in m.scenarios if x not in ids
        ]
        problems += [
            f"mutator '{m.mutator}': no scenario has tag '{t}'" for t in m.tags if t not in tags
        ]
    return problems


def lint_against_registry(suite: Suite, registry: ToolRegistry) -> list[str]:
    """Cross-check tool and rule names referenced by the suite."""
    problems: list[str] = []
    known = set(registry.tool_names)

    def check_tool(where: str, name: object) -> None:
        if isinstance(name, str) and name not in known:
            problems.append(f"{where}: unknown tool '{name}' (tools: {', '.join(sorted(known))})")

    for where, spec in _all_specs(suite):
        params = spec.params
        if spec.check in _TOOL_PARAM_CHECKS:
            check_tool(where, params.get("tool"))
        elif spec.check == "call_order":
            for name in params.get("tools", []):
                check_tool(where, name)
        elif spec.check == "rule" and registry.get_rule(str(params.get("name"))) is None:
            problems.append(f"{where}: unknown rule '{params.get('name')}'")

    for s in suite.scenarios:
        for name in s.tool_faults:
            check_tool(f"scenario '{s.id}' tool_faults", name)
    for m in suite.adversarial:
        for name in m.params.get("forbidden_tools", []):
            check_tool(f"mutator '{m.mutator}' forbidden_tools", name)
        if m.mutator == "tool_failure":
            check_tool(f"mutator '{m.mutator}'", m.params.get("tool"))
    return problems
