from __future__ import annotations

import textwrap

import pytest

from conftest import build_mini_registry
from gauntlet import ConfigError, ToolRegistry
from gauntlet.loader import lint_against_registry, load_suite
from gauntlet.tools import load_registry_factory


def _write(tmp_path, body: str):
    path = tmp_path / "suite.yaml"
    path.write_text(textwrap.dedent(body))
    return path


BASE = """
name: t
system_prompt: hi
tools_module: tools.py
scenarios:
  - id: s1
    turns: [hello]
    expect:
      - {check: tool_called, tool: get_order}
"""


def test_loads_valid_suite(tmp_path) -> None:
    suite = load_suite(_write(tmp_path, BASE))
    assert suite.scenarios[0].expect[0].params == {"tool": "get_order"}
    assert suite.tools_path == (tmp_path / "tools.py").resolve()
    assert [c.check for c in suite.global_checks] == ["schema_valid", "completed"]


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ("check: tool_called", "check: tool_callled"),
        ("id: s1", "id: s1\n    turns_typo: 1"),
    ],
)
def test_static_errors(tmp_path, patch: str, message: str) -> None:
    with pytest.raises(ConfigError):
        load_suite(_write(tmp_path, BASE.replace(patch, message)))


def test_duplicate_ids_rejected(tmp_path) -> None:
    body = BASE + "  - id: s1\n    turns: [again]\n"
    with pytest.raises(ConfigError, match="duplicate scenario ids"):
        load_suite(_write(tmp_path, body))


def test_unknown_mutator_and_tag(tmp_path) -> None:
    body = BASE + "adversarial:\n  - {mutator: nope}\n  - {mutator: typo_noise, tags: [missing]}\n"
    with pytest.raises(ConfigError) as err:
        load_suite(_write(tmp_path, body))
    assert "unknown mutator 'nope'" in str(err.value)
    assert "no scenario has tag 'missing'" in str(err.value)


def test_lint_against_registry(tmp_path) -> None:
    body = BASE.replace("tool: get_order", "tool: get_ordr") + (
        "adversarial:\n  - {mutator: tool_failure, tool: nope}\n"
    )
    suite = load_suite(_write(tmp_path, body))
    problems = lint_against_registry(suite, build_mini_registry())
    assert any("unknown tool 'get_ordr'" in p for p in problems)
    assert any("unknown tool 'nope'" in p for p in problems)


def test_registry_rejects_bad_schema_and_duplicates() -> None:
    reg = ToolRegistry()
    with pytest.raises(ConfigError, match="invalid JSON Schema"):
        reg.tool(description="x", parameters={"type": "nonsense"})(lambda: None)
    reg.tool(description="x", parameters={"type": "object"}, name="t")(lambda: None)
    with pytest.raises(ConfigError, match="registered twice"):
        reg.tool(description="x", parameters={"type": "object"}, name="t")(lambda: None)


def test_format_checker_enforces_dates() -> None:
    reg = ToolRegistry()
    reg.tool(
        description="x",
        parameters={"type": "object", "properties": {"d": {"type": "string", "format": "date"}}},
        name="t",
    )(lambda d: d)
    assert reg.validate("t", {"d": "2026-12-05"}) == []
    assert reg.validate("t", {"d": "December 5"})


def test_load_registry_factory_errors(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_registry_factory(tmp_path / "missing.py")
    bad = tmp_path / "bad.py"
    bad.write_text("build_registry = 3\n")
    with pytest.raises(ConfigError, match="callable"):
        load_registry_factory(bad)
    wrong = tmp_path / "wrong.py"
    wrong.write_text("def build_registry():\n    return {}\n")
    with pytest.raises(ConfigError, match="not ToolRegistry"):
        load_registry_factory(wrong)
