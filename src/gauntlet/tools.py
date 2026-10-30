"""Tool registry: the sandboxed world the agent under test acts upon.

A suite points at a Python module exposing ``build_registry() -> ToolRegistry``.
The runner calls that factory once *per scenario*, so every scenario starts from
fresh, isolated state (no bleed-through between concurrent scenarios), and the
same module also declares the business rules checks can reference.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from gauntlet.errors import ConfigError
from gauntlet.models import ToolSpec, Trace

F = TypeVar("F", bound=Callable[..., Any])
RuleOutcome = bool | tuple[bool, str]
RuleFn = Callable[[Trace], RuleOutcome]
RegistryFactory = Callable[[], "ToolRegistry"]


class ToolError(Exception):
    """Raise from a tool handler to return a *business* error to the agent.

    Unlike an unexpected exception, the message is considered safe to show to
    the model (e.g. "booking not found").
    """


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    side_effect: bool = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        self._rules: dict[str, RuleFn] = {}

    # -- registration ------------------------------------------------------ #

    def tool(
        self,
        *,
        description: str,
        parameters: dict[str, Any],
        name: str | None = None,
        side_effect: bool = False,
    ) -> Callable[[F], F]:
        """Decorator registering a function as a tool."""

        def decorator(fn: F) -> F:
            self.register(
                Tool(
                    name=name or fn.__name__,
                    description=description,
                    parameters=parameters,
                    handler=fn,
                    side_effect=side_effect,
                )
            )
            return fn

        return decorator

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ConfigError(f"tool '{tool.name}' registered twice")
        try:
            Draft202012Validator.check_schema(tool.parameters)
        except SchemaError as exc:
            raise ConfigError(
                f"tool '{tool.name}' has an invalid JSON Schema: {exc.message}"
            ) from exc
        self._tools[tool.name] = tool
        self._validators[tool.name] = Draft202012Validator(
            tool.parameters, format_checker=Draft202012Validator.FORMAT_CHECKER
        )

    def rule(self, name: str) -> Callable[[RuleFn], RuleFn]:
        """Decorator registering a named business rule evaluated over a trace."""

        def decorator(fn: RuleFn) -> RuleFn:
            if name in self._rules:
                raise ConfigError(f"rule '{name}' registered twice")
            self._rules[name] = fn
            return fn

        return decorator

    # -- lookup ------------------------------------------------------------ #

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    @property
    def rule_names(self) -> list[str]:
        return list(self._rules)

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def get_rule(self, name: str) -> RuleFn | None:
        return self._rules.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    # -- execution --------------------------------------------------------- #

    def validate(self, name: str, arguments: dict[str, Any]) -> list[str]:
        """Return human-readable schema violations (empty list == valid)."""
        errors = self._validators[name].iter_errors(arguments)
        return sorted(
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
        )

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        result = self._tools[name].handler(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result


# --------------------------------------------------------------------------- #
# Loading user code
# --------------------------------------------------------------------------- #


def import_from_path(path: Path) -> ModuleType:
    """Import a Python file by path under a stable, collision-free module name."""
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"module not found: {path}")
    digest = hashlib.sha1(str(path).encode(), usedforsecurity=False).hexdigest()[:12]
    module_name = f"gauntlet_user_{path.stem}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[module_name]
        raise ConfigError(f"error while importing {path}: {exc}") from exc
    return module


def load_registry_factory(path: Path, attr: str = "build_registry") -> RegistryFactory:
    module = import_from_path(path)
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise ConfigError(f"{path} must define a callable '{attr}() -> ToolRegistry'")
    probe = factory()
    if not isinstance(probe, ToolRegistry):
        raise ConfigError(f"{path}:{attr}() returned {type(probe).__name__}, not ToolRegistry")
    return factory  # type: ignore[no-any-return]
