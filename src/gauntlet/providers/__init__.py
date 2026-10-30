"""Provider factory.

Provider specs are plain strings so they work identically in the CLI, in YAML
and in CI matrices:

* ``anthropic:<model>``            Anthropic Messages API
* ``openai:<model>``               OpenAI-compatible Chat Completions
* ``python:<file.py>[:<factory>]`` your own agent; ``factory`` defaults to
  ``create_provider`` and must return an object implementing :class:`Provider`.
"""

from __future__ import annotations

from pathlib import Path

from gauntlet.errors import ConfigError
from gauntlet.providers.base import Provider

__all__ = ["Provider", "create_provider"]


def create_provider(spec: str) -> Provider:
    kind, sep, rest = spec.partition(":")
    if not sep or not rest:
        raise ConfigError(
            f"invalid provider spec '{spec}'; expected anthropic:<model>, openai:<model> "
            "or python:<file.py>[:<factory>]"
        )
    if kind == "anthropic":
        from gauntlet.providers.anthropic import AnthropicProvider

        return AnthropicProvider(rest)
    if kind == "openai":
        from gauntlet.providers.openai import OpenAIProvider

        return OpenAIProvider(rest)
    if kind == "python":
        return _load_python_provider(rest)
    raise ConfigError(f"unknown provider kind '{kind}' in '{spec}'")


def _load_python_provider(target: str) -> Provider:
    from gauntlet.tools import import_from_path

    # Split "path/to/agent.py:factory" without breaking Windows drive letters.
    path_str, factory_name = target, "create_provider"
    if target.count(":") and not target.endswith(".py"):
        path_str, _, factory_name = target.rpartition(":")
    module = import_from_path(Path(path_str))
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ConfigError(f"{path_str} must define a callable '{factory_name}()'")
    provider = factory()
    if not isinstance(provider, Provider):
        raise ConfigError(f"{path_str}:{factory_name}() did not return a Provider")
    return provider
