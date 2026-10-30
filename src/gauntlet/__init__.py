"""agent-gauntlet: adversarial evaluation harness for tool-calling LLM agents."""

from gauntlet.errors import ConfigError, GauntletError, ProviderError
from gauntlet.models import (
    AgentStep,
    CheckResult,
    Message,
    RunResult,
    Scenario,
    ScenarioResult,
    Suite,
    ToolCall,
    ToolSpec,
    Trace,
    Usage,
)
from gauntlet.tools import ToolError, ToolRegistry

__version__ = "0.1.0"

__all__ = [
    "AgentStep",
    "CheckResult",
    "ConfigError",
    "GauntletError",
    "Message",
    "ProviderError",
    "RunResult",
    "Scenario",
    "ScenarioResult",
    "Suite",
    "ToolCall",
    "ToolError",
    "ToolRegistry",
    "ToolSpec",
    "Trace",
    "Usage",
    "__version__",
]
