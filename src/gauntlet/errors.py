"""Exception hierarchy.

Kept in its own module so every other module can import it without cycles.
"""

from __future__ import annotations


class GauntletError(Exception):
    """Base class for all errors raised by gauntlet."""


class ConfigError(GauntletError):
    """A suite, tools module or CLI argument is invalid. Maps to exit code 2."""


class ProviderError(GauntletError):
    """A model provider call failed.

    ``retryable`` tells the resilience layer whether retrying can help
    (rate limits, timeouts, 5xx) or not (bad request, auth, invalid model).
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class CircuitOpenError(ProviderError):
    """Raised without calling the provider because its circuit breaker is open."""

    def __init__(self, message: str = "circuit breaker is open; failing fast") -> None:
        super().__init__(message, retryable=False)
