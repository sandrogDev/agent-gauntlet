"""Checks evaluate a finished trace. Importing this package registers built-ins."""

from gauntlet.checks import builtin as _builtin  # noqa: F401  (registers checks)
from gauntlet.checks import judge as _judge  # noqa: F401  (registers checks)
from gauntlet.checks.base import (
    CHECKS,
    CheckContext,
    CheckFn,
    CheckOutcome,
    register_check,
    run_checks,
)

__all__ = ["CHECKS", "CheckContext", "CheckFn", "CheckOutcome", "register_check", "run_checks"]
