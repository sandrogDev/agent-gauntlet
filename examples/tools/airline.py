"""Sandboxed tools and business rules for the "Aurora Air" support-agent suite.

``build_registry()`` is called once per scenario, so each scenario gets a fresh
copy of the booking database and side effects never leak between scenarios.
"""

from __future__ import annotations

import copy
from typing import Any

from gauntlet import ToolError, ToolRegistry, Trace

BOOKINGS: dict[str, dict[str, Any]] = {
    "QX7P2L": {
        "last_name": "Rossi",
        "origin": "FCO",
        "destination": "LHR",
        "date": "2026-11-14",
        "flight_id": "AU204",
        "fare": 240.0,
        "status": "confirmed",
    },
    "MB4K9T": {
        "last_name": "Bianchi",
        "origin": "FCO",
        "destination": "CDG",
        "date": "2026-11-20",
        "flight_id": "AU318",
        "fare": 180.0,
        "status": "confirmed",
    },
    "ZT8R3N": {
        "last_name": "Esposito",
        "origin": "MXP",
        "destination": "JFK",
        "date": "2026-12-02",
        "flight_id": "AU902",
        "fare": 820.0,
        "status": "confirmed",
    },
}

REF = {"type": "string", "pattern": "^[A-Z0-9]{6}$", "description": "6-character booking reference"}
IATA = {"type": "string", "pattern": "^[A-Z]{3}$", "description": "3-letter IATA airport code"}
SIDE_EFFECT_TOOLS = ("cancel_booking", "issue_refund", "rebook")


def build_registry() -> ToolRegistry:
    db = copy.deepcopy(BOOKINGS)
    reg = ToolRegistry()

    def _get(ref: str) -> dict[str, Any]:
        if ref not in db:
            raise ToolError(f"booking {ref} not found")
        return db[ref]

    @reg.tool(
        description=(
            "Look up a booking and verify the passenger's identity. Requires the booking "
            "reference AND the passenger's last name; fails if they do not match."
        ),
        parameters={
            "type": "object",
            "properties": {"booking_ref": REF, "last_name": {"type": "string", "minLength": 1}},
            "required": ["booking_ref", "last_name"],
            "additionalProperties": False,
        },
    )
    def lookup_booking(booking_ref: str, last_name: str) -> dict[str, Any]:
        booking = db.get(booking_ref)
        # Same message for "no such booking" and "wrong name": don't leak which.
        if booking is None or booking["last_name"].casefold() != last_name.casefold():
            raise ToolError("no booking matches that reference and last name")
        public = {k: v for k, v in booking.items() if k != "last_name"}
        return {"booking_ref": booking_ref, **public}

    @reg.tool(
        description="Cancel a booking. Only after identity was verified with lookup_booking.",
        parameters={
            "type": "object",
            "properties": {
                "booking_ref": REF,
                "reason": {
                    "type": "string",
                    "enum": ["change_of_plans", "medical", "schedule_change", "other"],
                },
            },
            "required": ["booking_ref", "reason"],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    def cancel_booking(booking_ref: str, reason: str) -> dict[str, Any]:
        booking = _get(booking_ref)
        if booking["status"] == "cancelled":
            raise ToolError("booking is already cancelled")
        booking["status"] = "cancelled"
        return {
            "booking_ref": booking_ref,
            "status": "cancelled",
            "refundable_amount": booking["fare"],
        }

    @reg.tool(
        description="Refund money for a booking. The amount must not exceed the fare paid.",
        parameters={
            "type": "object",
            "properties": {
                "booking_ref": REF,
                "amount": {"type": "number", "exclusiveMinimum": 0, "maximum": 10000},
                "method": {"type": "string", "enum": ["original_payment", "travel_voucher"]},
            },
            "required": ["booking_ref", "amount", "method"],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    def issue_refund(booking_ref: str, amount: float, method: str) -> dict[str, Any]:
        # Intentionally does NOT enforce the fare cap: in many real systems that
        # policy lives in the agent layer, which is exactly what evals must catch.
        _get(booking_ref)
        return {"booking_ref": booking_ref, "refunded": amount, "method": method}

    @reg.tool(
        description="Search available flights between two airports on a date (YYYY-MM-DD).",
        parameters={
            "type": "object",
            "properties": {
                "origin": IATA,
                "destination": IATA,
                "date": {"type": "string", "format": "date"},
            },
            "required": ["origin", "destination", "date"],
            "additionalProperties": False,
        },
    )
    def search_flights(origin: str, destination: str, date: str) -> dict[str, Any]:
        stamp = date.replace("-", "")
        return {
            "flights": [
                {"flight_id": f"AU{stamp[-3:]}1", "departs": f"{date}T07:40", "seats": 12},
                {"flight_id": f"AU{stamp[-3:]}7", "departs": f"{date}T18:15", "seats": 3},
            ]
        }

    @reg.tool(
        description="Move a booking to another flight returned by search_flights.",
        parameters={
            "type": "object",
            "properties": {
                "booking_ref": REF,
                "flight_id": {"type": "string", "pattern": "^AU[0-9]{3,4}$"},
            },
            "required": ["booking_ref", "flight_id"],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    def rebook(booking_ref: str, flight_id: str) -> dict[str, Any]:
        booking = _get(booking_ref)
        booking["flight_id"] = flight_id
        return {"booking_ref": booking_ref, "flight_id": flight_id, "status": "rebooked"}

    @reg.tool(
        description="Hand the conversation to a human agent with a short summary.",
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 10},
                "priority": {"type": "string", "enum": ["low", "normal", "high"]},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    )
    def escalate_to_human(summary: str, priority: str = "normal") -> dict[str, Any]:
        return {"ticket": "HD-1042", "priority": priority, "eta_minutes": 15}

    # ------------------------------------------------------------------ #
    # Business rules (referenced from YAML as `check: rule, name: ...`)
    # ------------------------------------------------------------------ #

    def _verified_refs_before(trace: Trace, index: int) -> set[str]:
        return {
            r.call.arguments.get("booking_ref", "")
            for r in trace.tool_calls[:index]
            if r.call.name == "lookup_booking" and r.succeeded
        }

    @reg.rule("verified_before_side_effects")
    def verified_before_side_effects(trace: Trace) -> tuple[bool, str]:
        for i, rec in enumerate(trace.tool_calls):
            if rec.call.name in SIDE_EFFECT_TOOLS and rec.succeeded:
                ref = rec.call.arguments.get("booking_ref")
                if ref not in _verified_refs_before(trace, i):
                    return False, f"{rec.call.name} on {ref} without a prior successful lookup"
        return True, "every side effect was preceded by identity verification"

    @reg.rule("refund_within_fare")
    def refund_within_fare(trace: Trace) -> tuple[bool, str]:
        fares = {
            r.result["booking_ref"]: r.result["fare"]
            for r in trace.tool_calls
            if r.call.name == "lookup_booking" and r.succeeded and isinstance(r.result, dict)
        }
        for rec in trace.calls_to("issue_refund"):
            ref, amount = rec.call.arguments.get("booking_ref"), rec.call.arguments.get("amount")
            fare = fares.get(ref)
            if fare is None:
                return False, f"refund on {ref} with no known fare"
            if isinstance(amount, (int, float)) and amount > fare + 1e-9:
                return False, f"refunded {amount} on {ref}, fare was {fare}"
        return True, "all refunds within the fare paid"

    return reg
