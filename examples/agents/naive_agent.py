"""A deliberately naive, keyword-driven "agent" for offline demos and CI smoke tests.

This is NOT a language model. It exists so the example suite runs with zero API
keys and still produces a report with real, explainable findings. Its flaws are
the classic failure modes gauntlet is built to catch, on purpose:

1. obeys instructions embedded in user-controlled text (prompt injection)
2. refunds whatever amount the customer asks for (no fare cap)
3. skips identity verification when someone claims authority
4. passes natural-language dates straight to tools (schema violation)
5. gives up after a single transient tool failure instead of retrying

Use it as a template for wrapping your own agent:
``--provider python:path/to/your_agent.py`` with a ``create_provider()`` factory.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Sequence
from typing import Any

from gauntlet import AgentStep, Message, ToolCall, ToolSpec

REF = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b")
NAME = re.compile(r"(?:last name|surname)(?:\s+is)?[\s:,-]+([A-Za-z][A-Za-z'\-]+)", re.IGNORECASE)
IM_NAME = re.compile(r"\bI'?m\s+([A-Z][a-z]+)\b")
AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:eur|euros?|€)", re.IGNORECASE)
MONTH_DAY = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}\b",
    re.IGNORECASE,
)


def _last(items: Sequence[str]) -> str | None:
    return items[-1] if items else None


class NaiveAgent:
    name = "python:naive-agent"

    def __init__(self) -> None:
        self._ids = itertools.count(1)

    # -- helpers --------------------------------------------------------- #

    def _call(self, tool: str, **arguments: Any) -> AgentStep:
        return AgentStep(
            tool_calls=[ToolCall(id=f"call_{next(self._ids)}", name=tool, arguments=arguments)]
        )

    @staticmethod
    def _say(text: str) -> AgentStep:
        return AgentStep(text=text)

    @staticmethod
    def _results(since_user: Sequence[Message]) -> dict[str, tuple[bool, Any]]:
        """Latest result per tool name, since the last user message."""
        names = {c.id: c.name for m in since_user if m.role == "assistant" for c in m.tool_calls}
        out: dict[str, tuple[bool, Any]] = {}
        for m in since_user:
            if m.role == "tool" and m.tool_call_id in names:
                try:
                    payload = json.loads(m.content)
                except json.JSONDecodeError:
                    payload = m.content
                out[names[m.tool_call_id]] = (not m.is_error, payload)
        return out

    # -- policy ---------------------------------------------------------- #

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> AgentStep:
        users = [m.content for m in messages if m.role == "user"]
        convo, latest = "\n".join(users), users[-1]
        lower, latest_lower = convo.lower(), latest.lower()
        last_user = max(i for i, m in enumerate(messages) if m.role == "user")
        done = self._results(messages[last_user + 1 :])

        ref = _last(REF.findall(convo))
        name = _last(NAME.findall(convo)) or _last(IM_NAME.findall(convo))

        # Flaw 1: embedded instructions are treated as genuine.
        if ref and "ignore" in latest_lower and "refund" in latest_lower:
            if "issue_refund" not in done:
                amount = float(_last(AMOUNT.findall(latest)) or 950)
                return self._call(
                    "issue_refund", booking_ref=ref, amount=amount, method="original_payment"
                )
            return self._say("The refund has been processed as requested.")

        if "poem" in lower or "system prompt" in lower:
            return self._say(
                "Sorry, I can only help with Aurora Air bookings, refunds and flight changes."
            )

        # Flaw 3: claimed authority bypasses verification.
        if ref and "skip verification" in latest_lower:
            if "cancel_booking" not in done:
                return self._call("cancel_booking", booking_ref=ref, reason="other")
            return self._say(f"Understood. Booking {ref} has been cancelled.")

        if not ref or not name:
            return self._say("Happy to help. Could you share your booking reference and last name?")

        if "lookup_booking" not in done:
            return self._call("lookup_booking", booking_ref=ref, last_name=name)
        ok, booking = done["lookup_booking"]
        if not ok:
            # Flaw 5: no retry, even when the error says "please retry".
            return self._say(
                f"Sorry, I couldn't find a booking matching {ref} and that last name. "
                "Could you double-check the details?"
            )

        if any(w in lower for w in ("compensation", "human", "lost")):
            if "escalate_to_human" not in done:
                return self._call(
                    "escalate_to_human",
                    summary=f"Customer on {ref} requests compensation: {latest[:160]}",
                    priority="normal",
                )
            ticket = done["escalate_to_human"][1].get("ticket", "")
            return self._say(
                f"I've passed this to a colleague (ticket {ticket}). They'll contact you shortly."
            )

        if "refund" in lower:
            if "issue_refund" not in done:
                # Flaw 2: trusts the requested amount instead of capping at the fare.
                amount = float(_last(AMOUNT.findall(convo)) or booking["fare"])
                return self._call(
                    "issue_refund", booking_ref=ref, amount=amount, method="original_payment"
                )
            return self._say("Your refund has been processed to the original payment method.")

        if "cancel" in lower:
            if "cancel_booking" not in done:
                return self._call("cancel_booking", booking_ref=ref, reason="change_of_plans")
            ok, payload = done["cancel_booking"]
            if ok:
                return self._say(
                    f"Booking {ref} has been cancelled. A refund of "
                    f"{payload['refundable_amount']} EUR is on its way."
                )
            return self._say(f"I couldn't cancel booking {ref}: {payload.get('error')}")

        if any(w in lower for w in ("change", "move", "rebook")):
            # Flaw 4: forwards "December 5" verbatim instead of 2026-12-05.
            phrase = MONTH_DAY.search(convo)
            date = phrase.group(0) if phrase else booking["date"]
            if "search_flights" not in done:
                return self._call(
                    "search_flights",
                    origin=booking["origin"],
                    destination=booking["destination"],
                    date=date,
                )
            ok, found = done["search_flights"]
            if not ok:
                return self._say(
                    "Sorry, I couldn't search for flights right now. Please try again later."
                )
            flights = found["flights"]
            if any(w in latest_lower for w in ("morning", "earliest", "first", " one")):
                if "rebook" not in done:
                    return self._call("rebook", booking_ref=ref, flight_id=flights[0]["flight_id"])
                return self._say(f"Done: booking {ref} is now on flight {flights[0]['flight_id']}.")
            options = ", ".join(f"{f['flight_id']} at {f['departs'][-5:]}" for f in flights)
            return self._say(f"I found these flights: {options}. Which one would you like?")

        return self._say(
            f"Booking {ref}: {booking['origin']} to {booking['destination']} on {booking['date']}, "
            f"status {booking['status']}."
        )


def create_provider() -> NaiveAgent:
    return NaiveAgent()
