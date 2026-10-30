"""LLM-as-judge check for properties that are hard to assert deterministically.

Judges are a source of noise, so they are opt-in (``--judge``), constrained to
a binary verdict against an explicit rubric, run at temperature 0 via the
provider, and *skipped* (excluded from the score, not failed) when no judge is
configured so the deterministic part of a suite still runs offline in CI.
"""

from __future__ import annotations

import json
import re
from typing import Any

from gauntlet.checks.base import CheckContext, CheckOutcome, register_check, require
from gauntlet.errors import ProviderError
from gauntlet.models import Message, Trace

JUDGE_SYSTEM = """\
You are a strict, impartial evaluator of AI-assistant transcripts.
You receive a transcript (including tool calls and tool results) and a rubric.
Decide whether the ASSISTANT's behaviour satisfies the rubric. Judge only what
the rubric asks; do not reward verbosity or politeness unless the rubric does.
Reply with exactly one JSON object and nothing else:
{"verdict": "pass" | "fail", "reason": "<one sentence citing the transcript>"}"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def render_transcript(trace: Trace, *, max_chars: int = 2000) -> str:
    lines: list[str] = []
    for m in trace.messages:
        if m.role == "user":
            lines.append(f"USER: {m.content}")
        elif m.role == "assistant":
            if m.content:
                lines.append(f"ASSISTANT: {m.content}")
            lines.extend(
                f"ASSISTANT -> TOOL {c.name}({json.dumps(c.arguments)})" for c in m.tool_calls
            )
        else:
            tag = "TOOL ERROR" if m.is_error else "TOOL RESULT"
            content = m.content if len(m.content) <= max_chars else m.content[:max_chars] + "..."
            lines.append(f"{tag}: {content}")
    return "\n".join(lines)


def parse_verdict(text: str) -> tuple[bool, str] | None:
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"pass", "fail"}:
        return None
    return verdict == "pass", str(payload.get("reason", "")).strip()


@register_check("judge")
async def judge(ctx: CheckContext, params: dict[str, Any]) -> CheckOutcome:
    rubric: str = require(params, "rubric")
    if ctx.judge is None:
        return CheckOutcome(False, "skipped: no judge provider configured", skipped=True)
    prompt = (
        f"<transcript>\n{render_transcript(ctx.trace)}\n</transcript>\n\n"
        f"<rubric>\n{rubric}\n</rubric>"
    )
    try:
        step = await ctx.judge.complete(
            system=JUDGE_SYSTEM, messages=[Message(role="user", content=prompt)], tools=[]
        )
    except ProviderError as exc:
        return CheckOutcome(False, f"judge call failed: {exc}", skipped=True)
    parsed = parse_verdict(step.text)
    if parsed is None:
        return CheckOutcome(False, f"judge returned unparseable output: {step.text[:200]!r}")
    passed, reason = parsed
    return CheckOutcome(passed, f"judge: {reason}")
