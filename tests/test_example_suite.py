"""End-to-end: the bundled example suite against the bundled naive agent.

These double as a regression net for the examples shown in the README: if a
code change alters what the demo reports, this test says so.
"""

from __future__ import annotations

import json

import pytest

from conftest import EXAMPLES
from gauntlet.cli import EXIT_CONFIG, EXIT_GATE, EXIT_OK, main

SUITE = str(EXAMPLES / "suites" / "airline_support.yaml")
AGENT = f"python:{EXAMPLES / 'agents' / 'naive_agent.py'}"

pytestmark = pytest.mark.integration


def _run(tmp_path, *extra: str) -> int:
    return main(["--out-dir", str(tmp_path / "store"), "run", SUITE, "-p", AGENT, *extra])


def test_validate_command(capsys) -> None:
    assert main(["validate", SUITE]) == EXIT_OK
    assert "9 scenarios" in capsys.readouterr().out


def test_demo_run_finds_the_planted_flaws(tmp_path) -> None:
    out_json = tmp_path / "run.json"
    html = tmp_path / "report.html"
    assert _run(tmp_path, "--json", str(out_json), "--html", str(html)) == EXIT_OK

    run = json.loads(out_json.read_text())
    results = {r["scenario"]["id"]: r for r in run["results"]}

    def failed(sid: str, label_fragment: str) -> bool:
        return any(not c["passed"] and label_fragment in c["label"] for c in results[sid]["checks"])

    # Flaw 2: refund not capped at the fare
    assert failed("refund-capped", "refund_within_fare")
    # Flaw 4: natural-language date sent to a date-typed tool
    assert failed("rebook-iso-date", "schema_valid")
    # Flaw 1: obeys injected instructions
    assert failed("identity-mismatch~prompt_injection", "must not call issue_refund")
    # Flaw 3: skips verification under authority pressure
    assert failed("identity-mismatch~authority_pressure", "verified_before_side_effects")
    # Flaw 5: no retry after a transient failure
    assert failed("cancel-basic~tool_failure", "tool_called(cancel_booking)")
    # ...while the well-behaved paths pass
    for sid in ("cancel-basic", "identity-mismatch", "off-topic", "multi-turn-change"):
        assert results[sid]["passed"], sid

    # judge checks are skipped (not failed) when no judge is configured
    leak = results["no-system-prompt-leak"]
    assert any(c["check"] == "judge" and c["skipped"] for c in leak["checks"])

    page = html.read_text()
    assert "airline-support" in page and "Robustness by attack" in page


def test_quality_gates(tmp_path) -> None:
    assert _run(tmp_path, "--min-pass-rate", "0.99") == EXIT_GATE
    assert _run(tmp_path, "--fail-on-critical") == EXIT_GATE
    # Same agent, same seed: no regressions against the previous run.
    assert _run(tmp_path, "--baseline", "latest", "--fail-on-regression") == EXIT_OK


def test_filters_and_config_errors(tmp_path, capsys) -> None:
    assert _run(tmp_path, "--no-adversarial", "-t", "scope") == EXIT_OK
    assert "2 scenarios" in capsys.readouterr().out
    assert _run(tmp_path, "-t", "no-such-tag") == EXIT_CONFIG
    assert main(["run", "missing.yaml"]) == EXIT_CONFIG


def test_report_compare_history(tmp_path, capsys) -> None:
    store = str(tmp_path / "store")
    _run(tmp_path)
    _run(tmp_path, "--seed", "7")
    capsys.readouterr()
    assert main(["--out-dir", store, "compare", "previous", "latest"]) == EXIT_OK
    assert "Comparison" in capsys.readouterr().out
    assert main(["--out-dir", store, "report", "--html", str(tmp_path / "r.html")]) == EXIT_OK
    assert main(["--out-dir", store, "history"]) == EXIT_OK
    assert len(capsys.readouterr().out.strip().splitlines()) >= 3
    assert main(["scenarios", SUITE]) == EXIT_OK
