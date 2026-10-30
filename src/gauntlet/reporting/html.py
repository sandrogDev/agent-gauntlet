"""Self-contained HTML report (single file, no external assets, CI-artifact friendly)."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from gauntlet.models import RunResult
from gauntlet.regression import RunComparison


def _env() -> Environment:
    env = Environment(
        loader=PackageLoader("gauntlet.reporting", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"
    env.filters["tojson_pretty"] = lambda v: json.dumps(
        v, indent=2, ensure_ascii=False, default=str
    )
    return env


def render_html(run: RunResult, comparison: RunComparison | None = None) -> str:
    template = _env().get_template("report.html.j2")
    diffs = {d.scenario_id: d for d in comparison.diffs} if comparison else {}
    return template.render(run=run, m=run.metrics, comparison=comparison, diffs=diffs)


def write_html(run: RunResult, path: Path, comparison: RunComparison | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(run, comparison), encoding="utf-8")
    return path
