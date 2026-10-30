"""Persist runs as JSON files: greppable, diffable, CI-artifact friendly."""

from __future__ import annotations

from pathlib import Path

from gauntlet.errors import ConfigError
from gauntlet.models import RunResult

RUNS_SUBDIR = "runs"


def runs_dir(root: Path) -> Path:
    return root / RUNS_SUBDIR


def save_run(run: RunResult, root: Path) -> Path:
    directory = runs_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run.run_id}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def list_runs(root: Path, suite: str | None = None) -> list[Path]:
    """Run files oldest-first (run ids start with a UTC timestamp)."""
    directory = runs_dir(root)
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob("*.json"))
    if suite is None:
        return paths
    return [p for p in paths if _suite_of(p) == suite]


def _suite_of(path: Path) -> str | None:
    try:
        return load_run_file(path).suite
    except ConfigError:
        return None


def load_run_file(path: Path) -> RunResult:
    try:
        return RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read run file {path}: {exc}") from exc


def resolve_run(ref: str, root: Path, suite: str | None = None) -> RunResult:
    """Resolve 'latest', 'previous', a run id, or a path to a run file."""
    if ref in ("latest", "previous"):
        runs = list_runs(root, suite)
        need = 1 if ref == "latest" else 2
        if len(runs) < need:
            raise ConfigError(f"no '{ref}' run found in {runs_dir(root)}")
        return load_run_file(runs[-need])
    candidate = Path(ref)
    if candidate.is_file():
        return load_run_file(candidate)
    by_id = runs_dir(root) / f"{ref}.json"
    if by_id.is_file():
        return load_run_file(by_id)
    raise ConfigError(f"run '{ref}' not found (looked for a file and in {runs_dir(root)})")
