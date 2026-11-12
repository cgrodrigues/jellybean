"""Run context: the state every tool needs to know which run it belongs to.

Tools never receive paths or seeds from the LLM for infrastructure concerns.
Instead the runtime activates a :class:`RunContext` (stored in a
:mod:`contextvars` variable, so it is safe across threads started with a copied
context) and tools read it with :func:`current_run`.
"""

from __future__ import annotations

import contextvars
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from jellybean.settings import Settings

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")


@dataclass(frozen=True)
class RunContext:
    """Immutable description of the active run.

    Attributes:
        run_id: Identifier of the run, also the name of its workspace folder.
        run_dir: Absolute path of ``workspace/<run_id>``.
        seed: Seed propagated to every tool with randomness.
        settings: Resolved settings for this run.
        agent: Name of the agent currently executing (for lineage/tracing).
    """

    run_id: str
    run_dir: Path
    seed: int
    settings: Settings
    agent: str = "system"
    # Row filters per dataset ({"traffic": "date >= '2026-10-01'"}), applied to every
    # dataset view of the run (set with `jellybean run --filter`).
    filters: dict[str, str] = field(default_factory=dict)

    # Well-known locations inside the run workspace (section 6 of the spec).
    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def findings_dir(self) -> Path:
        return self.run_dir / "findings"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def db_path(self) -> Path:
        return self.run_dir / "run.duckdb"

    @property
    def spec_path(self) -> Path:
        return self.run_dir / "spec.yaml"

    @property
    def trace_path(self) -> Path:
        return self.logs_dir / "trace.jsonl"

    def for_agent(self, agent: str) -> RunContext:
        """Return a copy of this context attributed to another agent."""
        return replace(self, agent=agent)

    @classmethod
    def create(
        cls, settings: Settings, seed: int, run_id: str | None = None, filters: dict[str, str] | None = None
    ) -> RunContext:
        """Create the workspace folders of a new run and return its context."""
        root = settings.workspace_root
        root.mkdir(parents=True, exist_ok=True)
        run_id = run_id or next_run_id(root)
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"Invalid run id: {run_id!r}")
        run_dir = root / run_id
        for sub in ("artifacts", "findings", "logs"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        return cls(run_id=run_id, run_dir=run_dir, seed=seed, settings=settings, filters=dict(filters or {}))

    @classmethod
    def open(cls, settings: Settings, run_id: str, seed: int, filters: dict[str, str] | None = None) -> RunContext:
        """Re-open an existing run workspace (used by ``resume``)."""
        run_dir = settings.workspace_root / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run workspace not found: {run_dir}")
        return cls(run_id=run_id, run_dir=run_dir, seed=seed, settings=settings, filters=dict(filters or {}))


def next_run_id(workspace_root: Path, today: date | None = None) -> str:
    """Return the next free id of the form ``YYYY-MM-DD_NNN``."""
    prefix = (today or date.today()).isoformat()
    existing = [p.name for p in workspace_root.glob(f"{prefix}_*") if p.is_dir()]
    numbers = [int(n.rsplit("_", 1)[1]) for n in existing if n.rsplit("_", 1)[1].isdigit()]
    return f"{prefix}_{max(numbers, default=0) + 1:03d}"


_current_run: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar("jellybean_current_run", default=None)


def current_run() -> RunContext:
    """Return the active run, raising a clear error when there is none."""
    ctx = _current_run.get()
    if ctx is None:
        raise RuntimeError("No active run. Tools must execute inside `use_run(...)`.")
    return ctx


def maybe_current_run() -> RunContext | None:
    """Return the active run or ``None``."""
    return _current_run.get()


@contextmanager
def use_run(ctx: RunContext) -> Iterator[RunContext]:
    """Activate ``ctx`` for the duration of the ``with`` block."""
    token = _current_run.set(ctx)
    try:
        yield ctx
    finally:
        _current_run.reset(token)
