# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Run heavy numeric work in a child process that can be killed at the timeout.

A tool's timeout interrupts its DuckDB queries, but a model fit (scikit-learn,
pyod, umap) is one long call into compiled code: it cannot be interrupted, and
while it holds the interpreter lock the caller cannot even notice that the
time is up. Such work therefore runs in a separate Python process:

* the caller thread only waits for the process, so the timeout fires on time;
* on timeout, cancellation or Ctrl+C the process is killed and its memory freed;
* the same function runs with the same arguments and seed as in-process, so
  the results are identical.

Arrays are exchanged as ``.npy`` files in a scratch folder (the child maps the
inputs copy-on-write); nothing large is pickled through a pipe.

    scores = run_isolated("pkg.module:function", {"X": matrix}, {"seed": 7})["result"]

The target function receives the arrays and the keyword arguments by name and
returns one array or a dict of arrays.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from jellybean.runtime.cancellation import ToolCancelledError, current_scope
from jellybean.runtime.context import maybe_current_run

CHILD_ENV = "JELLYBEAN_ISOLATED_CHILD"
_POLL_S = 0.1


class IsolatedWorkError(RuntimeError):
    """The work failed in the child process. ``kind`` is the name of the exception raised there."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def in_child() -> bool:
    """Whether this process is a worker started by :func:`run_isolated`."""
    return os.environ.get(CHILD_ENV) == "1"


def should_isolate(rows: int) -> bool:
    """Whether work on ``rows`` rows goes to a child process (``limits.isolate_above_rows``)."""
    if in_child():
        return False
    from jellybean.settings import get_settings

    run = maybe_current_run()
    threshold = (run.settings if run else get_settings()).config.limits.isolate_above_rows
    return threshold is not None and rows > threshold


def _scratch() -> Path:
    run = maybe_current_run()
    base = run.tmp_dir if run else Path(tempfile.gettempdir()) / "jellybean-isolated"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="work_", dir=base))


def run_isolated(target: str, arrays: dict[str, np.ndarray], kwargs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Call ``target`` ("module:function") in a child process and return its arrays.

    Raises:
        ToolCancelledError: The tool call was cancelled (timeout); the child was killed.
        IsolatedWorkError: The function raised, or the child died (e.g. out of memory).
    """
    scope = current_scope()
    folder = _scratch()
    process: subprocess.Popen[bytes] | None = None
    try:
        for name, array in arrays.items():
            np.save(folder / f"in_{name}.npy", np.ascontiguousarray(array), allow_pickle=False)
        (folder / "call.json").write_text(json.dumps({"target": target, "arrays": sorted(arrays), "kwargs": kwargs}))
        env = os.environ | {CHILD_ENV: "1"}
        with open(folder / "stderr.txt", "wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-m", "jellybean.runtime.isolated", str(folder)],
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                env=env,
            )
        if scope is not None:
            scope.register_process(process)
        while process.poll() is None:
            if scope is not None and scope.cancelled:
                raise ToolCancelledError("The tool call was cancelled: its worker process was stopped.")
            time.sleep(_POLL_S)
        if scope is not None and scope.cancelled:
            raise ToolCancelledError("The tool call was cancelled: its worker process was stopped.")
        error = folder / "error.json"
        if error.exists():
            info = json.loads(error.read_text())
            raise IsolatedWorkError(info["kind"], info["message"])
        if process.returncode != 0 or not (folder / "out.npz").exists():
            tail = (folder / "stderr.txt").read_text(errors="replace")[-300:].strip()
            reason = "killed (out of memory?)" if process.returncode in (-9, 137) else f"exit code {process.returncode}"
            raise IsolatedWorkError("WorkerDied", f"The worker process ended without a result: {reason}. {tail}")
        with np.load(folder / "out.npz", allow_pickle=False) as out:
            return {name: out[name] for name in out.files}
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            with contextlib.suppress(Exception):
                process.wait(5)
        shutil.rmtree(folder, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Child side                                                                  #
# --------------------------------------------------------------------------- #
def _die_with_parent() -> None:
    """Ask the kernel to kill this process when its parent dies (Linux), so no worker outlives the run."""
    with contextlib.suppress(Exception):
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() == 1:  # the parent was already gone
        os._exit(1)


def _child_main(folder: Path) -> int:
    _die_with_parent()
    call = json.loads((folder / "call.json").read_text())
    try:
        module, function = call["target"].split(":")
        fn = getattr(importlib.import_module(module), function)
        arrays = {name: np.load(folder / f"in_{name}.npy", mmap_mode="c") for name in call["arrays"]}
        result = fn(**arrays, **call["kwargs"])
        if not isinstance(result, dict):
            result = {"result": result}
        np.savez(folder / "out.npz", **{k: np.asarray(v) for k, v in result.items()})
    except BaseException as exc:
        (folder / "error.json").write_text(json.dumps({"kind": type(exc).__name__, "message": str(exc)}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_child_main(Path(sys.argv[1])))
