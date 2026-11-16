"""Cancellation of a running tool call (timeout, Ctrl+C).

Python threads cannot be killed, so a tool that reaches its timeout has to be
*asked* to stop. Nearly all the time of a tool is spent inside DuckDB, and a
DuckDB query can be interrupted from another thread. Each tool call therefore
owns a :class:`CallScope`:

* :func:`track_connection` (called by ``catalog.connect``) registers every
  connection the call opens;
* :meth:`CallScope.cancel` interrupts the running queries of the call and of the
  calls nested in it (a delegated agent's tools), and marks the scope so that any
  connection opened afterwards fails at once with :class:`ToolCancelledError`;
* long pure-Python loops can call :func:`check_cancelled` between steps.

The scope travels in a context variable, copied into the worker thread of the call.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from typing import Any


class ToolCancelledError(RuntimeError):
    """The tool call was cancelled (timeout or interruption): stop working."""


class CallScope:
    """Connections and nested calls of one tool call."""

    def __init__(self, parent: CallScope | None = None) -> None:
        self.parent = parent
        self.cancelled = False
        self._lock = threading.Lock()
        self._connections: list[Any] = []
        self._children: list[CallScope] = []
        if parent is not None:
            parent._adopt(self)

    # ------------------------------------------------------------- structure
    def _adopt(self, child: CallScope) -> None:
        with self._lock:
            if self.cancelled:
                raise ToolCancelledError("The enclosing tool call was cancelled.")
            self._children.append(child)

    def detach(self) -> None:
        """Forget this finished scope (so a long-lived parent does not accumulate them)."""
        if self.parent is not None:
            with self.parent._lock, contextlib.suppress(ValueError):
                self.parent._children.remove(self)

    # ----------------------------------------------------------- connections
    def register(self, con: Any) -> None:
        with self._lock:
            if self.cancelled:
                with contextlib.suppress(Exception):
                    con.close()
                raise ToolCancelledError("The tool call was cancelled: no new query is started.")
            self._connections.append(con)

    def connections(self) -> list[Any]:
        with self._lock:
            return list(self._connections)

    def has_queries(self) -> bool:
        """Whether this call, or a call nested in it, opened a connection."""
        with self._lock:
            if self._connections:
                return True
            children = list(self._children)
        return any(child.has_queries() for child in children)

    # ---------------------------------------------------------------- cancel
    def cancel(self) -> None:
        """Mark the scope cancelled and interrupt every running query under it."""
        with self._lock:
            self.cancelled = True
            children = list(self._children)
        self.interrupt()
        for child in children:
            child.cancel()

    def interrupt(self) -> None:
        """Interrupt the running queries of this scope (a no-op on idle or closed connections)."""
        for con in self.connections():
            with contextlib.suppress(Exception):
                con.interrupt()

    def close_connections(self) -> None:
        """Close every connection of this scope and of its nested scopes.

        Only call it once the worker thread has ended: a connection must not be
        closed under a thread that is still using it.
        """
        with self._lock:
            connections, self._connections = self._connections, []
            children = list(self._children)
        for con in connections:
            with contextlib.suppress(Exception):
                con.close()
        for child in children:
            child.close_connections()


_scope: contextvars.ContextVar[CallScope | None] = contextvars.ContextVar("jellybean_call_scope", default=None)


def current_scope() -> CallScope | None:
    return _scope.get()


def set_scope(scope: CallScope | None) -> contextvars.Token:
    return _scope.set(scope)


def track_connection(con: Any) -> Any:
    """Register ``con`` with the tool call in progress, if any, and return it."""
    scope = _scope.get()
    if scope is not None:
        scope.register(con)
    return con


def check_cancelled() -> None:
    """Raise :class:`ToolCancelledError` if the tool call in progress was cancelled."""
    scope = _scope.get()
    if scope is not None and scope.cancelled:
        raise ToolCancelledError("The tool call was cancelled.")
