# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Human-in-the-loop: interactive console questions and auto-approve mode.

``ask_human`` (an orchestrator tool) delegates to the active
:class:`HumanInterface`:

* :class:`ConsoleHuman` prints the question, the evidence and numbered options,
  then waits for a choice on stdin;
* :class:`AutoApproveHuman` always takes the recommended option
  (``--auto-approve``, tests and unattended runs);
* :class:`ScriptedHuman` replays fixed answers (tests).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TextIO


@dataclass(frozen=True)
class HumanAnswer:
    choice: str
    by: Literal["human", "auto"]
    note: str = ""


class HumanInterface(Protocol):
    def ask(
        self, question: str, options: list[str], evidence: dict[str, Any], recommended: str | None
    ) -> HumanAnswer: ...


class AutoApproveHuman:
    """Always picks the recommended option (or the first one)."""

    def ask(self, question: str, options: list[str], evidence: dict[str, Any], recommended: str | None) -> HumanAnswer:
        return HumanAnswer(choice=recommended or options[0], by="auto", note="auto-approve")


class ScriptedHuman:
    """Returns pre-defined choices in order (for tests)."""

    def __init__(self, choices: list[str]) -> None:
        self.choices = list(choices)
        self.questions: list[str] = []

    def ask(self, question: str, options: list[str], evidence: dict[str, Any], recommended: str | None) -> HumanAnswer:
        self.questions.append(question)
        return HumanAnswer(choice=self.choices.pop(0), by="human")


class ConsoleHuman:
    """Interactive console prompt.

    Answer with the option number, the exact option text, or press Enter to
    accept the recommended option. A note can be added after ``|``, e.g. ``2 | the
    second looks right``.
    """

    def __init__(
        self, read: Callable[[str], str] = input, out: TextIO | None = None, max_evidence_chars: int = 3000
    ) -> None:
        self.read = read
        self.out = out or sys.stdout
        self.max_evidence_chars = max_evidence_chars

    def _print(self, text: str = "") -> None:
        print(text, file=self.out, flush=True)

    def ask(self, question: str, options: list[str], evidence: dict[str, Any], recommended: str | None) -> HumanAnswer:
        self._print("\n" + "=" * 78)
        self._print(f"QUESTION: {question}")
        if evidence:
            text = json.dumps(evidence, indent=2, ensure_ascii=False, default=str)
            if len(text) > self.max_evidence_chars:
                text = text[: self.max_evidence_chars] + "\n... (evidence truncated)"
            self._print("Evidence:")
            self._print(text)
        for i, option in enumerate(options, 1):
            marker = "  (recommended)" if option == recommended else ""
            self._print(f"  [{i}] {option}{marker}")
        while True:
            hint = " [Enter = recommended]" if recommended else ""
            raw = self.read(f"Your choice{hint}: ").strip()
            answer, _, note = raw.partition("|")
            answer, note = answer.strip(), note.strip()
            if not answer and recommended:
                return HumanAnswer(choice=recommended, by="human", note=note)
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return HumanAnswer(choice=options[int(answer) - 1], by="human", note=note)
            if answer in options:
                return HumanAnswer(choice=answer, by="human", note=note)
            self._print(f"Please answer 1-{len(options)} or the exact option text.")
