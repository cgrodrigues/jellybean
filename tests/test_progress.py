# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Live progress view: rendering of trace events, CLI integration, generator steps."""

from __future__ import annotations

import re
from pathlib import Path

from jellybean.runtime.progress import ConsoleProgress
from jellybean.runtime.tracing import add_listener, remove_listener, write_event

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _render(events: list[dict], verbose: bool = False) -> list[str]:
    lines: list[str] = []
    view = ConsoleProgress(verbose=verbose, echo=lambda line: lines.append(ANSI.sub("", line)))
    for event in events:
        view(event)
    return lines


EVENTS = [
    {"kind": "spec_update", "phase": "discovery"},
    {"kind": "agent_start", "agent": "profiler", "task": "Profile the datasets in the spec"},
    {
        "kind": "llm_call",
        "agent": "profiler",
        "iteration": 1,
        "input_tokens": 9000,
        "output_tokens": 200,
        "cache_read_tokens": 6000,
        "duration_s": 1.5,
        "tool_calls": 1,
    },
    {
        "kind": "tool_call",
        "agent": "profiler",
        "tool": "profile_table",
        "arguments": {"dataset": "traffic"},
        "status": "ok",
        "duration_s": 1.52,
        "artifact_id": "profile_traffic_v1",
    },
    {
        "kind": "tool_call",
        "agent": "profiler",
        "tool": "test_join",
        "arguments": {"left": "a", "right": "b"},
        "status": "error",
        "duration_s": 0.1,
        "error": "Binder Error: column b.key not found",
    },
    {"kind": "tool_call", "agent": "profiler", "tool": "finish", "arguments": {}, "status": "ok", "duration_s": 0},
    {"kind": "agent_end", "agent": "profiler", "status": "done", "iterations": 6},
    {"kind": "decision", "topic": "cleaning_approval", "choice": "Approve recommended plan", "by": "auto"},
    {"kind": "spec_update", "fields": ["datasets"], "phase": None},
]


def test_progress_lines() -> None:
    lines = _render(EVENTS)
    text = "\n".join(lines)
    assert "== Phase: discovery ==" in text
    assert "▶ profiler: Profile the datasets in the spec" in text
    assert "  · profile_table(dataset=traffic)  1.5s" in text  # indented under its agent
    assert "test_join(left=a, right=b)  0.1s  ✖ error" in text
    assert "    ↳ Binder Error: column b.key not found" in text  # message under the failed call
    assert "finish(" not in text  # internal calls are hidden
    assert "✔ profiler done — 6 turns, 9k tokens (65% cached)" in text
    assert "◆ cleaning_approval → Approve recommended plan (auto)" in text
    assert "LLM turn" not in text  # only in verbose mode
    assert all(re.match(r"\[\d\d:\d\d\] ", line) for line in lines)
    assert len([line for line in lines if "Phase" in line]) == 1  # updates without a phase are silent


def test_verbose_shows_llm_turns_and_artifacts() -> None:
    text = "\n".join(_render(EVENTS, verbose=True))
    assert "LLM turn 1: in 9k (cached 6k), out 200" in text
    assert "→ profile_traffic_v1" in text


def test_listener_receives_trace_events_and_errors_are_contained(tmp_path: Path) -> None:
    received: list[dict] = []

    def broken(event: dict) -> None:
        raise RuntimeError("display failure")

    add_listener(received.append)
    add_listener(broken)
    try:
        write_event(tmp_path / "trace.jsonl", {"kind": "init_run"})  # must not raise
    finally:
        remove_listener(received.append)
        remove_listener(broken)
    assert received and received[0]["kind"] == "init_run" and "ts" in received[0]
    assert (tmp_path / "trace.jsonl").read_text().count("\n") == 1


def test_generator_reports_its_steps(tmp_path: Path) -> None:
    from jellybean.datagen import GeneratorConfig, generate

    steps: list[str] = []
    generate(
        GeneratorConfig(out=tmp_path / "d", truth=tmp_path / "t", customers=1000, days=7, scale=0.1),
        progress=steps.append,
    )
    assert len(steps) >= 6 and any("traffic" in s for s in steps) and any("ground truth" in s for s in steps)


def test_outputs_listed_at_agent_end() -> None:
    events = [
        {"kind": "agent_start", "agent": "detector", "task": "detect"},
        {
            "kind": "artifact",
            "artifact_id": "detection_kmeans_v1",
            "type": "table",
            "rows": 20000,
            "path": "/abs/run/artifacts/detection_kmeans_v1.parquet",
        },
        {
            "kind": "artifact",
            "artifact_id": "plot_f001_embedding_v1",
            "type": "plot",
            "rows": None,
            "path": "/abs/run/artifacts/plot_f001_embedding_v1.png",
        },
        {"kind": "artifact", "artifact_id": "profile_x_v1", "type": "json", "rows": None, "path": "/abs/p.json"},
        {"kind": "agent_end", "agent": "detector", "status": "done", "iterations": 3},
    ]
    text = "\n".join(_render(events))
    assert "↳ outputs:" in text
    assert "- detection_kmeans_v1 (table, 20,000 rows)  /abs/run/artifacts/detection_kmeans_v1.parquet" in text
    assert "- plot_f001_embedding_v1 (plot)" in text
    assert "profile_x_v1" not in text and "+ 1 document(s)" in text  # documents summarised
    assert "profile_x_v1" in "\n".join(_render(events, verbose=True))  # listed in verbose mode


def test_failures_are_never_silent() -> None:
    lines = _render(
        [
            {"kind": "agent_start", "agent": "orquestrador", "task": "t"},
            # delegate is normally hidden, but not when it fails
            {
                "kind": "tool_call",
                "tool": "delegate",
                "arguments": {"agent": "profiler"},
                "status": "error",
                "duration_s": 0.2,
                "error": "KeyError: 'spec'",
            },
            {"kind": "tool_call", "tool": "run_sql", "arguments": {}, "status": "error", "duration_s": 0.1},
            {"kind": "llm_error", "error": "ConnectionError: network down"},
            {
                "kind": "agent_end",
                "agent": "orquestrador",
                "status": "failed",
                "iterations": 1,
                "reason": "LLM call failed: ConnectionError: network down",
            },
        ]
    )
    text = "\n".join(lines)
    assert "delegate(agent=profiler)  0.2s  ✖ error" in text and "↳ KeyError: 'spec'" in text
    assert "↳ no error message recorded" in text
    assert "✖ LLM error: ConnectionError: network down" in text
    assert "✖ orquestrador failed" in text and "↳ LLM call failed: ConnectionError: network down" in text
