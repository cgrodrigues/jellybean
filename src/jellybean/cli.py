# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Command line interface (``jellybean ...``).

Commands: ``generate-data``, ``datasets``, ``snapshot``, ``snapshots``, ``run``,
``resume``, ``report``, ``feedback``, ``outputs``, ``ui`` and ``evaluate``.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer

from jellybean.settings import get_settings, load_env_file

app = typer.Typer(
    name="jellybean",
    help="Multi-agent anomaly detection over Parquet datasets.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("generate-data")
def generate_data(
    out: Annotated[Path | None, typer.Option(help="Data folder (default: JELLYBEAN_DATA_ROOT).")] = None,
    truth: Annotated[
        Path | None, typer.Option(help="Ground-truth folder (default: JELLYBEAN_GROUND_TRUTH_ROOT).")
    ] = None,
    data_type: Annotated[
        str,
        typer.Option(
            "--type",
            help="Data to generate: 'telecom' (traffic, cust_segmentation, orders; default) or "
            "'telemetry' (industrial machines: telemetry, machines).",
        ),
    ] = "telecom",
    seed: Annotated[int, typer.Option(help="Random seed; same seed gives identical data.")] = 42,
    customers: Annotated[int, typer.Option(min=1000, help="telecom: number of customers.")] = 20_000,
    machines: Annotated[int, typer.Option(min=100, help="telemetry: number of machines.")] = 1500,
    days: Annotated[
        int | None, typer.Option(min=7, help="Number of days (default: 30 for telecom, 14 for telemetry).")
    ] = None,
    with_readme: Annotated[bool, typer.Option("--with-readme", help="Write a README.md per dataset.")] = False,
    scale: Annotated[float, typer.Option(min=0.01, help="Volume multiplier for calls and orders.")] = 1.0,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace existing generated datasets.")] = False,
    partition: Annotated[
        str, typer.Option("--partition", help="traffic partitioning: 'hour' (date=/hour=, default) or 'day'.")
    ] = "hour",
    datasets: Annotated[
        str | None,
        typer.Option(
            "--datasets",
            help="telecom: comma-separated scenarios among traffic, cust_segmentation, orders, topup "
            "(default: all; topup also writes promotions, dealers and voucher_batches).",
        ),
    ] = None,
    topup_days: Annotated[int, typer.Option("--topup-days", min=60, help="telecom: days of top-up movements.")] = 90,
    opaque_codes: Annotated[
        bool,
        typer.Option("--opaque-codes", help="telecom: numeric codes for topup movement_type, channel, reason_code."),
    ] = False,
) -> None:
    """Generate synthetic datasets with planted anomalies and their ground truth.

    ``--type telecom`` (default) writes traffic, cust_segmentation and orders;
    ``--type telemetry`` writes an industrial-machine dataset from another domain
    (telemetry, machines), used to check that the system is domain-agnostic.
    """
    if data_type not in ("telecom", "telemetry"):
        typer.secho("--type must be 'telecom' or 'telemetry'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    if partition not in ("day", "hour"):
        typer.secho("--partition must be 'day' or 'hour'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    telecom_only = partition != "hour" or scale != 1.0 or customers != 20_000 or datasets or topup_days != 90
    if data_type == "telemetry" and (telecom_only or opaque_codes):
        typer.secho(
            "--partition, --scale, --customers, --datasets, --topup-days and --opaque-codes apply to "
            "--type telecom only",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    from jellybean.datagen.generator import SCENARIOS

    scenarios = tuple(d.strip() for d in (datasets or ",".join(SCENARIOS)).split(",") if d.strip())
    unknown = sorted(set(scenarios) - set(SCENARIOS))
    if unknown or not scenarios:
        typer.secho(f"--datasets: unknown {unknown}; choose from {list(SCENARIOS)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    settings = get_settings()
    out, truth = out or settings.data_root, truth or settings.ground_truth_root
    typer.echo(f"Generating {data_type} data in {out} (ground truth in {truth}) ...")
    started = time.monotonic()

    def progress(message: str) -> None:
        elapsed = int(time.monotonic() - started)
        typer.echo(f"[{elapsed // 60:02d}:{elapsed % 60:02d}] {message}")

    try:
        # Generators are imported lazily so other commands start fast.
        if data_type == "telemetry":
            from jellybean.datagen.telemetry import DATASETS, TRUTH_FILE, generate_telemetry

            result = generate_telemetry(
                out, truth, seed, machines, days or 14, overwrite, with_readme, progress=progress
            )
            summary = {
                "type": "telemetry",
                "datasets": {"telemetry": result["rows"], "machines": machines},
                "anomalies": {k: len(v) for k, v in result["anomalies"].items()},
                "out": str(out.resolve()),
                "truth": str((truth / TRUTH_FILE).resolve()),
                "note": f"datasets {list(DATASETS)}; `jellybean evaluate` supports telecom data only",
            }
        else:
            from jellybean.datagen import GeneratorConfig, generate

            cfg = GeneratorConfig(
                out=out,
                truth=truth,
                seed=seed,
                customers=customers,
                days=days or 30,
                with_readme=with_readme,
                scale=scale,
                overwrite=overwrite,
                partition=partition,  # type: ignore[arg-type]
                datasets=scenarios,
                topup_days=topup_days,
                opaque_codes=opaque_codes,
            )
            summary = generate(cfg, progress=progress)
    except (FileExistsError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(summary, indent=2))


@app.command("datasets")
def datasets() -> None:
    """List the datasets found in the data root (same view the agents get)."""
    from jellybean.skills._shared.data_tools import list_datasets

    result = list_datasets()
    typer.echo(json.dumps(result.model_dump(), indent=2, default=str))


@app.callback()
def _main() -> None:
    """Load ``.env`` into the environment before any command (AWS_* for Bedrock)."""
    load_env_file()


def _runtime(ctx, auto_approve: bool, options: dict):
    """Build the runtime (LLM client + human interface) or exit with a clear error."""
    from jellybean.runtime.agent_runner import Runtime
    from jellybean.runtime.human import AutoApproveHuman, ConsoleHuman
    from jellybean.runtime.llm_client import LLMConfigurationError, build_llm_client

    try:
        llm = build_llm_client(ctx.settings)
    except LLMConfigurationError as exc:
        typer.secho(f"LLM configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    human = AutoApproveHuman() if auto_approve else ConsoleHuman()
    return Runtime(ctx=ctx, llm=llm, human=human, options=options)


@contextmanager
def _progress(quiet: bool, verbose: bool):
    """Print the run's progress live (trace listener) unless --quiet."""
    from jellybean.runtime.progress import ConsoleProgress
    from jellybean.runtime.tracing import add_listener, remove_listener

    if quiet:
        yield
        return
    listener = ConsoleProgress(verbose=verbose)
    add_listener(listener)
    try:
        yield
    finally:
        remove_listener(listener)


@contextmanager
def _interruptible(ctx):
    """Make Ctrl+C return the prompt at once.

    The interruption has already stopped the running queries (see
    ``tool_registry._run_with_timeout``). Agents in worker threads may still be
    waiting for an LLM answer; the process exits without waiting for them, after
    removing the run's scratch files. The run can be continued with ``resume``.
    """
    try:
        yield
    except KeyboardInterrupt:
        import os
        import shutil
        import sys

        shutil.rmtree(ctx.tmp_dir, ignore_errors=True)
        typer.secho(f"\nInterrupted. Continue with: jellybean resume {ctx.run_id}", fg=typer.colors.YELLOW, err=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)


def _finish(runtime, result) -> None:
    """Persist the orchestrator result, (re)write report.md and print the outcome."""
    from jellybean.reporting import write_report_file
    from jellybean.runtime.agent_runner import run_outputs

    ctx = runtime.ctx
    outputs = run_outputs(ctx)
    payload = json.loads(result.model_dump_json()) | {"outputs_by_agent": _group_outputs(outputs)}
    (ctx.run_dir / "run_result.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report = write_report_file(ctx) if ctx.spec_path.exists() else None
    color = typer.colors.GREEN if result.status == "done" else typer.colors.YELLOW
    typer.secho(f"\nStatus: {result.status}", fg=color)
    typer.echo(result.summary)
    if result.issues:
        typer.echo("Issues: " + "; ".join(result.issues))
    _print_outputs(ctx, outputs)
    typer.echo(f"Tokens (budget units): {runtime.budget.used}  |  report: {report or 'not written (no spec.yaml)'}")
    if result.status == "failed":
        raise typer.Exit(code=1)


def _group_outputs(outputs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for o in outputs:
        grouped.setdefault(o["agent"], []).append({k: v for k, v in o.items() if k != "agent"})
    return grouped


def _print_outputs(ctx, outputs: list[dict], kind: str | None = None) -> None:
    """Print artifacts grouped by agent, each with its (clickable) path.

    Without ``kind``, datasets (tables) and plots are listed and documents are counted.
    """
    from jellybean.runtime.progress import display_path

    if kind is None:
        documents = sum(o["type"] not in ("table", "plot") for o in outputs)
        shown = [o for o in outputs if o["type"] in ("table", "plot")]
    else:
        documents, shown = 0, [o for o in outputs if o["type"] == kind]
    typer.secho(f"\nOutputs ({len(shown)}):", bold=True)
    for agent, items in _group_outputs(shown).items():
        typer.echo(f"  {agent}:")
        for o in items:
            rows = f", {o['rows']:,} rows" if isinstance(o.get("rows"), int) else ""
            typer.echo(f"    - {o['id']} ({o['type']}{rows})  {display_path(str(ctx.run_dir / o['path']))}")
    if documents:
        typer.echo(f"  + {documents} document(s): jellybean outputs {ctx.run_id} --type json")


def _open_run(run_id: str, need_data: bool = False):
    from jellybean.runtime.context import RunContext
    from jellybean.spec import load_spec

    settings = get_settings()
    path = settings.workspace_root / run_id / "spec.yaml"
    if not path.exists():
        typer.secho(f"Run '{run_id}' not found (no {path}).", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    spec = load_spec(path)
    source = None
    if spec.source:
        from jellybean.sources.snapshot import snapshot_dir

        # The run read its datasets from a snapshot: point the data root at it again.
        folder = snapshot_dir(settings, spec.source.name, spec.source.snapshot)
        if not folder.is_dir() and need_data:
            typer.secho(
                f"Run '{run_id}' used snapshot {spec.source.snapshot} of source '{spec.source.name}', which is "
                f"no longer in {folder.parent}. Take it again with `jellybean snapshot` (same window and sample).",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        settings, source = settings.with_data_root(folder), spec.source.model_dump()
    return RunContext.open(settings, run_id, seed=spec.seed, filters=spec.filters, source=source), spec


def _progress_line(started: float):
    def progress(message: str) -> None:
        elapsed = int(time.monotonic() - started)
        typer.echo(f"[{elapsed // 60:02d}:{elapsed % 60:02d}] {message}")

    return progress


def _take_snapshot(settings, source: str, start: str | None, end: str | None, sample: float | None, refresh: bool):
    """Copy (or reuse) the snapshot of a source, or exit with a clear error."""
    from jellybean.sources.snapshot import SnapshotError, make_request, take_snapshot

    try:
        request = make_request(start, end, sample)
        snapshot = take_snapshot(settings, source, request, refresh=refresh, progress=_progress_line(time.monotonic()))
    except SnapshotError as exc:
        typer.secho(f"Source error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    rows = sum(t.rows for t in snapshot.manifest.tables.values())
    state = "reused" if snapshot.reused else "taken"
    typer.echo(f"Snapshot {state}: {snapshot.manifest.describe()} -> {rows:,} rows in {snapshot.path}")
    return snapshot


_SOURCE_HELP = (
    "Read the datasets from this source of config.yaml (copied to a local snapshot) instead of the data root."
)
_FROM_HELP = "Start of the window, e.g. 2026-10-01 (applies to the tables with a time_column)."
_TO_HELP = "End of the window, excluded, e.g. 2026-11-01."
_SAMPLE_HELP = "Share of the entities to copy, e.g. 0.1; every row of a sampled entity is kept."
_REFRESH_HELP = "Copy again even if this snapshot already exists."


@app.command("snapshot")
def snapshot(
    source: Annotated[str, typer.Argument(help="Source name, as in the `sources` section of config.yaml.")],
    start: Annotated[str | None, typer.Option("--from", help=_FROM_HELP)] = None,
    end: Annotated[str | None, typer.Option("--to", help=_TO_HELP)] = None,
    sample: Annotated[float | None, typer.Option("--sample", help=_SAMPLE_HELP)] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help=_REFRESH_HELP)] = False,
) -> None:
    """Copy the tables of an external source to a local snapshot, without running anything.

    The source filters by window, sample and columns; the result is a folder of
    Parquet datasets that ``jellybean run --source`` (same options) reuses.
    """
    result = _take_snapshot(get_settings(), source, start, end, sample, refresh)
    typer.echo(json.dumps(result.manifest.model_dump(), indent=2))


@app.command("snapshots")
def snapshots() -> None:
    """List the local snapshots of external sources."""
    from jellybean.sources.snapshot import list_snapshots

    found = list_snapshots(get_settings().snapshots_root)
    if not found:
        typer.echo("No snapshots. Take one with: jellybean snapshot <source> --from ... --to ...")
    for item in found:
        m = item.manifest
        size = sum(t.bytes for t in m.tables.values()) / 1e6
        rows = sum(t.rows for t in m.tables.values())
        typer.echo(f"{m.describe()}  |  {rows:,} rows, {size:,.1f} MB  |  {m.created_at}  |  {item.path}")


@app.command("run")
def run(
    instruction: Annotated[str, typer.Argument(help="What to analyse, in natural language.")],
    auto_approve: Annotated[
        bool, typer.Option("--auto-approve", help="Always take the recommended option at control points.")
    ] = False,
    seed: Annotated[int | None, typer.Option(help="Run seed (default from config.yaml).")] = None,
    labels: Annotated[str | None, typer.Option(help="Dataset with confirmed cases (used by the critic).")] = None,
    run_id: Annotated[str | None, typer.Option(help="Explicit run id (default YYYY-MM-DD_NNN).")] = None,
    filters: Annotated[
        list[str] | None,
        typer.Option(
            "--filter",
            help="Row filter with dataset-prefixed columns, e.g. \"traffic.date >= '2026-10-01' and "
            "traffic.date < '2026-10-03'\". Repeatable; applied to every agent and tool of the run.",
        ),
    ] = None,
    entity: Annotated[
        str | None,
        typer.Option(
            "--entity",
            help="Entity to analyse, as <dataset>.<column> (e.g. traffic.msisdn); it fixes the primary dataset. "
            "Default: proposed by the profiler. One analysis per entity.",
        ),
    ] = None,
    source: Annotated[str | None, typer.Option("--source", help=_SOURCE_HELP)] = None,
    start: Annotated[str | None, typer.Option("--from", help=_FROM_HELP)] = None,
    end: Annotated[str | None, typer.Option("--to", help=_TO_HELP)] = None,
    sample: Annotated[float | None, typer.Option("--sample", help=_SAMPLE_HELP)] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help=_REFRESH_HELP)] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="Do not print progress.")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Also print every LLM turn and all tool arguments.")
    ] = False,
) -> None:
    """Run the agent team on an instruction, from discovery to report.md.

    The datasets come from the data root, or with ``--source`` from a local
    snapshot of an external source (taken now, or reused when it exists).
    """
    from jellybean.runtime.context import RunContext
    from jellybean.runtime.llm_client import LLMConfigurationError, build_chat_model

    settings = get_settings()
    try:
        build_chat_model(settings)  # validate the configuration before creating a workspace
    except LLMConfigurationError as exc:
        typer.secho(f"LLM configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    source_summary = None
    if source is None and (start or end or sample is not None or refresh):
        typer.secho("--from, --to, --sample and --refresh need --source.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    if source is not None:
        from jellybean.sources.snapshot import manifest_summary

        taken = _take_snapshot(settings, source, start, end, sample, refresh)
        settings, source_summary = settings.with_data_root(taken.path), manifest_summary(taken.manifest)
    from jellybean.filters import FilterError, available_datasets, parse_filters, validate_filters

    try:
        parsed = parse_filters(filters or [], available_datasets(settings))
        counts = validate_filters(settings, parsed)
    except FilterError as exc:
        typer.secho(f"Filter error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    for name, (after, before) in counts.items():
        typer.echo(f"Filter {name}: {parsed[name]} -> {after:,} of {before:,} rows")
    user_entity = None
    if entity:
        from jellybean.entity import EntityError, parse_entity, validate_entity

        try:
            dataset, column = parse_entity(entity, settings)
            stats = validate_entity(settings, dataset, column, parsed.get(dataset))
        except EntityError as exc:
            typer.secho(f"Entity error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
        user_entity = {"dataset": dataset, "column": column, "stats": stats}
        typer.echo(
            f"Entity {dataset}.{column}: {stats['entities']:,} entities, median {stats['median_events']:g} "
            f"events each (min {stats['min_events']:,}, max {stats['max_events']:,})"
        )
    ctx = RunContext.create(
        settings,
        seed=seed if seed is not None else settings.config.run.default_seed,
        run_id=run_id,
        filters=parsed,
        source=source_summary,
    )
    runtime = _runtime(ctx, auto_approve, {"instruction": instruction, "labels": labels, "entity": user_entity})
    typer.echo(f"Run {ctx.run_id} ({settings.env.provider}:{runtime.llm.model_name}) -> {ctx.run_dir}")
    with _progress(quiet, verbose), _interruptible(ctx):
        result = runtime.run_agent("orquestrador", instruction)
    _finish(runtime, result)


@app.command("resume")
def resume(
    run_id: Annotated[str, typer.Argument(help="Run to continue.")],
    auto_approve: Annotated[
        bool, typer.Option("--auto-approve", help="Always take the recommended option at control points.")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="Do not print progress.")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Also print every LLM turn and all tool arguments.")
    ] = False,
) -> None:
    """Continue a run from the phase recorded in its spec, reusing existing artifacts."""
    from jellybean.reporting import trace_stats

    ctx, spec = _open_run(run_id, need_data=True)
    runtime = _runtime(ctx, auto_approve, {"instruction": spec.instruction, "labels": spec.labels_dataset})
    # The token budget continues from what the run already used.
    stats = trace_stats(ctx)
    runtime.budget.used = int(
        stats.get("input_tokens", 0) - 0.9 * stats.get("cache_read_tokens", 0) + stats.get("output_tokens", 0)
    )
    task = (
        f"Resume run {run_id}. The spec phase is '{spec.phase}' (cleaning plan status "
        f"'{spec.cleaning_plan.status}'). Follow the 'Resuming' section of your instructions: continue from "
        "that point, reuse existing artifacts and findings, and do not repeat completed steps."
    )
    typer.echo(f"Resuming {run_id} at phase '{spec.phase}' ({ctx.settings.env.provider}:{runtime.llm.model_name})")
    with _progress(quiet, verbose), _interruptible(ctx):
        result = runtime.run_agent("orquestrador", task)
    _finish(runtime, result)


@app.command("report")
def report(
    run_id: Annotated[str, typer.Argument(help="Run whose report to (re)generate.")],
    show: Annotated[bool, typer.Option("--show", help="Print the report to the terminal.")] = False,
) -> None:
    """Regenerate report.md of a run from its workspace (no LLM involved)."""
    from jellybean.reporting import write_report_file

    ctx, _ = _open_run(run_id)
    path = write_report_file(ctx)
    typer.echo(Path(path).read_text(encoding="utf-8") if show else f"Report written to {path}")


@app.command("feedback")
def feedback(
    run_id: Annotated[str, typer.Argument(help="Run of the finding.")],
    finding_id: Annotated[str, typer.Argument(help="Finding id, e.g. F003.")],
    verdict: Annotated[str, typer.Argument(help="confirmed | false_alarm")],
    note: Annotated[str, typer.Option("--note", help="Optional note stored with the decision.")] = "",
) -> None:
    """Record human feedback on a finding; positive feedback registers its pattern (no LLM involved)."""
    from jellybean.reporting import write_report_file
    from jellybean.runtime.context import use_run
    from jellybean.skills._shared.findings import FindingNotFoundError, load_finding, save_finding
    from jellybean.skills._shared.spec_tools import add_decision
    from jellybean.skills.analista.tools import register_pattern
    from jellybean.spec import Decision

    if verdict not in ("confirmed", "false_alarm"):
        typer.secho("verdict must be 'confirmed' or 'false_alarm'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    ctx, _ = _open_run(run_id)
    with use_run(ctx):
        try:
            f = load_finding(finding_id)
        except FindingNotFoundError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
        f.human_feedback = verdict  # type: ignore[assignment]
        save_finding(f)
        add_decision(
            Decision(
                topic=f"finding_review:{finding_id}",
                choice=verdict,
                by="human",
                evidence={"note": note} if note else {},
                options=["confirmed", "false_alarm"],
            )
        )
        message = f"{finding_id}: feedback '{verdict}' recorded."
        if verdict == "confirmed":
            result = register_pattern(finding_id)
            if result.status == "ok":
                message += f" Pattern {result.summary['pattern']} registered ({result.summary.get('key', '')})."
            else:
                message += f" Pattern not registered: {result.summary.get('error')}"
        write_report_file(ctx)
    typer.echo(message)


@app.command("outputs")
def outputs(
    run_id: Annotated[str, typer.Argument(help="Run whose outputs to list.")],
    kind: Annotated[
        str | None,
        typer.Option("--type", help="Only this type: table, json, plot or text (default: tables and plots)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON (with absolute paths).")] = False,
) -> None:
    """List the datasets and other artifacts a run produced, grouped by agent, with their paths."""
    from jellybean.runtime.agent_runner import run_outputs

    ctx, _ = _open_run(run_id)
    items = run_outputs(ctx)
    if as_json:
        shown = [o | {"path": str(ctx.run_dir / o["path"])} for o in items if kind is None or o["type"] == kind]
        typer.echo(json.dumps({"run_id": run_id, "outputs": shown}, indent=2, default=str))
    else:
        _print_outputs(ctx, items, kind)


@app.command("ui")
def ui(
    port: Annotated[int, typer.Option(help="Port of the local web server.")] = 8501,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open the browser.")] = True,
) -> None:
    """Open the results explorer: a local, read-only web view of the runs in the workspace.

    Runs, their report, findings (with members and their events), every table
    (filter, sort, SQL) and the steps executed. It only listens on localhost.
    Needs the optional extra: ``uv sync --extra ui``.
    """
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("streamlit") is None:
        typer.secho("The explorer needs Streamlit: run `uv sync --extra ui`.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    script = Path(__file__).parent / "ui" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.address",
        "localhost",
        "--server.port",
        str(port),
        "--server.headless",
        "false" if open_browser else "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    typer.echo(f"Explorer on http://localhost:{port} (workspace: {get_settings().workspace_root}). Ctrl+C to stop.")
    try:
        raise typer.Exit(code=subprocess.call(command))
    except KeyboardInterrupt:
        raise typer.Exit(code=0) from None


def ctx_is_topup(ctx: Any) -> bool:
    """True when the run's primary dataset is the top-up ledger (it has its own ground truth)."""
    from jellybean.spec import load_spec

    spec = load_spec(ctx.spec_path)
    return bool(spec.primary() and spec.primary().name == "topup")


@app.command("evaluate")
def evaluate(
    run_ids: Annotated[
        list[str], typer.Argument(help="Run(s) to evaluate; several runs (one per entity) are also combined.")
    ],
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table.")] = False,
) -> None:
    """Compare runs with the dummy-data ground truth; writes evaluation.json per run.

    Telecom runs use the criteria of spec section 10, top-up runs those of
    requirements_2.md section 11. Several runs (one per entity) are also combined.
    """
    from jellybean.evaluation import combine, format_combined, format_evaluation
    from jellybean.evaluation import evaluate as run_evaluation

    results = []
    for run_id in run_ids:
        ctx, _ = _open_run(run_id)
        truth = ctx.settings.ground_truth_root
        if not (truth / "anomalies.parquet").exists() or (
            ctx_is_topup(ctx) and not (truth / "expected_spec_topup.yaml").exists()
        ):
            hint = (
                " Only telecom data (generate-data --type telecom) can be evaluated; telemetry runs are "
                "checked by tests/test_unknown_domain.py."
                if (truth / "telemetry_truth.json").exists()
                else ""
            )
            scenario = "top-up" if ctx_is_topup(ctx) else "telecom"
            typer.secho(f"No {scenario} ground truth in {truth}.{hint}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)
        result = run_evaluation(ctx, truth)
        (ctx.run_dir / "evaluation.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        results.append(result)
    if as_json:
        payload = results[0] if len(results) == 1 else {"runs": results, "combined": combine(results)}
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    for result in results:
        typer.echo(format_evaluation(result) + "\n")
    if len(results) > 1:
        typer.echo(format_combined(combine(results)))


if __name__ == "__main__":
    app()
