# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""``report.md``: the final, deterministic report of a run (spec sections 8 and 12).

The report is assembled from the workspace only: spec, decisions, cleaning
plan, join, features, findings and trace. The only free text written by an LLM
is each confirmed finding's description (by the analyst). The report is
written in the spec's ``report_language`` (``en`` or ``pt``), and can be
regenerated at any time with ``jellybean report <run_id>``. A run that stopped
early (token budget, failure) gets a partial report.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import duckdb

from jellybean.runtime.context import RunContext
from jellybean.skills._shared.artifacts import ArtifactNotFoundError, ArtifactStore
from jellybean.skills._shared.catalog import sql_escape
from jellybean.skills._shared.findings import all_findings
from jellybean.spec import AnalysisSpec, load_spec

_TEXT = {
    "en": {
        "title": "Jellybean report",
        "run": "Run",
        "instruction": "Instruction",
        "generated": "Generated",
        "status": "Status",
        "complete": "complete",
        "partial": "PARTIAL: the run stopped at phase '{phase}'",
        "summary": "Summary",
        "spec": "Analysis specification",
        "datasets": "Datasets",
        "joins": "Joins",
        "grain": "Grain",
        "entity": "Entity",
        "alternatives": "alternatives",
        "time": "Time column",
        "decisions": "Decisions",
        "topic": "Topic",
        "choice": "Choice",
        "by": "By",
        "at": "At",
        "prep": "Data preparation",
        "applied": "Operations applied",
        "not_applied": "Operations not applied",
        "not_addressed": "Quality issues intentionally not addressed",
        "join_result": "Join result",
        "unmatched": "Rows without a match are kept, flagged and analysed as a group",
        "features": "Features",
        "findings": "Findings by verdict",
        "confirmed": "Confirmed",
        "explained": "Explained (real patterns with a legitimate cause)",
        "explained_by": "Explanation",
        "artifact": "Data artefacts",
        "weak": "Weak",
        "unstable": "Unstable",
        "pending": "Not judged",
        "members": "members",
        "methods": "methods",
        "novelty": "novelty",
        "top": "Top features",
        "rules": "Surrogate rules",
        "exemplars": "Exemplars (masked)",
        "plots": "Plots",
        "critic": "Critic evidence",
        "reasons": "reasons",
        "feedback": "Human feedback and pattern registry",
        "limitations": "Limitations",
        "hygiene": "Run hygiene",
        "none": "none",
        "rows": "rows",
        "issues": "Open issues reported by the agents",
        "filters": "Row filters",
        "overview": "Findings on the PCA map",
        "notes": "Context notes (from the datasets' README.md)",
        "clusters": "Cluster map",
        "outputs": "Outputs (datasets and other artifacts)",
        "no_desc": "(no description)",
    },
    "pt": {
        "title": "Relatório Jellybean",
        "run": "Execução",
        "instruction": "Instrução",
        "generated": "Gerado",
        "status": "Estado",
        "complete": "completo",
        "partial": "PARCIAL: a execução parou na fase '{phase}'",
        "summary": "Resumo",
        "spec": "Especificação da análise",
        "datasets": "Datasets",
        "joins": "Joins",
        "grain": "Granularidade",
        "entity": "Entidade",
        "alternatives": "alternativas",
        "time": "Coluna temporal",
        "decisions": "Decisões",
        "topic": "Tópico",
        "choice": "Escolha",
        "by": "Por",
        "at": "Quando",
        "prep": "Preparação dos dados",
        "applied": "Operações aplicadas",
        "not_applied": "Operações não aplicadas",
        "not_addressed": "Problemas de qualidade não tratados intencionalmente",
        "join_result": "Resultado do join",
        "unmatched": "As linhas sem correspondência são mantidas, marcadas e analisadas como grupo",
        "features": "Features",
        "findings": "Achados por veredicto",
        "confirmed": "Confirmados",
        "explained": "Explicados (padrões reais com causa legítima)",
        "explained_by": "Explicação",
        "artifact": "Artefactos dos dados",
        "weak": "Fracos",
        "unstable": "Instáveis",
        "pending": "Sem veredicto",
        "members": "membros",
        "methods": "métodos",
        "novelty": "novidade",
        "top": "Features principais",
        "rules": "Regras substitutas",
        "exemplars": "Exemplos (mascarados)",
        "plots": "Gráficos",
        "critic": "Evidência do crítico",
        "reasons": "razões",
        "feedback": "Feedback humano e registo de padrões",
        "limitations": "Limitações",
        "hygiene": "Higiene da execução",
        "none": "nenhum",
        "rows": "linhas",
        "issues": "Problemas em aberto reportados pelos agentes",
        "filters": "Filtros de linhas",
        "overview": "Achados no mapa PCA",
        "notes": "Notas de contexto (do README.md dos datasets)",
        "clusters": "Mapa de clusters",
        "outputs": "Outputs (datasets e outros artefactos)",
        "no_desc": "(sem descrição)",
    },
}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def trace_stats(ctx: RunContext) -> dict[str, Any]:
    """Hygiene numbers from ``logs/trace.jsonl``."""
    if not ctx.trace_path.exists():
        return {}
    events = [json.loads(line) for line in ctx.trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tools = [e for e in events if e["kind"] == "tool_call"]
    llm = [e for e in events if e["kind"] == "llm_call"]
    input_tokens = sum(e.get("input_tokens", 0) for e in llm)
    cached = sum(e.get("cache_read_tokens", 0) or 0 for e in llm)
    return {
        "tool_calls": len(tools),
        "tool_errors": sum(e["status"] == "error" for e in tools),
        "llm_calls": len(llm),
        "input_tokens": input_tokens,
        "output_tokens": sum(e.get("output_tokens", 0) for e in llm),
        "cache_read_tokens": cached,
        "cache_hit_share": round(cached / input_tokens, 4) if input_tokens else 0.0,
        "max_tool_output_chars": max((e["output_chars"] for e in tools), default=0),
        "max_raw_rows": max((e["raw_rows"] for e in tools), default=0),
        "truncated_outputs": sum(bool(e.get("truncated")) for e in tools),
    }


def _join_result(ctx: RunContext, spec: AnalysisSpec) -> dict[str, Any] | None:
    if not spec.enriched_artifact:
        return None
    store = ArtifactStore(ctx)
    record = store.get(spec.enriched_artifact)
    flag = record.params.get("flag")
    if not flag:
        return None
    path = sql_escape(str(store.path_of(record.id)))
    entity = spec.entity.column if spec.entity else None
    con = duckdb.connect()
    try:
        ent = f', count(DISTINCT "{entity}") FILTER (WHERE NOT "{flag}")' if entity else ", NULL"
        rows, unmatched, unmatched_entities = con.execute(
            f"SELECT count(*), count(*) FILTER (WHERE NOT \"{flag}\") {ent} FROM read_parquet('{path}')"
        ).fetchall()[0]
    finally:
        con.close()
    return {"rows": rows, "unmatched_rows": unmatched, "unmatched_entities": unmatched_entities, "flag": flag}


def _unresolved(ctx: RunContext, spec: AnalysisSpec) -> tuple[list[dict], list[dict], list[dict]]:
    """(applied ops with affected rows, ops not applied, issues not addressed)."""
    store = ArtifactStore(ctx)
    if not spec.cleaning_plan.artifact_id:
        return [], [], []
    plan = store.load_json(spec.cleaning_plan.artifact_id)
    logs: dict[str, dict] = {}
    for artifact_id in spec.prepared.values():
        for entry in store.get(artifact_id).params.get("log", []):
            logs[entry["op_id"]] = entry
    approved = set(spec.cleaning_plan.approved_ops) if spec.cleaning_plan.status in ("approved", "applied") else set()
    applied = [
        o | {"affected_rows": logs.get(o["id"], {}).get("affected_rows")} for o in plan["ops"] if o["id"] in approved
    ]
    not_applied = [o for o in plan["ops"] if o["id"] not in approved]
    return applied, not_applied, plan.get("not_addressed", [])


def build_report(ctx: RunContext, run_result: dict[str, Any] | None = None) -> str:
    """Render the report of a run as Markdown."""
    spec = load_spec(ctx.spec_path)
    t = _TEXT.get(spec.report_language, _TEXT["en"])
    findings = all_findings(ctx)
    stats = trace_stats(ctx)
    lines: list[str] = []
    add = lines.append

    complete = spec.phase in ("report", "done")
    add(f"# {t['title']}: {spec.run_id}\n")
    add(f"- **{t['instruction']}:** {spec.instruction}")
    add(f"- **{t['generated']}:** {datetime.now(UTC).isoformat(timespec='seconds')}")
    status = t["complete"] if complete else t["partial"].format(phase=spec.phase)
    if run_result and run_result.get("status") not in (None, "done"):
        status += f" ({run_result['status']})"
    add(f"- **{t['status']}:** {status}\n")

    by_verdict: dict[str, list] = {}
    for f in findings:
        by_verdict.setdefault(f.critic_verdict or "pending", []).append(f)
    add(f"## {t['summary']}\n")
    counts = ", ".join(f"{t.get(v, v)}: {len(fs)}" for v, fs in sorted(by_verdict.items())) or t["none"]
    add(f"{t['findings']}: {counts}\n")
    for f in by_verdict.get("confirmed", []):
        first = (f.description.split(". ")[0] + ".") if f.description else t["no_desc"]
        add(f"- **{f.id}** ({f.type}, {f.size} {t['members']}, {t['novelty']} {f.novelty}): {first}")
    add("")

    # ------------------------------------------------------------------ spec
    add(f"## {t['spec']}\n")
    add(f"**{t['datasets']}:** " + ", ".join(f"`{d.name}` ({d.role})" for d in spec.datasets))
    if spec.context_notes:
        add(f"\n**{t['notes']}:**\n")
        add("> " + spec.context_notes.strip().replace("\n", "\n> ") + "\n")
    if spec.filters:
        add(f"\n**{t['filters']}:** " + "; ".join(f"`{name}`: {cond}" for name, cond in spec.filters.items()))
    if spec.joins:
        add(f"\n**{t['joins']}:**\n")
        add("| left | right | keys | normalisation | type | cardinality | coverage | fan-out | confidence | status |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for j in spec.joins:
            keys = ", ".join(f"{a} = {b}" for a, b in j.keys)
            add(
                f"| {j.left} | {j.right} | {keys} | {', '.join(j.key_normalization) or '-'} | {j.type} | "
                f"{j.cardinality} | {_fmt(j.coverage)} | {_fmt(j.fanout)} | {_fmt(j.confidence)} | {j.status} |"
            )
    if spec.grain:
        add(f"\n**{t['grain']}:** `{spec.grain.dataset}` → {', '.join(spec.grain.columns)}")
    if spec.entity:
        alts = ", ".join(spec.entity.alternatives) or "-"
        add(f"\n**{t['entity']}:** `{spec.entity.column}` ({t['alternatives']}: {alts})")
    add(f"\n**{t['time']}:** `{spec.time_column}`\n")

    add(f"## {t['decisions']}\n")
    add(f"| {t['topic']} | {t['choice']} | {t['by']} | {t['at']} |")
    add("|---|---|---|---|")
    for d in spec.decisions:
        add(f"| {d.topic} | {str(d.choice)[:120]} | {d.by} | {d.at} |")
    add("")

    # ----------------------------------------------------------- preparation
    try:
        applied, not_applied, not_addressed = _unresolved(ctx, spec)
    except (ArtifactNotFoundError, FileNotFoundError, KeyError):
        applied, not_applied, not_addressed = [], [], []
    add(f"## {t['prep']}\n")
    add(f"**{t['applied']}:**\n")
    for o in applied or []:
        op = o["operation"]
        target = op.get("column") or op.get("keys") or op.get("where") or ""
        affected = f" — {_fmt(o['affected_rows'])} {t['rows']}" if o.get("affected_rows") is not None else ""
        add(f"- `{o['id']}` {o['dataset']}: {op['op']} {target}{affected}. {o['justification']}")
    if not applied:
        add(f"- {t['none']}")
    if not_applied:
        add(f"\n**{t['not_applied']}:**\n")
        for o in not_applied:
            add(f"- `{o['id']}` {o['dataset']}: {o['operation']['op']} — {o['justification']}")
    if not_addressed:
        add(f"\n**{t['not_addressed']}:**\n")
        for n in not_addressed:
            add(f"- `{n['issue_id']}`: {n['reason']}")
    join = _join_result(ctx, spec)
    if join:
        add(
            f"\n**{t['join_result']}:** {_fmt(join['rows'])} {t['rows']}; "
            f"{_fmt(join['unmatched_rows'])} {t['rows']} / {_fmt(join['unmatched_entities'])} "
            f"{t['entity'].lower()} — `{join['flag']} = false`. {t['unmatched']}."
        )
    add("")

    # -------------------------------------------------------------- features
    if spec.features.table:
        store = ArtifactStore(ctx)
        record = store.get(spec.features.table)
        add(f"## {t['features']}\n")
        add(
            f"- `{record.id}`: {_fmt(record.row_count)} {t['entity'].lower()}; "
            f"attributes: {', '.join(record.params.get('attributes', [])) or '-'}"
        )
        if spec.features.graph:
            add(f"- graph: `{spec.features.graph}`")
        add("")

    # -------------------------------------------------------------- findings
    add(f"## {t['findings']}\n")
    overview = _findings_overview(ctx, findings)
    if overview:
        add(f"![{t['overview']}]({overview})\n")
    cluster_plots = [r for r in ArtifactStore(ctx).list("plot") if r.params.get("kind") == "clusters"]
    for r in cluster_plots:
        add(f"![{t['clusters']}: {r.params.get('run')}]({r.path})\n")
    order = ["confirmed", "explained", "artifact", "weak", "unstable", "pending"]
    for verdict in order:
        group = by_verdict.get(verdict, [])
        if not group:
            continue
        add(f"### {t[verdict]} ({len(group)})\n")
        for f in group:
            add(f"#### {f.id} — {f.type}, {f.size} {t['members']}\n")
            add(f"- {t['methods']}: {', '.join(f.methods)}; {t['novelty']}: {f.novelty or '-'}")
            if f.context.get("column"):
                add(f"- context: {f.context.get('column')} = {f.context.get('value')}")
            if verdict == "explained":
                why = (f.checks.get("explanation") or {}).get("summary")
                add(f"- {t['explained_by']}: {why or '-'}")
            if verdict == "confirmed":
                add(f"\n{f.description or t['no_desc']}\n")
                if f.top_features:
                    add(f"**{t['top']}:**\n")
                    for feat in f.top_features[:6]:
                        detail = (
                            f"{feat.get('group_median')} vs {feat.get('population_median')}"
                            if feat.get("kind") == "numeric"
                            else f"{feat.get('value')}: {feat.get('group_share')} vs {feat.get('population_share')}"
                            if feat.get("kind") == "categorical"
                            else ""
                        )
                        add(f"- `{feat['feature']}` {feat['direction']} (effect {feat['effect']}) {detail}")
                if f.rules:
                    add(f"\n**{t['rules']}:**\n")
                    for r in f.rules:
                        add(f"- `{r['rule']}` — precision {r['precision']}, coverage {r['coverage']}")
                if f.exemplars:
                    add(f"\n**{t['exemplars']}:**\n")
                    cols = list(f.exemplars[0])
                    add("| " + " | ".join(cols) + " |")
                    add("|" + "---|" * len(cols))
                    for e in f.exemplars[:5]:
                        add("| " + " | ".join(_fmt(e[c]) for c in cols) + " |")
                if f.plots:
                    add(f"\n**{t['plots']}:** " + ", ".join(f"[{p.rsplit('/', 1)[-1]}]({p})" for p in f.plots))
            checks = f.checks
            evidence = []
            if "stability" in checks:
                evidence.append(f"stability {checks['stability']['stability']}")
            if checks.get("ensemble", {}).get("applicable"):
                evidence.append(f"mean votes {checks['ensemble']['mean_votes']}")
            if "artifact" in checks and checks["artifact"].get("low_support_share") is not None:
                evidence.append(f"low support {checks['artifact']['low_support_share']}")
            if evidence:
                add(f"\n- {t['critic']}: " + "; ".join(evidence))
            reasons = checks.get("artifact", {}).get("reasons", [])
            if reasons and verdict != "confirmed":
                add(f"- {t['reasons']}: " + " | ".join(reasons[:3]))
            add("")

    # -------------------------------------------------------------- feedback
    add(f"## {t['feedback']}\n")
    reviewed = [f for f in findings if f.human_feedback or f.registered_pattern]
    for f in reviewed:
        add(f"- {f.id}: {f.human_feedback or '-'}" + (f" → `{f.registered_pattern}`" if f.registered_pattern else ""))
    if not reviewed:
        add(f"- {t['none']}")
    add("")

    # ----------------------------------------------------------- limitations
    add(f"## {t['limitations']}\n")
    limitations = [f"`{n['issue_id']}`: {n['reason']}" for n in not_addressed]
    limitations += [f"`{o['id']}` not applied: {o['justification']}" for o in not_applied]
    if not complete:
        limitations.append(t["partial"].format(phase=spec.phase))
    for item in limitations or [t["none"]]:
        add(f"- {item}")
    if run_result and run_result.get("issues"):
        add(f"\n**{t['issues']}:**\n")
        for issue in run_result["issues"][:30]:
            add(f"- {issue}")
    add("")

    # --------------------------------------------------------------- outputs
    records = ArtifactStore(ctx).list() if ctx.manifest_path.exists() else []
    if records:
        add(f"## {t['outputs']}\n")
        add("| artifact | type | rows | created by | link |")
        add("|---|---|---|---|---|")
        for r in sorted(records, key=lambda r: (r.type != "table", r.created_at)):
            tool = f" / {r.created_by.tool}" if r.created_by.tool else ""
            rows = _fmt(r.row_count) if r.row_count is not None else "-"
            add(f"| `{r.id}` | {r.type} | {rows} | {r.created_by.agent}{tool} | [{r.path}]({r.path}) |")
        add("")

    # --------------------------------------------------------------- hygiene
    if stats:
        add(f"## {t['hygiene']}\n")
        for key, value in stats.items():
            add(f"- {key}: {_fmt(value)}")
        add("")
    return "\n".join(lines)


def _findings_overview(ctx: RunContext, findings: list) -> str | None:
    """Draw report_findings_overview.png (regenerated with the report); returns its relative path."""
    from jellybean.skills._shared.plots import findings_map, save_figure

    try:
        fig = findings_map(ctx, findings)
    except Exception:
        return None
    if fig is None:
        return None
    save_figure(fig, ctx.run_dir / "report_findings_overview.png")
    return "report_findings_overview.png"


def write_report_file(ctx: RunContext) -> str:
    """Write ``report.md`` (using ``run_result.json`` when present) and return its path."""
    result_path = ctx.run_dir / "run_result.json"
    run_result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    path = ctx.run_dir / "report.md"
    path.write_text(build_report(ctx, run_result), encoding="utf-8")
    return str(path)
