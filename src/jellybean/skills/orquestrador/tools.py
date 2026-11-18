# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Orchestrator tools (spec section 7.1).

The orchestrator interprets the instruction, keeps the plan and the spec,
delegates to other agents and runs the human control points. It never touches
the data beyond the shared tools.

Control 1 (confirm the spec) is deterministic: :func:`prepare_spec_confirmation`
applies the section-8 rules to the evidence artifacts, and :func:`confirm_spec`
refuses to confirm an ambiguous spec unless a human (or ``--auto-approve``)
answered the confirmation question.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from jellybean.runtime.agent_runner import current_runtime
from jellybean.runtime.context import current_run
from jellybean.runtime.tool_registry import ToolResult, tool
from jellybean.runtime.tracing import trace_event
from jellybean.skills._shared.artifacts import ArtifactNotFoundError, ArtifactStore
from jellybean.skills._shared.catalog import discover_datasets, init_run_database
from jellybean.skills._shared.spec_tools import (  # noqa: F401 - update_spec is re-exported as an orchestrator tool
    SpecMissingError,
    add_decision,
    patch_spec,
    read_spec,
    update_spec,
)
from jellybean.spec import AnalysisSpec, Decision, EntitySpec, record_decision, save_spec

# Generous timeouts: delegation runs a whole agent, ask_human waits for a person.
_LONG_TIMEOUT_S = 24 * 3600
_ORCHESTRATOR = "orquestrador"
_CRITIC = "critico"
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")
SPEC_CONFIRMATION_TOPIC = "spec_confirmation"
CONFIRM_OPTION = "Confirm proposal"
# The entity is auto-confirmed when its score leads the runner-up by this margin.
_ENTITY_MARGIN = 0.15


@tool
def init_run(instruction: str, report_language: str = "en") -> ToolResult:
    """Create the initial spec.yaml of this run (idempotent: returns the existing spec if already created).

    Args:
        instruction: The user's instruction, verbatim.
        report_language: Language of the final report: "en" (default) or "pt" when the instruction is in Portuguese.
    """
    ctx = current_run()
    warnings = []
    if ctx.spec_path.exists():
        return ToolResult.ok(read_spec(ctx).compact_summary(), warnings=["spec.yaml already exists; reused."])
    runtime = current_runtime()
    original = runtime.options.get("instruction")
    if original and original.strip() != instruction.strip():
        warnings.append("Instruction differs from the user's text; the original was stored.")
        instruction = original
    spec = AnalysisSpec(
        run_id=ctx.run_id,
        instruction=instruction,
        seed=ctx.seed,
        report_language=report_language,
        labels_dataset=runtime.options.get("labels"),
        filters=dict(ctx.filters),
    )
    user_entity = runtime.options.get("entity")  # {"dataset", "column", "stats"} from --entity
    if user_entity:
        spec = spec.model_copy(
            update={"entity": EntitySpec(column=user_entity["column"], source="user", dataset=user_entity["dataset"])}
        )
        spec = record_decision(
            spec,
            Decision(
                topic="entity",
                choice=f"{user_entity['dataset']}.{user_entity['column']}",
                by="human",
                evidence={"source": "--entity", **user_entity.get("stats", {})},
            ),
        )
    save_spec(ctx.spec_path, spec)
    init_run_database(ctx)
    trace_event("init_run", report_language=report_language)
    datasets = sorted(discover_datasets(ctx.settings.data_root))
    return ToolResult.ok({"spec": spec.compact_summary(), "available_datasets": datasets}, warnings=warnings)


@tool(timeout_s=_LONG_TIMEOUT_S)
def delegate(agent: str, task: str, artifact_ids: list[str] | None = None) -> ToolResult:
    """Run another agent with a fresh context and return its structured result.

    Only artifact ids and this short task text are passed: never data or
    conversation history. For the critic, `task` must be just the finding id.

    Args:
        agent: Agent (skill) name, e.g. "profiler".
        task: Short instruction for the agent (for "critico": only the finding id).
        artifact_ids: Artifact ids the agent should work with.
    """
    runtime = current_runtime()
    if agent == _ORCHESTRATOR:
        return ToolResult.error("The orchestrator cannot delegate to itself.")
    if agent not in runtime.skills:
        return ToolResult.error(f"Unknown agent '{agent}'.", available=sorted(set(runtime.skills) - {_ORCHESTRATOR}))
    ids = list(artifact_ids or [])
    store = ArtifactStore(current_run())
    unknown = [a for a in ids if not store.exists(a)]
    if unknown:
        return ToolResult.error(f"Unknown artifact ids: {unknown}")
    if agent == _CRITIC:
        # Isolation: the critic gets the finding id and artifact ids only.
        if not _ID_RE.match(task.strip()):
            return ToolResult.error("For the critic, task must be only the finding id (no reasoning).")
        task = f"Try to refute finding {task.strip()}."
    if len(task) > 2000:
        return ToolResult.error("Task text is too long (max 2000 chars); pass artifact ids instead of content.")
    # Rework limit (spec section 4): detection and each finding's critique may run
    # at most 1 + max_rework_cycles times.
    limit = 1 + current_run().settings.config.limits.max_rework_cycles
    key = f"critico:{task}" if agent == _CRITIC else agent if agent == "detector" else None
    if key is not None:
        done = runtime.options.setdefault("delegation_counts", {})
        if done.get(key, 0) >= limit:
            return ToolResult.error(
                f"Rework limit reached for {key} ({limit} runs, max_rework_cycles in config): "
                "stop reworking and report the open requests as issues."
            )
        done[key] = done.get(key, 0) + 1
    trace_event("delegate", to=agent, artifact_ids=ids)
    result = runtime.run_agent(agent, task, ids)
    data = result.model_dump(exclude={"input_tokens", "output_tokens"})
    # Agents see artifact ids, never file paths.
    data["outputs"] = [{"id": o["id"], "type": o["type"], "rows": o["rows"]} for o in result.outputs]
    return ToolResult.ok(data)


@tool(timeout_s=_LONG_TIMEOUT_S)
def ask_human(
    question: str,
    options: Annotated[list[str], Field(min_length=2, max_length=10)],
    evidence: dict[str, Any] | None = None,
    recommended: str | None = None,
    topic: str | None = None,
) -> ToolResult:
    """Ask the human to choose among options, showing the evidence. The decision is recorded in spec.yaml.

    In --auto-approve mode the recommended option is chosen automatically.

    Args:
        question: Clear question for the human.
        options: 2 to 10 distinct options.
        evidence: Small dict with the numbers that support the decision.
        recommended: The option you recommend (must be one of `options`).
        topic: Short key for the decision (e.g. "spec_confirmation", "dataset_resolution").
    """
    if len(set(options)) != len(options):
        return ToolResult.error("Options must be distinct.")
    # Control points: the question package prepared by the tool must be used as is.
    package = current_runtime().pending_questions.get(topic or "")
    if package is not None and (options != package["options"] or recommended != package["recommended"]):
        return ToolResult.error(
            f"For topic '{topic}', pass exactly the options and recommended option returned by the "
            "preparation tool. The human can choose another option."
        )
    if recommended is not None and recommended not in options:
        return ToolResult.error("recommended must be one of the options.")
    answer = current_runtime().human.ask(question, options, evidence or {}, recommended)
    current_runtime().pending_questions.pop(topic or "", None)
    if answer.choice not in options:
        return ToolResult.error("The answer was not one of the options.")
    decision = Decision(
        topic=topic or question[:120],
        choice=answer.choice,
        by="human" if answer.by == "human" else "auto",
        evidence=evidence or {},
        options=options,
    )
    try:
        add_decision(decision)
    except SpecMissingError:
        trace_event("decision", topic=decision.topic, choice=decision.choice, by=decision.by, stored=False)
    return ToolResult.ok({"choice": answer.choice, "by": decision.by, "note": answer.note})


# --------------------------------------------------------------------------- #
# Control 1                                                                    #
# --------------------------------------------------------------------------- #
def _spec_checks(spec: AnalysisSpec, store: ArtifactStore) -> tuple[list[dict[str, Any]], list[str]]:
    """Evaluate each spec item against its evidence. Returns (items, problems)."""
    items: list[dict[str, Any]] = []
    problems: list[str] = []
    primary = spec.primary()
    if primary is None:
        problems.append(
            "No primary dataset in the spec. Next: give one dataset the role 'primary' with update_spec "
            '({"datasets": [{"name": ..., "path": ..., "role": "primary"}, ...]}).'
        )
        return items, problems

    auto_join_keys: set[str] = set()
    for i, join in enumerate(spec.joins):
        if join.status == "rejected":
            continue
        item: dict[str, Any] = {
            "item": f"join[{i}]",
            "value": f"{join.left}.{[k for k, _ in join.keys]} -> "
            f"{join.right}.{[k for _, k in join.keys]} ({', '.join(join.key_normalization) or 'none'})",
            "status": join.status,
            "auto": False,
        }
        if join.evidence_artifact is None:
            item["reason"] = "no evidence artifact"
        else:
            test = store.load_json(join.evidence_artifact)
            a, m = test["assessment"], test["metrics"]
            item["auto"] = bool(a["auto_confirm"])
            item["evidence"] = {
                "coverage": m["coverage"],
                "key_coverage": m.get("key_coverage"),
                "right_uniqueness": m["right_uniqueness"],
                "fanout": m["fanout"],
                "score_margin": a["criteria"]["score_margin"]["value"],
                "cardinality": m["cardinality"],
                "duplicate_keys": m["duplicate_keys"],
                "unmatched_rows": m["unmatched_rows"],
                "failed_criteria": [k for k, c in a["criteria"].items() if not c["ok"]],
                "always_confirm": a["requires_confirmation_reasons"],
            }
            if item["auto"] and join.left == primary.name:
                auto_join_keys.update(k for k, _ in join.keys)
        items.append(item)

    # Entity: top candidate with a clear margin, or the key of an auto-confirmable join.
    if spec.entity is None:
        problems.append(
            "No entity in the spec. Next: if the profiler has not run propose_spec yet, delegate to it once. If it "
            "reported that no entity candidate exists, do not delegate again: finish with status needs_human and "
            "ask the user to run again with --entity <dataset>.<column> (the column that says who a row belongs to)."
        )
    elif spec.entity.source == "user":
        if spec.entity.dataset != primary.name:
            problems.append(
                f"The entity was set by the user on '{spec.entity.dataset}', but the primary dataset is "
                f"'{primary.name}': make '{spec.entity.dataset}' the primary dataset."
            )
        items.append(
            {
                "item": "entity",
                "value": spec.entity.column,
                "auto": True,
                "by": "human",
                "evidence": {"source": "set by the user (--entity)"},
            }
        )
    else:
        item = {"item": "entity", "value": spec.entity.column, "alternatives": spec.entity.alternatives, "auto": False}
        evidence_id = spec.evidence_artifacts.get("entity")
        if evidence_id:
            cands = store.load_json(evidence_id)["candidates"]
            top = cands[0] if cands else None
            margin = (cands[0]["score"] - cands[1]["score"]) if len(cands) > 1 else 1.0
            is_top = top is not None and top["column"] == spec.entity.column
            item["evidence"] = {
                "top_candidate": top["column"] if top else None,
                "score_margin": round(margin, 4),
                "is_join_key": spec.entity.column in auto_join_keys,
            }
            item["auto"] = is_top and (margin >= _ENTITY_MARGIN or spec.entity.column in auto_join_keys)
        items.append(item)

    # Time column: the only non-partition timestamp of the primary dataset.
    if spec.time_column is None:
        items.append({"item": "time_column", "value": None, "auto": True, "reason": "no time column (allowed)"})
    else:
        item = {"item": "time_column", "value": spec.time_column, "auto": False}
        types_id = spec.evidence_artifacts.get(f"semantic_types:{primary.name}")
        if types_id:
            info = discover_datasets(current_run().settings.data_root).get(primary.name)
            partitions = set(info.partition_columns) if info else set()
            stamps = [
                n
                for n, c in store.load_json(types_id)["columns"].items()
                if c["semantic_type"] == "timestamp" and n not in partitions
            ]
            item["evidence"] = {"timestamp_columns": stamps}
            item["auto"] = stamps == [spec.time_column]
        items.append(item)

    # Grain: must be the top candidate and not merely "near unique".
    if spec.grain is None:
        problems.append(
            "No grain in the spec. Next: delegate to the profiler once (infer_grain, then propose_spec). If it is "
            "still missing after that, the table has no key: set it with update_spec "
            f'({{"grain": {{"dataset": "{primary.name}", "columns": [], "unique": false}}}}) and call this tool again.'
        )
    else:
        unique = spec.grain.unique
        item = {"item": "grain", "value": spec.grain.columns if unique else spec.grain.label(), "auto": False}
        grain_id = spec.evidence_artifacts.get("grain")
        if grain_id:
            doc = store.load_json(grain_id)
            cands = doc["candidates"]
            top = cands[0] if cands else None
            item["evidence"] = {
                "status": (top["status"] if top else None) if unique else "no unique key",
                "exact_duplicate_rows": doc["rows"] - doc["distinct_rows"],
            }
            item["auto"] = (
                unique and bool(top) and top["columns"] == spec.grain.columns and top["status"] != "near_unique"
            )
        if not unique:
            # A table without a key is a valid grain, but a person must agree with it.
            item["evidence"] = item.get("evidence", {}) | {
                "meaning": "event-level rows: several rows per entity and timestamp are kept as they are"
            }
            item["auto"] = False
        items.append(item)
    return items, problems


@tool
def prepare_spec_confirmation() -> ToolResult:
    """Control 1: check the proposed spec (joins, entity, time column, grain) against the confidence rules.

    Returns whether everything can be confirmed automatically and, if not, a
    ready-made question (question, options, evidence, recommended, topic) to
    pass to ask_human. Does not modify the spec.
    """
    ctx = current_run()
    try:
        spec = read_spec(ctx)
        items, problems = _spec_checks(spec, ArtifactStore(ctx))
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    except (ArtifactNotFoundError, FileNotFoundError, KeyError) as exc:
        return ToolResult.error(f"Evidence artifact missing or malformed: {exc}")
    if problems:
        return ToolResult.ok(
            {"ready": False, "problems": problems, "items": items},
            warnings=["The proposal is incomplete: each problem says what to do next. Do not loop on the profiler."],
        )
    auto = all(i["auto"] for i in items)
    pending = [i["item"] for i in items if not i["auto"]]
    summary: dict[str, Any] = {"ready": True, "auto_confirmable": auto, "items": items}
    if not auto:
        summary["question"] = {
            "question": f"Please confirm the analysis spec. Items needing confirmation: {', '.join(pending)}.",
            "options": [CONFIRM_OPTION, "Change something", "Stop the run"],
            "evidence": {i["item"]: {"value": i["value"], **i.get("evidence", {})} for i in items},
            "recommended": CONFIRM_OPTION,
            "topic": SPEC_CONFIRMATION_TOPIC,
        }
        current_runtime().pending_questions[SPEC_CONFIRMATION_TOPIC] = summary["question"]
    return ToolResult.ok(summary)


@tool
def confirm_spec(note: str = "") -> ToolResult:
    """Control 1: mark the proposed joins, entity, time column and grain as confirmed and move to phase "cleaning".

    Allowed only when prepare_spec_confirmation reports auto_confirmable, or when
    the latest "spec_confirmation" decision (from ask_human) is "Confirm proposal".

    Args:
        note: Optional note stored with the decision.
    """
    ctx = current_run()
    try:
        spec = read_spec(ctx)
        items, problems = _spec_checks(spec, ArtifactStore(ctx))
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    except (ArtifactNotFoundError, FileNotFoundError, KeyError) as exc:
        return ToolResult.error(f"Evidence artifact missing or malformed: {exc}")
    if problems:
        return ToolResult.error("The spec is incomplete.", problems=problems)
    auto = all(i["auto"] for i in items)
    human = [d for d in spec.decisions if d.topic == SPEC_CONFIRMATION_TOPIC]
    approved = bool(human) and human[-1].choice == CONFIRM_OPTION
    if not auto and not approved:
        return ToolResult.error(
            "Spec needs confirmation: call ask_human with the question from prepare_spec_confirmation first."
        )
    joins = [
        j.model_dump(mode="json") | ({"status": "confirmed"} if j.status == "proposed" else {}) for j in spec.joins
    ]
    patch_spec({"joins": joins, "phase": "cleaning"}, ctx)
    by = "auto" if auto else human[-1].by
    for item in items:
        add_decision(
            Decision(
                topic=f"control_1:{item['item']}",
                choice=f"confirmed {item['value']}",
                by=item.get("by", by),
                evidence=item.get("evidence", {}),
            ),
            ctx,
        )
    if note:
        add_decision(Decision(topic="control_1:note", choice=note[:500], by=by), ctx)
    return ToolResult.ok({"confirmed": [i["item"] for i in items], "by": by, "phase": "cleaning"})


# --------------------------------------------------------------------------- #
# Control 2                                                                    #
# --------------------------------------------------------------------------- #
CLEANING_TOPIC = "cleaning_approval"
APPROVE_RECOMMENDED = "Approve recommended plan"
APPROVE_ALL = "Approve plan including inferred fixes"
REJECT_PLAN = "Reject plan"


@tool
def prepare_cleaning_approval() -> ToolResult:
    """Control 2: build the approval question for the proposed cleaning plan (pass it to ask_human).

    The recommended option excludes operations that require explicit approval
    (fixes inferred from evidence, such as unit conversions).
    """
    ctx = current_run()
    try:
        spec = read_spec(ctx)
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    ref = spec.cleaning_plan
    if ref.status != "proposed" or not ref.artifact_id:
        return ToolResult.error(f"No proposed cleaning plan (status '{ref.status}'): delegate to the preparer first.")
    plan = ArtifactStore(ctx).load_json(ref.artifact_id)
    explicit = [o["id"] for o in plan["ops"] if o["requires_explicit_approval"]]
    options = [APPROVE_RECOMMENDED] + ([APPROVE_ALL] if explicit else []) + [REJECT_PLAN]
    evidence = {
        "operations": [
            f"{o['id']} {o['dataset']}: {o['operation']['op']} "
            f"{o['operation'].get('column') or o['operation'].get('keys') or o['operation'].get('where') or ''}"
            f"{' [explicit approval]' if o['requires_explicit_approval'] else ''} - {o['justification']}"
            for o in plan["ops"]
        ],
        "not_addressed": [f"{n['issue_id']}: {n['reason']}" for n in plan["not_addressed"]],
    }
    question = {
        "question": "Approve the cleaning plan? Operations marked [explicit approval] change values based on an "
        "inference and are only applied with the second option.",
        "options": options,
        "evidence": evidence,
        "recommended": APPROVE_RECOMMENDED,
        "topic": CLEANING_TOPIC,
    }
    current_runtime().pending_questions[CLEANING_TOPIC] = question
    return ToolResult.ok(
        {
            "plan": ref.artifact_id,
            "operations": len(plan["ops"]),
            "explicit_approval_ops": explicit,
            "question": question,
        }
    )


@tool
def approve_cleaning_plan() -> ToolResult:
    """Control 2: record the human's answer to the cleaning question in the spec (approved operations or rejection).

    Requires a prior ask_human call with topic "cleaning_approval". Returns
    status "unchanged" when the approved operations are exactly those already
    applied (a re-proposal that changes nothing): no rework is then needed.
    """
    ctx = current_run()
    try:
        spec = read_spec(ctx)
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    ref = spec.cleaning_plan
    if ref.status != "proposed" or not ref.artifact_id:
        return ToolResult.error(f"No proposed cleaning plan (status '{ref.status}').")
    answers = [d for d in spec.decisions if d.topic == CLEANING_TOPIC]
    if not answers:
        return ToolResult.error("Ask the human first: ask_human with the question from prepare_cleaning_approval.")
    choice = answers[-1].choice
    plan = ArtifactStore(ctx).load_json(ref.artifact_id)
    if choice == REJECT_PLAN:
        patch_spec({"cleaning_plan": {"status": "rejected", "approved_ops": []}}, ctx)
        return ToolResult.ok({"status": "rejected"})
    if choice == APPROVE_ALL:
        approved = [o["id"] for o in plan["ops"]]
    elif choice == APPROVE_RECOMMENDED:
        approved = [o["id"] for o in plan["ops"] if not o["requires_explicit_approval"]]
    else:
        return ToolResult.error(f"Unexpected answer '{choice}'.")
    excluded = [o["id"] for o in plan["ops"] if o["id"] not in approved]
    from jellybean.skills.preparer.tools import operation_signature

    signature = operation_signature([o for o in plan["ops"] if o["id"] in approved])
    if ref.applied_signature and signature == ref.applied_signature:
        # Same operations as the cleaning already in effect: nothing to re-apply.
        patch_spec({"cleaning_plan": {"status": "applied", "approved_ops": approved}}, ctx)
        return ToolResult.ok(
            {
                "status": "unchanged",
                "approved_ops": approved,
                "excluded_ops": excluded,
                "by": answers[-1].by,
                "note": "The approved operations are the ones already applied: no rework is needed.",
            }
        )
    patch_spec({"cleaning_plan": {"status": "approved", "approved_ops": approved}}, ctx)
    return ToolResult.ok(
        {"status": "approved", "approved_ops": approved, "excluded_ops": excluded, "by": answers[-1].by}
    )


# --------------------------------------------------------------------------- #
# Control 3                                                                    #
# --------------------------------------------------------------------------- #
REVIEW_PREFIX = "finding_review:"
FEEDBACK_CONFIRMED = "Confirmed pattern"
FEEDBACK_FALSE = "False alarm"
FEEDBACK_NONE = "No feedback"


def _review_question(f) -> dict[str, Any]:
    return {
        "question": f"Is finding {f.id} ({f.type}, {f.size} members, novelty {f.novelty}) a real pattern?",
        "options": [FEEDBACK_CONFIRMED, FEEDBACK_FALSE, FEEDBACK_NONE],
        "evidence": {
            "description": f.description[:600] or "(not described)",
            "top_features": [f"{t['feature']} ({t['direction']}, effect {t['effect']})" for t in f.top_features[:5]],
            "rules": [r.get("rule") for r in f.rules[:2]],
            "critic": (f.critic_evidence or {}).get("evidence"),
        },
        "recommended": FEEDBACK_NONE,
    }


@tool(timeout_s=_LONG_TIMEOUT_S)
def review_findings() -> ToolResult:
    """Control 3: ask the human about every confirmed finding and store the answers in the findings.

    One question per confirmed finding (evidence: description, top features,
    rules, critic evidence). The recommended option is "No feedback", so
    --auto-approve never teaches the pattern registry. Returns the findings to
    register as patterns (positive feedback).
    """
    from jellybean.skills._shared.findings import all_findings, save_finding

    runtime = current_runtime()
    findings = [f for f in all_findings() if f.critic_verdict == "confirmed"]
    feedback, to_register = {}, []
    for f in findings:
        q = _review_question(f)
        answer = runtime.human.ask(q["question"], q["options"], q["evidence"], q["recommended"])
        choice = answer.choice if answer.choice in q["options"] else FEEDBACK_NONE
        add_decision(
            Decision(
                topic=f"{REVIEW_PREFIX}{f.id}",
                choice=choice,
                by="human" if answer.by == "human" else "auto",
                options=q["options"],
                evidence={"note": answer.note} if answer.note else {},
            )
        )
        f.human_feedback = {FEEDBACK_CONFIRMED: "confirmed", FEEDBACK_FALSE: "false_alarm"}.get(choice)
        save_finding(f)
        feedback[f.id] = f.human_feedback
        if f.human_feedback == "confirmed" and not f.registered_pattern:
            to_register.append(f.id)
    return ToolResult.ok({"reviewed": len(findings), "feedback": feedback, "register_as_patterns": to_register})


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
@tool
def write_report() -> ToolResult:
    """Generate report.md from the spec, decisions, cleaning, features and findings (in the report language).

    Deterministic: the only LLM-written text it includes are the analyst's descriptions.
    """
    from jellybean.reporting import write_report_file

    ctx = current_run()
    try:
        read_spec(ctx)
    except SpecMissingError as exc:
        return ToolResult.error(str(exc))
    path = write_report_file(ctx)
    return ToolResult.ok({"report": f"{ctx.run_id}/report.md", "chars": len(Path(path).read_text(encoding="utf-8"))})
