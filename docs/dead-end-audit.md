# Dead-end audit: mandatory steps and their exits

A **dead end** is a step the workflow requires that has no exit when its input is
legitimately absent, empty, or could not be computed in time. The agent can then only
repeat the call, improvise, or stop. Three were found on real data (run `2026-11-17_003`,
`subscribers_events`, 592,282 entities):

- `propose_spec` refused a table without a unique key, and `prepare_spec_confirmation`
  refused a spec without a grain;
- `run_clustering` and `run_timeseries_anomaly` failed in ways the agent could only answer
  by repeating the call;
- `set_verdict` required a stability check that could never complete.

## Scope and method

- **Reviewed:** every tool in `src/jellybean/skills/` (each `ToolResult.error`, each
  "run X before Y" check, each required artifact), every `SKILL.md` procedure, the phase
  transitions in `skills/orquestrador/tools.py`, the required fields of `spec.py`, and the
  agent loop in `runtime/agent_runner.py`.
- **Question asked of each gate:** what happens when the required input is absent, empty,
  or could not be computed in time?
- **One of three exits was applied:** (1) the empty or failed state becomes a recorded
  outcome with a reason, and the workflow continues with reduced confidence; (2) the error
  names the next action, and the `SKILL.md` documents the fallback; (3) the case is
  escalated to the human at the next control point.
- **Tested:** one test per fixed gate (`tests/test_dead_ends.py`, `test_no_key_grain.py`,
  `test_stability_budget.py`, `test_isolated_work.py`, `test_artifact_prefixes.py`,
  `test_runtime.py`), and one end-to-end test on an awkward dataset
  (`tests/test_awkward_dataset.py`: no unique key, one day of data, one category level,
  upper-case column names, no enrichment table) whose every tool call must succeed.
- **Not verified:** the runs on the real `subscribers_events` data. It is not on the
  development machine; the cases were reproduced on synthetic tables of the same shape.

Line numbers are the line of the function definition at the time of writing.

## Gates fixed

| # | File | Line | Gate | Failure condition | Exit that now exists |
| --- | --- | --- | --- | --- | --- |
| 1 | `skills/profiler/tools.py` | 1501 | `propose_spec` needs a grain candidate | The table has no unique key (event-level rows). | Recorded: `grain = {dataset, columns: [], unique: false}`; the rest of the proposal is written. Only an artifact that is not from `infer_grain` is refused. |
| 2 | `skills/profiler/tools.py` | 385 | `infer_grain` returns an empty candidate list | Same. | Says `unique_key: false` with the exact duplicate rows, and that this artifact goes to `propose_spec` as it is. |
| 3 | `skills/orquestrador/tools.py` | 196 | `prepare_spec_confirmation` / `confirm_spec` need a grain | `No grain in the spec.` sent the orchestrator back to the profiler. | A non-unique grain is accepted and shown as "no unique key" with the duplicate count; it always asks a person. A missing grain names the next action (one profiler pass, then set it by hand). |
| 4 | `skills/profiler/tools.py` | 1501 | `propose_spec` needs an entity candidate | `suggest_entities` found none (and no `--entity`). | The rest of the proposal is written, with a warning. Control 1 reports the missing entity with its exit: finish with `needs_human` and ask for `--entity`. |
| 5 | `skills/orquestrador/tools.py` | 196 | Control 1 problems | "Ask the profiler to complete the proposal first" could loop. | Every problem states its next action; `SKILL.md` limits the profiler to one extra pass. |
| 6 | `skills/_shared/spec_tools.py` | 165 | `update_spec` rejects a patch | Wrong shape of `grain`, `columns`... with raw validation text. | The error names the field, what was sent and the expected shape; the shapes are in the tool description. |
| 7 | `skills/_shared/catalog.py` | 206 | Any tool given a JSON artifact as a table | `get_schema` listed the datasets. | The error says it is a JSON artifact and points to `read_json_artifact` / `get_artifact_info`. `semantic_types` and `correlations` artifacts are readable. |
| 8 | `skills/_shared/data_tools.py` | 62 | `list_datasets` | A dataset folder whose name is not a valid table name was skipped silently, so the dataset "did not exist". | A warning lists such folders and says to rename them. |
| 9 | `skills/preparer/tools.py` | 492 | `propose_cleaning_plan` needs a quality-issues artifact | No quality issues were compiled. | The artifact defaults to the one in the spec; without any, the plan is empty and can be approved and applied. |
| 10 | `skills/preparer/tools.py` | 223 | Dedupe keyed on the grain | The grain is not unique: the dedupe would delete legitimate events. | Refused with the reason. On such a table the plan offers exact-row dedupe only with explicit approval, and never a keyed one. |
| 11 | `skills/preparer/tools.py` | 802 | `materialize_join` needs a join | A single dataset, or every join rejected: `No join at index 0`. | Returns `skipped` with what happens next, and moves the phase on. Other errors name the next action (`needs_rework`). |
| 12 | `skills/feature_engineer/tools.py` | 74 | `build_entity_features` dropped every grain column | With a wide grain, the other party, type and device left the analysis. | Grain columns are used by semantic type; only a technical row id is excluded. |
| 13 | `skills/_shared/artifacts.py` | 164 | Artifact prefix built from a name | An upper-case measure (`EVENT_DURATION`) made `run_timeseries_anomaly` fail; the same held for dataset and column names in the profiler. | `prefix_part()` everywhere a prefix is built from a name; the original name stays in the params. |
| 14 | `skills/detector/tools.py` | 348 | `run_timeseries_anomaly` returns 0 rows or fails | "0 rows" could not be told from a failure; no timestamps crashed. | `result` says "no anomalous buckets" (valid) or "too few time buckets"; errors name the next choice (`group_by`, granularity, measure). |
| 15 | `runtime/tool_registry.py` | 421 | Tool timeout | A model fit cannot be interrupted: the call ran for 697 s against a 300 s limit and still wrote artifacts. | Fits above `limits.isolate_above_rows` run in a worker process killed at the timeout (`runtime/isolated.py`); a timed-out call leaves no artifact; `limits.tool_timeouts` per tool. |
| 16 | `runtime/agent_runner.py` | 220 | Repeating a call that timed out | The agent repeated the identical call (another 704 s). | The same tool and arguments are refused at once in the same agent run, with what to change. |
| 17 | `skills/_shared/detection.py` | 236 | Clustering needs enough entities | Fewer entities than clusters (or than twice `min_cluster_size`) raised. | HDBSCAN returns "no cluster"; k-means and GMM use at most one cluster per entity. HDBSCAN's sample is capped so it fits the limit, and `params={"max_rows": n}` lowers it. |
| 18 | `skills/detector/tools.py` | 621 | `consolidate_findings` needs every run id to exist | A method failed or timed out, so its id did not exist: `Unknown run`. | Ids that are not detection runs are ignored with a warning. When a required kind is missing, the error says which method to run (`kmeans` always finishes). |
| 19 | `skills/critico/tools.py` | 242 | `stability_check` must finish | Re-detection with slow methods exceeded the tool limit; every repetition was lost. | Works inside a time budget and returns the repetitions that finished (`runs_completed` / `runs_requested`); re-detects on a capped sample; records the check as attempted before any work, and `stability: null` with a reason when nothing could be measured. |
| 20 | `skills/critico/tools.py` | 639 | `artifact_check` must finish | A timeout, or a finding matching no rows, left no `artifact` entry. | Recorded as attempted (`measured: false`, with the reason) before any work. |
| 21 | `skills/critico/tools.py` | 1386 | `set_verdict` requires stability and artifact checks | A check that could not complete blocked every verdict. | With a check attempted but not measured, `weak` and `unstable` are accepted (never `confirmed` or `explained`), and the evidence says what was not measured. |
| 22 | `runtime/agent_runner.py` | 278 | Agent loop | An agent could spend its whole budget repeating a failing call. | The same error from the same tool twice in a row, or `limits.max_consecutive_tool_errors` errors in a row, stops the agent: status `failed`, issue `blocked:<tool>`, the blocker in the summary. The orchestrator decides (skip an optional step, or ask the human). |

## Gates reviewed and left as they are

| File | Gate | Why it is not a dead end |
| --- | --- | --- |
| `spec.py` | `time_column` | Optional; every reader accepts `None` (no time-series or timestamp features). |
| `spec.py` | `joins` | May be empty; see gate 11. |
| `spec.py` | `entity` | Required by everything after discovery: there is no analysis without it. Its absence is escalated (gate 4). |
| `skills/orquestrador/tools.py` | `confirm_spec` without a human answer | The error says to call `ask_human` with the prepared question. |
| `skills/orquestrador/tools.py` | `approve_cleaning_plan`, `prepare_cleaning_approval` | A rejected plan is a recorded outcome (the raw datasets are used); an empty plan is valid (gate 9). |
| `skills/orquestrador/tools.py` | `delegate` rework limit | The error says to stop reworking and report the open requests. |
| `skills/detector/tools.py` | `consolidate_findings` with no group found | Returns zero findings; critique and analysis are skipped and the report says so. |
| `skills/detector/tools.py` | `run_graph_analysis`, `run_contextual_outliers` | Optional steps; their errors name the alternative, and `SKILL.md` says to skip them. |
| `skills/critico/tools.py` | `ensemble_agreement`, `explain_by_reference`, `match_pattern_registry` | Not required by `set_verdict`; temporal findings record "not applicable". |
| `skills/analista/tools.py` | `write_finding` needs `describe_group` and a confirmed verdict | Both are named in the error; a finding that is not confirmed is simply not described. |
| `skills/profiler/tools.py` | `check_temporal_coverage` on one day | Fixed earlier (a single day gives no hour-of-day profile, not an error). |

## Known limits

- A step that is **required and fails for a real reason** (the feature table cannot be
  built, the LLM is unreachable) still stops the run. The difference is that the agent
  now stops after two identical errors with the blocker in its result, and the
  orchestrator asks the human instead of retrying.
- The blocked-agent guard compares the error text. A tool that fails with a different
  message on every call is only stopped by the count of consecutive errors.
- Work that runs in-process (fits on fewer rows than `limits.isolate_above_rows`, pandas
  code) still cannot be interrupted. It is short by construction, and whatever it writes
  after a timeout is discarded.
