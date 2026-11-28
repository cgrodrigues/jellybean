<h1><img src="docs/assets/jellybean.svg" alt="Jellybean logo" width="64" align="middle"> Jellybean</h1>

[![CI](https://github.com/cgrodrigues/jellybean/actions/workflows/ci.yml/badge.svg)](https://github.com/cgrodrigues/jellybean/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/packaging-uv-DE5FE9.svg)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/linting-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](https://docs.pytest.org/)
[![gitleaks](https://img.shields.io/badge/secrets-gitleaks-blue.svg)](https://github.com/gitleaks/gitleaks)
[![pip-audit](https://img.shields.io/badge/dependencies-pip--audit-blue.svg)](https://github.com/pypa/pip-audit)
[![Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-025E8C.svg?logo=dependabot)](.github/dependabot.yml)
[![DuckDB](https://img.shields.io/badge/engine-DuckDB-FFF000.svg)](https://duckdb.org/)
[![LangChain](https://img.shields.io/badge/LLM-LangChain-1C3C3C.svg)](https://www.langchain.com/)
[![Changelog](https://img.shields.io/badge/changelog-Keep%20a%20Changelog-E05735.svg)](CHANGELOG.md)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-green.svg)](CITATION.cff)
[![Status: proof of concept](https://img.shields.io/badge/status-proof%20of%20concept-orange.svg)](#principles)

Proof of concept of a **multi-agent team** that receives a natural-language
instruction (e.g. *"analyzes the data in the traffic and enriches it with
cost_segmentation"*), loads Parquet datasets through DuckDB, discovers how to
join them, cleans them, builds features and runs unsupervised ML to find
anomalies and patterns, **without ever putting the data into the LLM context**.
The result is a report of validated findings with their evidence.


## Principles

1. **Data never enters the LLM context.** Tools return small summaries plus an
   artifact id, with at most 20 raw rows and 4000 characters per call. The
   `@tool` decorator enforces this for every tool.
2. **Tools are deterministic and do all computation.** The LLM chooses methods
   and parameters and interprets results. Control points are decided by tools
   applying documented rules.
3. **Everything is versioned.** `./data` is read-only. Each transformation is an
   immutable artifact with lineage in `manifest.json`.
4. **Whoever generates a finding does not validate it.** The critic runs in an
   isolated context per finding and receives only its id.
5. **Autonomy with confirmation.** The system proceeds alone when confidence is
   high and asks a human otherwise, at three control points.
6. **"New" requires memory.** A persistent pattern registry distinguishes
   known patterns, variants and new ones.

## The agents

| Agent | Role |
| --- | --- |
| `orquestrador` | Interprets the instruction, keeps `spec.yaml`, delegates, runs control points 1-3, writes the report |
| `profiler` | Profiles datasets, semantic types, grain, entity, time column, quality issues; discovers and tests joins |
| `preparer` | Proposes a justified cleaning plan, applies it once approved, materialises the joins (unmatched rows kept and flagged) |
| `feature_engineer` | Builds the per-entity feature table (DuckDB), graph features, scaling and PCA/UMAP |
| `detector` | Runs outlier, clustering, graph, time-series and contextual methods; consolidates candidate findings |
| `critico` | Tries to refute each finding: stability, ensemble agreement, artefact check, registry; sets the verdict |
| `analista` | Describes confirmed findings (effect sizes, rules, exemplars, plots); registers patterns after positive feedback |

Each agent is a folder in [src/jellybean/skills/](src/jellybean/skills/) with a
`SKILL.md` (its instructions) and a `tools.py` (its tools).

## Requirements and installation

- Python **3.12.3** (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync                      # core dependencies + dev tools (pytest, ruff)
uv sync --extra explain      # optional: UMAP and SHAP
uv sync --extra deep         # optional: PyTorch for the pyod AutoEncoder (>1 GB)
cp .env.example .env         # then edit provider, model and key
```

## Configuration

**`.env`** holds environment-specific settings and secrets. It is never committed.

| Variable | Meaning | Default |
| --- | --- | --- |
| `JELLYBEAN_PROVIDER` | `anthropic`, `openai` or `bedrock` (all called through LangChain) | `anthropic` |
| `JELLYBEAN_MODEL` | Model id for the chosen provider | *(none, required)* |
| `JELLYBEAN_API_KEY` | API key for `anthropic` / `openai` | — |
| `AWS_REGION`, `AWS_PROFILE`, `AWS_BEARER_TOKEN_BEDROCK` | Bedrock credentials (standard AWS chain) | — |
| `JELLYBEAN_DATA_ROOT` | Root folder of the input datasets | `./data` |
| `JELLYBEAN_WORKSPACE_ROOT` | Run workspaces and pattern registry | `./workspace` |
| `JELLYBEAN_GROUND_TRUTH_ROOT` | Ground truth of the dummy data (always blocked for agents) | `./ground_truth` |
| `JELLYBEAN_SNAPSHOTS_ROOT` | Local copies of external sources | `./snapshots` |
| `JELLYBEAN_CONFIG_FILE` | Path of the limits/thresholds file | `./config.yaml` |
| `JELLYBEAN_SOURCE_<NAME>_USER`, `_PASSWORD` | Credentials of an external source | — |

| Provider | LangChain class | Credentials |
| --- | --- | --- |
| `anthropic` | `ChatAnthropic` | `JELLYBEAN_API_KEY` |
| `openai` | `ChatOpenAI` | `JELLYBEAN_API_KEY` |
| `bedrock` | `ChatBedrockConverse` | AWS chain: `AWS_REGION` plus a profile, keys or `AWS_BEARER_TOKEN_BEDROCK` |

**`config.yaml`** holds every limit and threshold, each documented inline:
iterations, token budget, tool timeout, rework cycles, output caps, LLM call
parameters (including prompt caching), join-confidence rules,
pattern-registry thresholds, detection and critic thresholds, and PII masking
patterns, and the external sources. Defaults live in
[src/jellybean/settings.py](src/jellybean/settings.py), so the file is optional.
**Every parameter, with its default and meaning, is in
[docs/config-reference.md](docs/config-reference.md).**

**Resources.** Three groups of settings keep a run within the machine:

- `limits.duckdb_memory_limit`, `duckdb_threads`, `duckdb_max_temp_size`: every
  DuckDB connection has a memory and thread limit and spills to
  `workspace/<run_id>/tmp` above it.
- `limits.tool_timeout_s`: at the timeout the tool's queries are interrupted and
  the tool is stopped. Ctrl+C does the same and returns the prompt.
- `profiling.*` and the size caps in `detection.*` (`cluster_max_rows`,
  `group_max_pool`, `plot_max_points`, `fit_max_rows`,
  `exact_population_max_rows`, `surrogate_max_rows`, `max_candidates_per_run`,
  `graph_max_edges`): above these sizes a step samples, approximates or
  refuses, and says so in a warning or an error. Below them the results are
  exact.

The feature tables are never loaded whole to look at one group: the critic and
the analyst read only the rows of a finding and the columns they need.

- `limits.tool_timeouts` (per tool), `limits.isolate_above_rows` (model fits
  on more rows run in a worker process that the timeout can kill) and
  `limits.max_consecutive_tool_errors` (an agent that keeps failing is stopped
  and reports what blocked it).

What is still expensive on large data is listed in
[docs/performance-audit.md](docs/performance-audit.md). The fields of
`spec.yaml`, including a grain with **no unique key** and the feature windows,
are in [docs/spec-reference.md](docs/spec-reference.md); how every mandatory
step exits when its input is missing is in
[docs/dead-end-audit.md](docs/dead-end-audit.md).

## Dataset notes (README.md per dataset)

A dataset folder may contain a `README.md` with human context: what each
column means, units (seconds vs milliseconds, currency), known particularities
and what is normal for the business. For the dummy data, `generate-data
--with-readme` writes one with column meanings only, never hints about the
anomalies. That lets you compare runs with and without context (spec section 9).

- The profiler reads the notes (`read_dataset_notes`) at the start of
  discovery and summarises them into `context_notes` in `spec.yaml`.
- Every agent receives `context_notes` in the spec summary at the start of its
  task, and through `get_spec`. Their SKILL.md tells them to use the notes to
  interpret columns, and to cite contradictions between the notes and the data
  as evidence. For example, a column documented in seconds whose values look
  like milliseconds.
- `report.md` shows the notes in its specification section.

The notes are optional: without them the system still assumes nothing about
columns or domain.

## Quick start (the example)

```bash
# 1. Synthetic data with planted anomalies (about 35 s)
uv run jellybean generate-data --with-readme

# 2. Run the team (interactive: the control points ask you in the console)
uv run jellybean run "analyzes the data in the traffic and enriches it with cost_segmentation"
#    unattended: always takes the recommended option
uv run jellybean run "analyzes the data in the traffic and enriches it with cost_segmentation" --auto-approve

# 3. Read the report, evaluate against the ground truth
uv run jellybean report <run_id> --show
uv run jellybean evaluate <run_id>          # or: uv run python scripts/evaluate_run.py <run_id>

# 4. Teach the registry, continue or resume
uv run jellybean feedback <run_id> F003 confirmed --note "known fraud pattern"
uv run jellybean resume <run_id>            # continues from the phase recorded in spec.yaml
```

| Command | What it does |
| --- | --- |
| `generate-data [--type telecom\|telemetry] [--out --truth --seed --days --with-readme --overwrite]` | Writes synthetic datasets with planted anomalies and their ground truth. `telecom` (default): `traffic` (partitioned by date and hour), `cust_segmentation`, `orders` and the prepaid ledger `topup` with `promotions`, `dealers`, `voucher_batches`; options `--datasets traffic,cust_segmentation,orders,topup` (default: all) `--customers --scale --partition hour\|day --topup-days 90 --opaque-codes`. `telemetry`: the unknown-domain dataset (`telemetry`, `machines`); option `--machines` |
| `datasets` | Lists the datasets the agents will see |
| `run "<instruction>" [--auto-approve] [--seed N] [--labels <dataset>] [--run-id ID] [--verbose\|--quiet]` | The whole flow, ending with `report.md` |
| `run ... --entity <dataset>.<column>` | Fixes the analysis entity (the "who" the rows belong to) instead of letting the profiler choose it, e.g. `--entity traffic.msisdn` |
| `run ... --filter "<dataset>.<column> ..."` | Restricts a dataset's rows for the whole run (repeatable), e.g. `--filter "traffic.date >= '2026-10-01' and traffic.date < '2026-10-03'"` |
| `resume <run_id> [--auto-approve] [--verbose\|--quiet]` | Continues a run from its recorded phase, reusing artifacts and findings |
| `report <run_id> [--show]` | Regenerates `report.md` from the workspace (no LLM) |
| `feedback <run_id> <finding> confirmed\|false_alarm [--note]` | Records human feedback; positive feedback registers the pattern (no LLM) |
| `outputs <run_id> [--type table\|plot\|json] [--json]` | Lists the datasets and other artifacts of a run, grouped by agent, with their paths |
| `evaluate <run_id>... [--json]` | Compares runs with the ground truth (telecom: spec section 10; top-up: `requirements_2.md` section 11), writes `evaluation.json` per run; several runs (one per entity) are also combined |

## External sources (ClickHouse)

The datasets of a run can come from ClickHouse instead of the data root. The
tables are first copied to a local **snapshot** (a folder of Parquet datasets);
the run then works on that folder exactly as on `./data`. The agents and tools
never connect to the database.

```bash
uv sync --extra clickhouse           # once: installs the ClickHouse client

# copy and run in one step (the snapshot is reused if it already exists)
uv run jellybean run "find unusual subscribers" \
    --source ch_prod --from 2026-10-01 --to 2026-11-01 --sample 0.1

# or copy first, run later with the same options
uv run jellybean snapshot ch_prod --from 2026-10-01 --to 2026-11-01 --sample 0.1
uv run jellybean snapshots           # what is on disk
```

The source and its tables are declared in `config.yaml` (`sources`, see
[docs/config-reference.md](docs/config-reference.md#sources)); the user and
password come from `JELLYBEAN_SOURCE_<NAME>_USER` / `_PASSWORD` in `.env`.

| Option | Meaning |
| --- | --- |
| `--source NAME` | Read the datasets from this source. Without it: the data root. |
| `--from`, `--to` | Window on the tables with a `time_column`; the end is excluded. Required when a table has one. A date or `2026-10-01T06:00`, in the time zone of the column. |
| `--sample 0.1` | Copy 10% of the **entities** (chosen by a hash of `entity_column`) with **all** their rows, the same entities in every table and on every run. |
| `--refresh` | Copy again even if this snapshot exists. |

What to know:

- **ClickHouse does the filtering.** Only the rows of the window, of the
  sampled entities and of the configured columns leave it, already as Parquet
  streamed to disk. A table with a `time_column` is copied one day per file,
  and an interrupted copy continues where it stopped.
- **A snapshot is identified by what defines its content**: the tables'
  configuration, the window and the sample. The same request reuses it; a
  change of any of them makes a new one. Snapshots live in
  `./snapshots/<source>/<id>/` with a `snapshot.json` (queries, rows, date) and
  can be deleted by hand when no longer needed.
- **The run records its source** in `spec.yaml` (`source`) and in the report.
  `resume` needs the snapshot to be still on disk; `report` and the explorer
  do not.
- **Timestamps** are stored as the wall-clock time of the column's time zone,
  without a zone, so hours and days are those of the source on any machine.
  UUIDs, IP addresses and enums are stored as text.
- **A sample changes what counts mean.** Every row of a sampled entity is
  there, so per-entity profiles are complete; but counts over the whole table,
  rankings and the counterparts of an entity (who it calls) describe the copy,
  not the source.

## Results explorer

A local, read-only web view of the runs in the workspace, to navigate the
results of the runs made on the command line:

```bash
uv sync --extra ui      # once: installs Streamlit
uv run jellybean ui     # http://localhost:8501 (--port, --no-open)
```

| View | What it shows |
| --- | --- |
| Runs | Every run: instruction, primary dataset, entity, phase, status, findings and how many were confirmed. |
| Overview | The report with its plots, the confirmed spec (datasets, joins, grain, entity, time column) and the decisions taken at the control points. |
| Findings | Filter by verdict and type. For a finding: the features that set it apart, its rules, the critic's checks, its plots, its members with their features next to the population medians, and the raw events of any member. |
| Tables | Any table artifact or dataset: schema, filter (a SQL condition), sort and pages of 200 rows, plus a read-only SQL box over all the tables of the run. Rows can be downloaded as CSV. |
| Steps | Every tool call with its duration, status and error; time per tool and tokens per agent. |
| Snapshots | The local copies of external sources: window, sample, tables, rows, size and the runs that read each one. |

A run made with `--source` shows its window and sample on every page and reads
its raw tables from its snapshot. If the snapshot was deleted, the report,
findings and artifacts still open and the pages say the source rows are gone.

It never writes to a run, so it can stay open while a run is in progress
("Reload" re-reads the files). It listens on `localhost` only and shows the
data **as stored**: the "Mask personal data" switch hides identifiers the way
the agents see them, for when the screen is shared. Tables are paged and
filtered in DuckDB, so a table of millions of rows is never loaded whole.

## How a run works

1. **Interpretation.** The orchestrator resolves the datasets in the
   instruction and their roles (primary / enrichment).
2. **Discovery.** The profiler profiles everything and proposes the join,
   grain, entity and time column, with evidence. It also runs deterministic
   integrity checks (see below): invariants, references, duplicated keys,
   structural nulls and sign conventions.
3. **Control 1: confirm the spec.** The orchestrator confirms automatically
   only if every section-8 rule passes (coverage, key uniqueness, fan-out,
   score margin, no composite/N:M/temporal keys, no sparse key with
   non-structural nulls). Otherwise it asks you, showing the evidence.
4. **Cleaning plan.** The preparer proposes one justified operation per
   quality issue.
5. **Control 2: approve the cleaning.** The recommended option excludes fixes
   *inferred* from evidence (such as a unit conversion, or `flip_sign` for a
   systematic sign error), so nothing is rewritten on a guess without a human.
6. **Join and features.** The confirmed joins are materialised as left joins,
   with unmatched rows kept and flagged. Then per-entity features, graph
   features, the typed-event models below, scaling and PCA.
7. **Detection.** Several methods run, and their results are consolidated into
   at most 12 candidate findings.
8. **Critique.** One isolated critic per finding. Verdicts: `confirmed`,
   `explained` (real and stable, with a legitimate cause visible in the
   reference tables), `artifact`, `unstable`, `weak`. `set_verdict` refuses
   verdicts the evidence contradicts.
9. **Rework.** At most `max_rework_cycles` cycles; `delegate` enforces the limit.
10. **Analysis.** Confirmed findings are described.
11. **Control 3: review the findings.** Your feedback teaches the pattern
    registry. The recommended answer is "No feedback", so `--auto-approve`
    never teaches it.
12. **Report.** `report.md` in English, or in Portuguese for a Portuguese
    instruction. It contains the final spec, decisions, cleaning (applied and
    not), the join result with unmatched rows, findings by verdict and
    limitations. It is also written, as a partial report, when a run stops
    early (for example on the token budget).

**Integrity checks (discovery).** These rules are verified without any model,
inside DuckDB, on the distinct rows of the primary dataset. Their problems go to
the quality issues (and the cleaning plan) and never to the detector.

| Tool | What it finds |
| --- | --- |
| `discover_arithmetic_invariants` | relations between measures that hold on almost every row (`a + b = c`, `a = b`, `a >= 0`); violations are split into a **systematic** block (one category value on some dates, with the sign flip that explains it) and **scattered** rows |
| `classify_nulls` | whether nulls are **structural** (decided by another column, e.g. a type) or errors; structural nulls are recorded in `spec.yaml` and never imputed |
| `check_referential_integrity` | orphan references, self-references included |
| `find_duplicate_keys` | keys that should be unique but repeat in different rows (values repeated by rows of different types are shared by design and ignored) |
| `infer_sign_convention` | sign of a measure per category value, and inversions concentrated on a few dates |
| `discover_self_references`, `check_reciprocity` | columns referring to another column of the same dataset; rows `A -> B` without their `B -> A` counterpart |

Joins with a sparse key are measured on the rows with a non-null key. `test_join`
reports `null_key_share` and whether those nulls are structural. A sparse key
whose nulls are not structural needs a human at control 1. `spec.yaml` records
`references`, `invariants` and `structural_nulls` next to the joins.

At the console, answer with the option number or its text, or press Enter for
the recommended option. You can add a note after `|`.

**Row filters.** `--filter` takes a SQL condition whose columns are prefixed
with the dataset name:
```bash
uv run jellybean run "analisa os dados em traffic e enriquece-os com cust_segmentation" \
    --filter "traffic.date >= '2026-10-01' and traffic.date < '2026-10-03'" \
    --filter "cust_segmentation.segment = 'Business'"
```
- It is applied to the dataset's view, so every agent and tool of the run
  (profiling, cleaning, features, detection and the critic's checks) sees the
  same filtered rows. Filters on Hive partition columns such as `date` also
  skip the unneeded files.
- Before the run starts, each expression is checked by the same SQL guard as
  `run_sql`, and the number of remaining rows is printed. The run is refused if
  an expression refers to more than one dataset, uses an unknown column or
  keeps no rows.
- The filters are recorded in `spec.yaml` (`filters`): agents can read them but
  cannot change them. `resume` reapplies them, and the report and the
  evaluation show them. A filtered run cannot find anomalies outside its rows.

**Analysis entity.** Every analysis is about one entity: the column whose
values the rows belong to (a customer, a machine, a dealer). The features, the
detectors and the findings are per entity. Without `--entity`, the profiler
proposes it with evidence and control 1 confirms it, as before. With it:
```bash
uv run jellybean run "analisa os dados em traffic e enriquece-os com cust_segmentation" --entity traffic.msisdn
```
- Before the run starts, the column is checked (after the filters) and its
  activity is printed: `Entity traffic.msisdn: 20,000 entities, median 57
  events each (min 1, max 4,188)`. The run is refused if the dataset or column
  does not exist, the column has fewer than 2 values, or it has almost one
  value per row (more than 50% distinct, such as `traffic.call_id`):
  row-level analysis is not supported.
- The choice is recorded in `spec.yaml` (`entity.source: user`) and as a human
  decision. Control 1 accepts it without asking, and no agent can change it.
  The entity's dataset must be the primary dataset of the run.
- To analyse several entities in the same data (for example the customer and
  the agent who sells to them), run one analysis per entity and evaluate them
  together: `jellybean evaluate <run_a> <run_b>`. Each planted anomaly is
  counted only in the runs whose entity it belongs to ("not applicable"
  elsewhere), and the combined table shows the best run for each one.

**Live progress.** `run` and `resume` print each step as it happens: phase
changes, each agent starting and finishing (turns, tokens, cache share,
duration), each tool call with its duration (errors in red) and every control
point decision. Lines are indented by delegation, with the elapsed time at the
start. `--verbose` also shows every LLM turn (tokens, cache reads) and all tool
arguments. `--quiet` turns the view off. `generate-data` prints its steps too.
Only what the trace records is shown (arguments and sizes, never data).

```
[00:41] == Phase: cleaning ==
[00:41]   ▶ preparer: apply: apply the approved cleaning plan and materialise the confirmed joins.
[00:42]     · apply_cleaning_plan()  4.2s
[00:47]     · materialize_join(join_index=0)  0.7s
[00:49]   ✔ preparer done — 5 turns, 38k tokens (74% cached), 8.1s
[00:49]     ↳ outputs:
[00:49]       - clean_traffic_v1 (table, 1,546,248 rows)  workspace/<run_id>/artifacts/clean_traffic_v1.parquet
[00:49]       - enriched_traffic_v1 (table, 1,546,248 rows)  workspace/<run_id>/artifacts/enriched_traffic_v1.parquet
[00:49]       + 2 document(s) (json): jellybean outputs <run_id> --type json
```

**Overview plots.** `run_clustering` draws a cluster map
(`plot_clusters_<method>_vN.png`): the entities on the PCA plane, coloured by
cluster, with noise in grey and the small clusters (the candidate anomalous
groups) highlighted and labelled with their size. `plot_cluster_run` redraws it
for an older run. `report.md` embeds the cluster maps and a findings map
(`report_findings_overview.png`), which shows every finding's members on the
same plane, marked by verdict. The analyst also draws per-finding plots:
feature distributions, the finding on the PCA map, and time series.

**Outputs.** Every dataset or artifact a run creates is recorded in
`manifest.json`. That includes cleaned tables, the enriched table, features,
each detection run (`detection_hdbscan_v1`…), the members of each finding,
plots and documents. Each one has its id, type, row count, the agent and tool
that made it, and its lineage.
- When an agent finishes, the live view prints `↳ outputs:` with the datasets
  and plots it created, as clickable paths. `--verbose` lists the documents
  too.
- Each agent's structured result has `outputs: [{id, type, rows, path, tool}]`.
  `run_result.json` stores them by agent, under `outputs_by_agent`.
- At the end of a run the CLI prints all outputs grouped by agent, and
  `report.md` has an "Outputs" section with links.
- `jellybean outputs <run_id>` lists them at any time. Parquet outputs can be
  opened with DuckDB, pandas, or `run.duckdb`, which has a view per table
  artifact.
- Agents themselves only ever see artifact ids, never paths.

Each run writes `workspace/<run_id>/`: `spec.yaml`, `manifest.json`,
`artifacts/`, `findings/`, `run.duckdb`, `logs/trace.jsonl`, `report.md`,
`run_result.json` and (after `evaluate`) `evaluation.json`. The trace has one
line per LLM call (tokens, cache reads, duration) and per tool call
(arguments, output size, raw rows, duration). Tool outputs are never written to
it. A failed call also records its error message (cut, with values that look
like personal data masked), and the console prints that message under the
failed call. The full traceback of an unexpected error goes to
`logs/errors.log`.

**Cost.** A full run with Claude Haiku uses on the order of 1M tokens, most of
them in the per-finding critique. Prompt caching (`llm.prompt_caching`, on by
default for Anthropic and Bedrock) caches the stable prompt prefix between an
agent's turns. Cache reads count at 10% toward `limits.token_budget_per_run`.

## The dummy data

| Dataset | Grain | Notes |
| --- | --- | --- |
| `traffic` | one call (about 1.55 M, partitioned by `date=YYYY-MM-DD/hour=HH`, 720 files) | `msisdn` with a `+351` prefix |
| `cust_segmentation` | one customer (19,415) | no prefix; 3% of customers missing; 15 duplicated keys; `region_code` is a false join candidate |
| `orders` | one order (about 117 k) | |

Planted anomalies (traffic, cust_segmentation, orders):

| Id | Anomaly | Where |
| --- | --- | --- |
| A | night bursts of long, expensive international calls | traffic |
| B | hundreds of 1-3 s calls | traffic |
| C | 40 SIMs on 3 IMEIs and 2 cells | traffic |
| D | 30 customers sharing devices/addresses | **orders** |
| E | 6-hour volume spike in one cell | traffic |
| F | basic plan with a business profile, visible only after enrichment | traffic |

The data also contains quality problems and two traps for the critic: a day
whose `duration_sec` is in milliseconds (all 24 hourly files of that day), and
imputation-prone nulls. The profiler reports a scale shift that covers every
file of a partition as one issue for that partition (`date=…/`), and
`fix_units` can correct a whole partition. Because `traffic` is partitioned by
hour, a filter such as `--filter "traffic.hour between 8 and 18"` only reads the
matching files. The ground truth goes to
`./ground_truth/`, which the agents can never read: `run_sql` refuses it and
the DuckDB sandbox blocks it.

### Prepaid top-ups (`requirements_2.md`)

`topup` is a ledger of wallet movements: one row per movement, 16 movement
types, about 543k rows over 90 days, partitioned by `date=`. It comes with three
reference tables: `promotions` (24), `dealers` (600) and `voucher_batches`
(400). The customers are the same as in `traffic` (`msisdn` with `+351`), so
it joins `cust_segmentation` the same way. Each movement type fills its own
columns, and the others are left null by structure (not by error).

| Id | Anomaly | Entity |
| --- | --- | --- |
| T1 | 3 operators make 240 night credit adjustments on the same 15 customers, reason `OTHER` | `agent_user` |
| T2 | 35 customers repeat recharge, bonus, reversal, and keep the bonus (no clawback) | `msisdn` |
| T3 | promotion `P017` pays 100% instead of 10% for 36 hours (1,500 bonuses, a configuration error, not fraud) | `promo_id` |
| T4 | 2 voucher batches redeemed in 4 hours by 20 customers, 6 days before distribution | `voucher_batch_id` |
| T5 | 4 independent dealers: last-hour recharges, 70% of the value to 10 customers each | `dealer_id` |
| T6 | 3 collector accounts fed by 120 customers who recharge by card and are reversed days later | `msisdn` (graph) |
| T7 | 80 customers take an advance, buy a bundle and disappear | `msisdn` |
| T8 | 140 bonuses with the right percentage but not eligible (below the minimum or out of dates) | `msisdn` (needs `promotions`) |

The data also has:
- reconciliation problems R1-R5: 300 rows where `balance_after` is not
  `balance_before + amount`, 150 recharges credited twice with the same
  `payment_ref`, 100 reversals of movements that do not exist, 80 recharges above
  their face value, and 60 transfers sent but not received;
- quality problems: 0.5% duplicated rows, `movement_type` spelling variants, a
  day in another time zone, and reasons missing in 5% of adjustments;
- three traps: a legitimate campaign day (TR1, `explained`), 5 days of USSD
  debits with a positive sign (TR2, `artifact`), and retail chains with huge
  volumes (TR3, `explained`).

The ground truth adds `reconciliation_issues.parquet` and
`expected_spec_topup.yaml` (joins, grain, entities, invariant, structural
nulls), plus `topup` entries in `anomalies.parquet` (the entity, then every
movement involved), `quality_issues.json`, `traps.json` and
`anomaly_details.json`. These files are merged by dataset, so regenerating one
scenario keeps the truth of the others. The top-up generator has its own random
stream: adding it does not change the telecom data, and `--datasets topup`
gives the same ledger as the full generation.

```bash
uv run jellybean generate-data --datasets topup --with-readme   # only the ledger and its tables
uv run jellybean generate-data --datasets topup --opaque-codes  # numeric codes, no hints in names
```

Discovery on `topup` meets the criteria of `requirements_2.md` (checked by
`tests/test_topup_discovery.py`, with readable and opaque codes and no README):
- the four enrichment joins (sparse keys) and the self-reference are found;
- the balance invariant is found;
- R1-R5 are each reported by an issue of their own (recall and precision >= 0.9);
- the sign error (TR2) is reported as a systematic block apart from R1;
- no structural-null column is imputed.

**Typed-event features (phase 4).** When a type column decides what a row means
(`spec.structural_nulls` is not empty), the feature engineer adds:

| Tool | Features |
| --- | --- |
| `build_category_features` | per type: count, sum and weight of the measure; ratios between types (pairs taken from the self-reference, e.g. reversals per recharge) |
| `build_linked_event_features` | rows referred to by others (share, time to the reference) and **expected responses learnt from the data**: a type answered by another in most cases (an advance and its recovery), or a response expected after a sibling (a reversed recharge's bonus clawed back) |
| `build_sequence_features` | time between consecutive events of different types; inactivity after the last event |
| `build_conformity_features` | observed value vs a value expected from the reference tables. The agent writes the rule as one SQL expression (checked like `run_sql`, no queries); `ref_<column>` reads the row a row refers to |
| `build_interaction_graph` | graph of who transacts with whom (counterpart column): partners, component size, partner degree (stars) |

`flip_sign` (in `apply_transform` and the cleaning plan) inverts a numeric
column on one segment and date range, only on the rows that break the invariant.
With it approved, the TR2 rows are fixed and only R1's 300 rows still break the
balance invariant. `tests/test_topup_features.py` checks that each customer
anomaly (T2, T6, T7, T8) is separated from normal customers by at least one new
feature: 100% of its members beyond the normal 1st or 99th percentile.

**Detection per entity and the `explained` verdict (phase 5).** One analysis
per entity (`--entity topup.promo_id`, `topup.dealer_id`, `topup.msisdn`...):
- **Feature families.** `run_outlier_detector(..., families=["linked"])` uses
  only the features of some templates, so an anomaly that lives in a few
  features is not diluted among hundreds. Consolidation votes within a family,
  and each family has its own slot when the 12 findings are chosen.
- **Activity per entity.** `run_timeseries_anomaly(group_by=<entity>)` judges
  each group within its own active life, against a 7-day moving median scaled by
  the level shared by all groups, so a month-end peak shared by everyone is not
  an anomaly of one group. A separate burst test catches a group that lived its
  whole life in one bucket (a voucher batch used in hours). Sums are scaled by
  the value of one event.
- **Small populations** (e.g. 40 operators): each method flags at least the
  minimum group size.
- **`explain_by_reference`** (critic) finds a legitimate cause in two ways:
  1. the members are most of one category of a reference table (e.g. the
     large retail chains);
  2. the entity is the key of a reference table, and the activity follows every
     conformity rule built from that table (at most 1% of rows off, checked on
     the window's rows) inside its validity period (e.g. a campaign day within
     the campaign's dates and rules).
  Such a finding must be `explained`, never `confirmed`, and the report shows it
  in its own section with the explanation.

`tests/test_topup_detection.py` runs the whole chain (no LLM) for four entities:
- **promotions:** the misconfigured promotion (T3) is the only one confirmed, and
  the campaign day (TR1) is explained;
- **voucher batches:** T4 is confirmed;
- **dealers:** T5 is confirmed;
- **operators:** T1 is confirmed;
- **customers:** T2, T6 and T8 are confirmed with recall of at least 0.6.

A third outcome of `explain_by_reference` is `attributed_to`: the members'
rule violations come from one reference object that breaks the rule as a whole
(e.g. the customers of the misconfigured promotion). Such a group cannot be
`confirmed` (it is `weak`, pointing to the entity to analyse), so T3 is reported
once, at the promotion level.

Known limitation: T7 (an advance, then silence) is found as a group, but most of
its members have little activity, so the critic rules it `weak` (best recall in
confirmed findings 0.3).

## Top-up acceptance (`requirements_2.md` section 11)

A run analyses one entity, so the top-up scenario is evaluated over several runs:

```bash
uv run jellybean run "analisa os dados em topup e enriquece-os com promotions, dealers, voucher_batches e cust_segmentation" \
    --entity topup.promo_id --auto-approve          # and topup.voucher_batch_id, topup.dealer_id, topup.msisdn, topup.agent_user
uv run jellybean run "analisa os dados em topup e enriquece-os com dealers, voucher_batches e cust_segmentation" \
    --entity topup.msisdn --auto-approve            # without promotions: T8 must not be found
uv run jellybean evaluate <run_promo> <run_batch> <run_dealer> <run_customer> <run_agent> <run_no_promotions>
```

Each run reports:
- **Discovery:** the joins and the self-reference, the balance invariant, and
  R1-R5. For R1-R5, the recall and precision of the best issue are measured on
  the rows its SQL predicate selects.
- **Sign error (TR2):** whether it is a systematic block apart from R1, and
  whether `flip_sign` fixed it.
- **Cleaning plan:** whether any structural-null column was imputed.
- **Anomalies:** the recall of the anomalies of its entity, and where the others
  were lost.
- **T3:** whether it is a single promotion-level finding and not a group of
  customers.
- **Traps:** the verdicts of TR1, TR2 and TR3.
- **Hygiene.**

The combined evaluation adds two criteria that need several runs: at least 5 of
the 8 anomalies with recall ≥ 0.6, and T8 found only when `promotions` is joined.

`tests/test_topup_detection.py` (`-m slow`) runs this protocol without an LLM on
five entities plus the run without promotions, and checks every criterion of
section 11. The best recalls are T1 1.0, T2 0.8, T3 1.0, T4 1.0, T5 1.0, T6 1.0,
T7 0.3 and T8 0.88: 7 of the 8 anomalies are at or above 0.6.

## Acceptance (spec section 10)

`jellybean evaluate <run_id>` prints:
- join and spec correctness;
- recall per anomaly in confirmed findings, with **where each anomaly was lost**
  (not flagged / flagged but not a finding / a finding with another verdict /
  confirmed);
- entity precision, finding precision and precision in the top 100 by score;
- trap verdicts;
- planted quality problems detected;
- hygiene: largest tool output, raw rows, tokens and cache share;
- the acceptance checklist.

A real run with Claude Haiku on the default data (seed 42, `--auto-approve`)
gave:
- recall 1.0 for A, B, C, E and F;
- the join correct and `region_code` rejected;
- the unit-error trap ruled `artifact`;
- all 8 planted quality problems detected;
- no tool output over the limits.

Precision is the main quality goal left to improve: some contextual groups on
other attributes are also confirmed (entity precision about 0.6, finding
precision about 0.7).

Three criteria need extra runs:
- **F needs the enrichment:** run `jellybean run "analisa os dados em traffic"`
  (no enrichment) and evaluate: F should not be confirmed.
- **"Known" after feedback:** run `jellybean feedback <run_id> <finding>
  confirmed` on a confirmed finding, then run the same instruction again: the
  critic classifies that finding as `known`.
- **D lives in `orders`:** run `jellybean run "analisa os dados em orders"` and
  evaluate.

## Unknown-domain test

`scripts/generate_unknown_domain.py` generates a dataset that has nothing to do
with the telecom simulator. It produces industrial machine telemetry, about
505k hourly readings for 1,500 machines, plus a `machines` reference table
whose key has another format (`MCH-00123` vs `00123`). Planted in it:
- 5 anomalies: degrading bearings, a stuck sensor, a site power surge,
  machines consuming like another model (visible only after the join), and a
  dev firmware with errors;
- a decoy: legitimate maintenance;
- quality problems, including a day with pressure in kPa.

The private ground truth goes to `telemetry_truth.json`. The generator lives
in `jellybean.datagen.telemetry`; no tool or skill uses it.

```bash
uv run jellybean generate-data --type telemetry --out /tmp/unknown/data --truth /tmp/unknown/truth
# same thing: uv run python scripts/generate_unknown_domain.py --out /tmp/unknown/data --truth /tmp/unknown/truth
JELLYBEAN_DATA_ROOT=/tmp/unknown/data uv run jellybean run "analyse telemetry enriched with machines" --auto-approve
```

`jellybean evaluate` knows the telecom and top-up ground truths; telemetry runs
are checked by the test below (or by reading the report against
`telemetry_truth.json`).

`tests/test_unknown_domain.py` runs the whole tool chain on it (no LLM), with
every decision taken the way the SKILL.md files prescribe, and checks:
- discovery: the hyphenated-prefix join (auto-confirmed), grain, entity, time
  column and quality problems;
- the kPa day ruled `artifact`;
- recall of the anomalies (S1, S2, S5 ≥ 0.9; S4 ≥ 0.6) and the surge day;
- a precision floor;
- the tool output limits.

The known limitation is recorded as a strict `xfail`: legitimate maintenance is
confirmed. The `explained` verdict only finds causes in the reference tables,
and here the maintenance is visible only in the telemetry itself (a status
value), with no README to say it is planned. When that is fixed, the test
reports it.

The test found four hidden telecom assumptions, all now fixed generically:
- key prefixes containing `-`;
- a "low activity" rule that failed when every entity has the same number of
  events;
- tied scores being flagged;
- redundant or poorly chosen contextual features.

## Tests

```bash
uv run pytest            # everything that does not need an LLM (about 3 min)
uv run pytest -m llm     # end-to-end with the real LLM from .env (skipped if not configured)
uv run pytest -m slow    # long acceptance runs: top-up detection per entity (about 13 min)
uv run --with chdb pytest tests/test_sources_snapshot.py   # also runs the snapshot SQL on an embedded ClickHouse
uv run ruff check . && uv run ruff format --check .
```

| Level | Files |
| --- | --- |
| Unit: tool decorator, artifacts, shared tools, spec, keys/types, each skill's tools | `test_tool_decorator.py`, `test_artifacts.py`, `test_data_tools.py`, `test_spec.py`, `test_keys_and_types.py`, `test_profiler_tools.py`, `test_preparer_tools.py`, `test_feature_tools.py` |
| Contract: every tool returns a valid `ToolResult` ≤ 4000 chars and ≤ 20 rows | `test_contract.py` |
| Security: `run_sql` read-only, sandbox, no ground truth | `test_security.py` |
| Generator: determinism, planted anomalies, problems, traps | `test_generator.py` |
| Runtime with `FakeLLM`: loader, registry, loop, limits, delegation, isolation | `test_runtime.py`, `test_llm_client.py` |
| Flow with `FakeLLM`: controls 1-3, rework limit, scripted end-to-end run | `test_orchestrator_flow.py` |
| Phase checks on dummy data (no LLM) | `test_profiler_dummy.py`, `test_phase4_dummy.py`, `test_phase5_dummy.py`, `test_phase6.py` |
| Unknown domain: industrial telemetry, never seen by the system (no LLM) | `test_unknown_domain.py` |
| User-chosen entity (`--entity`), per-entity evaluation | `test_entity.py` |
| Top-up generator: sizes, invariant, structural nulls, T1-T8, R1-R5, traps, merged truth | `test_topup_generator.py` |
| Top-up discovery (no LLM): sparse joins, references, invariant, R1-R5, TR2, structural nulls | `test_topup_discovery.py` |
| Top-up features (no LLM): flip_sign, expected responses, separation of T2/T6/T7/T8 | `test_topup_features.py` |
| Top-up detection per entity (no LLM): T3/T4/T5 and T2/T6/T8 confirmed, TR1 explained | `test_topup_detection.py` |
| Domain-agnostic detection helpers (activity, ties, context features) | `test_detection_core.py` |
| External sources: snapshot queries, window, sample, reuse, resume, the run's source, explorer | `test_sources_snapshot.py` |
| End to end with a real LLM (`-m llm`) | `test_e2e_llm.py` |

## Project layout

```
.
├── pyproject.toml / uv.lock      # pinned dependencies
├── config.yaml                   # limits, thresholds and external sources (docs/config-reference.md)
├── .env.example                  # provider, model, key, paths
├── scripts/
│   ├── generate_dummy_data.py
│   ├── generate_unknown_domain.py   # = generate-data --type telemetry (unknown-domain test)
│   └── evaluate_run.py
├── src/jellybean/
│   ├── cli.py                    # `jellybean` command (typer)
│   ├── settings.py               # .env + config.yaml
│   ├── spec.py                   # spec.yaml schema, patching, join-confidence rules
│   ├── reporting.py              # report.md
│   ├── sources/                  # external sources: snapshot.py (copy to local Parquet), clickhouse.py (client)
│   ├── ui/                       # results explorer (`jellybean ui`): data.py (read-only access), app.py (Streamlit)
│   ├── evaluation.py             # comparison with the ground truth
│   ├── runtime/
│   │   ├── context.py            # active run (run id, folders, seed)
│   │   ├── tool_registry.py      # @tool decorator, ToolResult, schemas, timeouts, ToolRegistry
│   │   ├── cancellation.py       # stops a tool call: interrupts its DuckDB queries
│   │   ├── skill_loader.py       # SKILL.md -> agent definitions
│   │   ├── agent_runner.py       # tool-use loop, Runtime, delegation, token budget
│   │   ├── llm_client.py         # LLMClient, LangChain providers, prompt caching, FakeLLM
│   │   ├── human.py              # console / auto-approve / scripted answers
│   │   ├── progress.py           # live progress view (trace listener)
│   │   └── tracing.py            # logs/trace.jsonl, logs/errors.log
│   ├── skills/
│   │   ├── _shared/              # data tools, artifacts, catalog/sandbox, SQL guard, masking,
│   │   │                         # column stats, keys, spec tools, findings, detection, registry
│   │   ├── orquestrador/  profiler/  preparer/  feature_engineer/
│   │   └── detector/  critico/  analista/         # SKILL.md + tools.py each
│   └── datagen/                  # synthetic data: generator.py (telecom), topup.py (prepaid ledger),
│                                 # telemetry.py (unknown domain)
└── tests/
```

`data/`, `ground_truth/` and `workspace/` are generated and git-ignored.

## How `run_sql` stays safe

There are two independent layers:

1. **Static guard** ([sql_guard.py](src/jellybean/skills/_shared/sql_guard.py)).
   DuckDB's own parser must see exactly one `SELECT` statement (`WITH` and
   `UNION` included). Functions that run dynamic SQL or read raw files are
   refused, and so are literals that point at the ground truth.
2. **Sandboxed connection** ([catalog.py](src/jellybean/skills/_shared/catalog.py)).
   It sets `allowed_directories` to the data root plus the run folder,
   `enable_external_access=false`, disables extension auto-loading and sets
   `lock_configuration=true`. Even if a query got past the guard, DuckDB would
   refuse any file outside those folders and any attempt to change the settings.

Agents never see paths. Every dataset and table artifact is exposed as a view
with its own name, for example `SELECT count(*) FROM traffic`. Source-file
provenance is used only inside trusted tool code; the LLM sees aggregated,
relative file names.

## Adding a tool

1. Write a function in the skill's `tools.py` (or in `skills/_shared/` for a
   shared tool), with type hints and a Google-style docstring. The first
   paragraph becomes the description the LLM sees, and `Args:` becomes the
   parameter descriptions.
2. Decorate it with `@tool` (optionally `@tool(max_output_chars=..., timeout_s=...)`)
   and return `ToolResult.ok(summary, artifact_id=..., warnings=[...])` or
   `ToolResult.error("helpful message")`.
3. Put raw data rows only under `summary["rows"]`, and store large results as
   artifacts through `ArtifactStore`. Add a `seed: int | None = None`
   parameter if the tool uses randomness, and the run seed is injected for you.
4. Add the tool to the contract test list in `tests/test_contract.py`. It
   fails until every tool is listed.

The decorator validates arguments, turns exceptions into error results,
applies the timeout, enforces the size and row caps and writes the trace line,
so the tool body only does the computation. A shared tool becomes available to
an agent when the agent lists it in `shared_tools`.

## Adding a skill

Create `src/jellybean/skills/<name>/` with an `__init__.py`, a `tools.py`
(write and test the tools first) and a `SKILL.md` (written last, describing
what the tools actually do):

```markdown
---
name: <name>                      # must match the folder name
description: One line, shown to the orchestrator when it delegates.
max_iterations: 30                # optional, default from config.yaml
shared_tools: [get_spec, run_sql] # shared tools this agent may use
---
# Role
...
# Procedure
...
```

The runtime discovers the skill automatically. The agent sees only its own
tools, the shared tools it declares and `finish`, and the orchestrator can
`delegate` to it. Every agent run starts from a clean context (system prompt,
spec summary, task and artifact ids) and returns `status`, `summary`,
`artifact_ids`, `issues` and `requests`.

## License, authors and citation

Jellybean is licensed under the [Apache License 2.0](LICENSE); see also
[NOTICE](NOTICE) and [AUTHORS](AUTHORS). Source files carry an SPDX header
(`SPDX-License-Identifier: Apache-2.0`).

To cite the project, use [CITATION.cff](CITATION.cff) (GitHub shows a
"Cite this repository" button from it). Changes are recorded in
[CHANGELOG.md](CHANGELOG.md).
