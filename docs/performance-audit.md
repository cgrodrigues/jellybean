# Performance audit: memory and CPU

Review of `src/jellybean/` for code that does not scale to large datasets
(100M+ event rows, about 2.3M entities) on a machine with about 11 GB of RAM.

## Scope and method

- **Reviewed:** all of `src/jellybean/`, with priority on `skills/` and `skills/_shared/`.
- **How:** every file was read for the patterns listed in the task brief (unbounded fetches,
  `SELECT DISTINCT *`, statements in loops, uncapped combinatorial loops, repeated scans,
  full tables in pandas, unclosed resources, growing caches, per-file scans).
- **Measured:** the profiler tools, on a synthetic `subscribers_events` dataset of 6M rows,
  1.7M entities, 12 measures and a timezone-aware timestamp. The real dataset was not
  available on the development machine.
- **Measured in the second pass:** the shared pieces of the later stages (feature-table
  reads, effect sizes, signatures, ECOD, HBOS, scaling), on synthetic feature tables of
  1M entities and 120 features (about 1 GB per table in memory). The development machine
  has 11 GB of RAM, so 2.3M entities could not be tried.
- **Not measured:** whole runs of the feature, detection, critic and analyst stages on
  large data. Where a figure below is not in the measurement tables, it is an estimate from
  reading the code: one copy of a 2.3M x 200 `float64` feature matrix is about 3.7 GB.

Line numbers are the line of the function definition at the time of writing.

## Risk levels

| Level | Meaning |
| --- | --- |
| high | Can exhaust memory or run for many minutes at the target scale. |
| medium | Slow or wasteful, but bounded. |
| low | Minor. |

## What changed for every tool (Task 5)

These two changes do not remove any single finding, but they change how every one of them
fails. Before, a heavy tool could freeze the machine; now it returns an error.

| Change | Where | Effect |
| --- | --- | --- |
| DuckDB memory, thread and spill limits | `skills/_shared/catalog.py:111` `connect` | A query above `limits.duckdb_memory_limit` (default 4 GB) spills to `<run>/tmp`. Operators that cannot spill fail with an out-of-memory error instead of taking the machine down. |
| Real timeouts | `runtime/tool_registry.py:411` `_run_with_timeout`, `runtime/cancellation.py` | At the timeout the tool's queries are interrupted and its connections closed. A timed-out tool no longer keeps running under the agent's retry. |

**Limit of this protection:** it covers DuckDB only. Memory used by pandas, numpy and
scikit-learn in the same process is not capped by anything. Pure-Python work is stopped
only where the code checks for cancellation: the loops over blocks of columns, row groups
and scoring batches added in the second pass do; a single long library call (one model
fit) does not, and runs to its end after the agent got its timeout error.

## Measurements (profiler stage, 6M rows)

| Tool | Before | After |
| --- | --- | --- |
| `discover_arithmetic_invariants` | timed out at 300 s, 9.2 GB peak | 6.6 s, 2.9 GB peak |
| All profiler tools, DuckDB limited to 600 MB | `profile_table`, `infer_semantic_types`, `infer_grain` failed (out of memory) | all complete in 3 to 6 s each, under 0.6 GB peak |
| `discover_join_candidates`, `test_join` against a 1.7M-row dimension, 600 MB | not measured | 6.1 s and 0.5 s, 0.8 GB peak |

The 600 MB limit stands in for a dataset about seven times larger under the default 4 GB limit.

## High-risk findings fixed in the first pass

| File | Line | Function | Problem | Fix applied |
| --- | --- | --- | --- | --- |
| `skills/profiler/integrity.py` | 77 | `_working_rows` (all integrity checks) | `SELECT DISTINCT *` was an inline subquery recomputed by every statement (37 times in one call). | Materialised once per call; above `profiling.distinct_max_rows` the dataset is read as is, with a warning that exact duplicates were not removed. |
| `skills/profiler/integrity.py` | 341 | `discover_arithmetic_invariants` | 738 candidate relations as 1,476 filtered aggregates in one statement. | Candidates screened on a seeded sample, then counted exactly in batches (`profiling.sample_rows`, `profiling.candidate_batch`). |
| `skills/profiler/integrity.py` | 139 | `_cells` (was `_concentration`, `_systematic_segment`) | One or two scans per category column, per invariant. | One scan for all category columns. |
| `skills/profiler/integrity.py` | 252 | `classify_nulls` | One statement per column and determinant. | One statement per determinant for all columns. |
| `skills/profiler/integrity.py` | 906 | `infer_sign_convention` | One statement per category value; any column accepted as category. | One statement for all values; more than 200 values is refused. |
| `skills/profiler/integrity.py` | 1051 | `discover_self_references` | Distinct values of every identifier column materialised, all ordered pairs joined, no cap. | Pairs that cannot reach the coverage are dropped first; at most 16 identifier columns, with a warning. |
| `skills/profiler/integrity.py` | 1123 | `check_reciprocity` | Copied every column of the rows with a counterpart. | Copies only the columns it uses. |
| `skills/_shared/column_stats.py` | 245 | `compute_column_stats` | Exact `count(DISTINCT)` and `quantile_cont` for every column in one aggregate (keeps every value in memory); recomputed by about 10 call sites per dataset. | Cached per run for datasets and table artifacts. Above `profiling.exact_stats_max_rows`: approximate quantiles, and exact distinct counts computed column by column so they spill to disk. |
| `skills/_shared/column_stats.py` | 245 | `compute_column_stats` (samples) | One sample query per column. | `with_samples="classify"` samples only the columns whose classification reads samples. |
| `skills/_shared/column_stats.py` | 215 | `count_duplicate_rows` (used by `profile_table`, `infer_grain`) | `count(*)` over `SELECT DISTINCT *`, computed twice per dataset. | Rows grouped by a 64-bit row hash; only rows whose hash repeats are compared in full. Exact, and cached per run. |
| `skills/profiler/tools.py` | 382 | `infer_grain` | Up to 28 exact `SELECT DISTINCT a, b` scans. | Pairs that cannot be unique are dropped; the rest are screened in one scan with a sketch; only plausible pairs are counted exactly. |
| `skills/profiler/tools.py` | 956 | `test_join` | Row-level copies of both key columns, the row-level left join executed to count the fan-out, distinct left keys computed twice. | Each side reduced once to its distinct keys with counts; every metric derives from those two tables. |
| `skills/profiler/tools.py` | 956 | `test_join` | A full statistics run on the right side only to find validity columns by name. | Uses the schema. |
| `skills/profiler/tools.py` | 800 | `discover_join_candidates` | Per pair: up to 300,000 values fetched twice and 3 to 9 statements, each recomputing both distinct sets. No cap on pairs. | Samples, value sets and normalised sets computed once per column and reused; at most 12 key-like columns per side, with a warning. |
| `skills/profiler/tools.py` | 124 | `_scale_shift_by_file` | Exact median of every measure per file. | Approximate median above `profiling.exact_stats_max_rows`. |
| `skills/preparer/tools.py` | 207 | `_apply` | One full copy of the dataset per cleaning operation, all kept until the end. | Each step's input is dropped once the step is logged: one copy at a time. |
| `skills/preparer/tools.py` | 746 | `materialize_join` | Every duplicated key fetched into Python only to count them. | Counted in SQL. |
| `skills/feature_engineer/models.py` | 113 | `_open_events` | Connection left open when the copy of the events failed. | Closed on every failure. |
| `skills/_shared/detection.py` | 102 | `cluster_labels` (hdbscan) | Pairwise algorithm on every entity; does not finish at 2.3M. | Fitted on a seeded sample above `detection.cluster_max_rows`; rows outside the sample are left unclustered. |
| `skills/detector/tools.py` | 525 | `_group_entities` | Dense pairwise distance matrix with no cap on the pool (46,000 entities need 8.5 GB). | Only the `detection.group_max_pool` most deviating entities are grouped. |
| `skills/_shared/plots.py` | 73, 134 | `cluster_map`, `findings_map` | Every entity drawn; a figure leaked when drawing failed. | Population thinned to `detection.plot_max_points` (members and small clusters always drawn); figures closed on failure. |

## Second pass: stages after the profiler

The first pass deferred every high-risk finding outside the profiler. They are fixed here,
under one rule: **below a size threshold the result is the same as before** (the acceptance
tests did not change), and above it the tool samples or caps and says so in a warning.
`tests/test_pipeline_resources.py` keeps the replaced implementations as references and
checks that the new code gives the same values.

### Measurements (1M entities, 120 features)

| Operation | Before | After |
| --- | --- | --- |
| Effect sizes of one group (`describe_group`, each finding of `consolidate_findings`) | 108 s, 4.1 GB (both tables loaded; the machine was swapping) | 10 s and 0.95 GB for the first group, 0.6 s for the next ones |
| Signature of one group (`match_pattern_registry`, `register_pattern`) | 101 s, 4.6 GB | 2 s, 0.55 GB |
| HBOS scores | 136 s, 3.4 GB | 33 s, 2.1 GB |
| ECOD scores | no result after 15 minutes (out of memory, swapping) | 143 s, 2.7 GB |
| `scale_encode` | 51 s, 3.7 GB | 34 s, 1.8 GB |

The matrix itself is 0.96 GB: the detectors now peak at two to three times the matrix, and
reading a group costs about one row group whatever the size of the table.

### High-risk findings fixed in the second pass

| File | Function | Problem | Fix applied |
| --- | --- | --- | --- |
| `skills/_shared/findings.py` | `feature_tables` and its 11 call sites | Raw and scaled tables loaded in full for a few rows or columns. | `feature_source`, `read_features`, `read_numeric_matrix`: only the needed columns, and only the members, read one Parquet row group at a time. No tool calls `feature_tables` any more. |
| `skills/_shared/findings.py` | `effect_sizes` | Population copied and its medians recomputed for every finding. | `group_effect_sizes`: above `detection.exact_population_max_rows` the population statistics are computed once per table, a few columns at a time, and reused; only the group is read. Below it, unchanged. |
| `skills/_shared/findings.py` | `signature` | Both tables loaded for one centroid. | `group_signature` reads the members only. |
| `skills/_shared/artifacts.py` | `save_dataframe` | Tables written as one row group, so a filtered read decoded the whole table. | Row groups of 100,000 rows. |
| `skills/_shared/detection.py` | `outlier_scores` (ecod, hbos) | Several full-size matrices inside the libraries. | Run a few columns at a time above 20M cells (both add per-column scores; same result). |
| `skills/_shared/detection.py` | `outlier_scores` (lof) | Every row compared with a 50,000-row sample. | The sample shrinks so that rows x sample stays under a fixed number of pairs (6 billion since the third pass). |
| `skills/_shared/detection.py` | `outlier_scores` (autoencoder) | Trained on every row. | Above `detection.fit_max_rows`: trained on a seeded sample, scored in batches. |
| `skills/_shared/detection.py` | `context_defining_features`, `contextual_scores` | Six to seven full-size temporary matrices. | Computed one column at a time; only the chosen features are transformed. |
| `skills/_shared/detection.py` | `grouped_series_anomalies` | One frame per group built in a Python loop. | The grid is built in one vectorised step (`_life_grid`). The number of series points was already capped at 3M. |
| `skills/detector/tools.py` | `_load_matrix`, `run_outlier_detector`, `run_contextual_outliers` | Full tables read, then three copies made. | Only the numeric columns (of the requested families) are read, straight into one array. |
| `skills/detector/tools.py` | `run_graph_analysis` | Every edge into networkx. | Components on a sparse matrix (`component_table`); communities and centrality refuse graphs above `detection.graph_max_edges`. |
| `skills/detector/tools.py` | `consolidate_findings` | Both feature tables held; uncapped pairwise merge. | Only the pooled entities' rows are read; the `detection.max_candidates_per_run` largest groups of each run are merged, with a warning. |
| `skills/critico/tools.py` | `_redetect`, `stability_check` | Matrices re-read in every repetition; the same re-detection recomputed for every finding. | Matrices loaded once per check; what a run finds on a subsample is cached and reused by every finding of the run. |
| `skills/critico/tools.py` | `artifact_check` | 50 to 90 scans of the source per finding; feature tables loaded twice; cleaning logs loaded whole. | The finding's rows are materialised once; one statement for all columns; the dataset-wide facts are computed once per run; null shares come from the Parquet metadata; logs are counted in DuckDB. |
| `skills/critico/tools.py` | `_rule_on_windows`, `_rule_relation`, `_attribution` | Full-width `SELECT DISTINCT *` self-join, repeated per rule and key. | The referred rows keep only the columns the rule reads; the rows off a rule are materialised once. |
| `skills/critico/tools.py` | `inject_synthetic_anomalies` | Every method refitted on all rows plus a full copy. | Above `detection.fit_max_rows` the planted rows are judged against a seeded sample. |
| `skills/critico/tools.py` | `explain_by_reference`, `match_pattern_registry` | Both feature tables loaded. | Column-only and members-only reads. |
| `skills/analista/tools.py` | `fit_surrogate_rules` | Uncapped one-hot encoding and two trees on every row. | Above `detection.surrogate_max_rows` non-members: fitted on the members and a seeded sample, precision scaled back to the population; attributes with more than 100 values are not encoded. |
| `skills/analista/tools.py` | `describe_group`, `get_exemplars`, `make_plot`, `register_pattern` | Both feature tables loaded; a figure leaked if drawing failed. | Members-only and column-only reads; the embedding plot is thinned; figures are always closed. |
| `skills/feature_engineer/tools.py` | `build_entity_features` | Whole event table copied. | Only the columns the aggregations read are copied. |
| `skills/feature_engineer/tools.py` | `scale_encode` | Three to four full-size frames. | Scaled one column at a time from blocks of columns read from the file; written without assembling a frame. |
| `skills/feature_engineer/tools.py` | `reduce_dimensions` (umap) | UMAP on every row. | Above `detection.fit_max_rows`: fitted on a seeded sample, the rest projected in batches. |
| `skills/feature_engineer/tools.py` | `build_graph` | networkx and a Python loop over all entities. | Degrees, weighted degrees, components and clustering on sparse matrices (same values); refuses more than 50M edges. |
| `skills/feature_engineer/models.py` | `_open_events` and the four model tools | Whole event table copied on every call. | Only the columns each tool reads (for conformity rules: the columns named in the expressions). |
| `skills/feature_engineer/models.py` | `build_conformity_features` | Full-width `SELECT DISTINCT *` self-join. | Follows from the projection: the referred rows hold only the rule's columns. |
| `skills/feature_engineer/models.py` | `build_linked_event_features` | Sibling join quadratic per referenced row. | Siblings reduced to (referenced row, type) first; the per-entity count uses `EXISTS`. |
| `skills/feature_engineer/models.py` | `build_interaction_graph` | Every pair as Python tuples, networkx, Python loop. | Partner counts in DuckDB, components on a sparse matrix; refuses more than 50M pairs. |
| `skills/preparer/tools.py` | `_apply` (drop rows) | `EXCEPT ALL` between two full copies to log removed rows. | The removed rows are read with the operation's own predicate. |
| `skills/profiler/tools.py` | `correlation_summary` | Up to 105 exact pair distinct counts in one statement. | Eight pairs per statement. |

**Behaviour changes to know about** (all others give the same output as before):

- Two rows of the referred table with the same key and the same values in the columns a
  rule reads now count once (they used to count twice when they differed elsewhere).
- A row with two different types among its own duplicates is no longer excluded from being
  its own sibling in `build_linked_event_features`.
- `artifact_check` lists the example files of a scale shift in a fixed (sorted) order.

## High-risk findings still open

| File | Function | Problem | Why it is left |
| --- | --- | --- | --- |
| `skills/critico/tools.py` | `stability_check` | Each repetition refits every method on 80% of the entities. It is now done once per run, not once per finding, but it is still `stability_runs` times the cost of detection. | It is what the check measures. Lower `critic.stability_runs`, or the subsample, on very large populations. |
| `skills/detector/tools.py` | `run_contextual_outliers` (`all_features`) | The z-scores of every feature are kept and saved with the run. | They are the tool's output (consolidation groups on them). The default method keeps a few features only. |
| `skills/_shared/detection.py` | `outlier_scores` (isolation forest), `cluster_labels` (kmeans, gmm) | Every row is scored; `n_init=10`. | Memory is one matrix; scikit-learn scores in chunks. CPU only. |
| `skills/feature_engineer/models.py` | `build_linked_event_features`, `build_sequence_features` | A `DISTINCT` and three window functions over all events. | On the projected columns only, and they spill to disk under the DuckDB limit. |
| `skills/preparer/tools.py` | `_apply` (dedupe) | Full-width `DISTINCT *` or window, and `EXCEPT ALL` for its log. | Removing exact duplicates needs the full row; it spills to disk. |
| `skills/feature_engineer/models.py`, `tools.py` | `_extend`, `build_graph` | The feature table is rewritten through pandas for every extension. | One copy of the raw feature table per call; bounded, listed below as medium. |

What bounds none of the above: memory outside DuckDB has no cap. The fixes keep the peak
near one to three copies of the feature matrix, but a population that does not fit in RAM
as one matrix still needs a larger machine or a filter (`run`'s `--filter`).

## Third pass: timeouts for work that is not a query

Run `2026-11-17_003` (592,282 entities) showed that the timeout of the first pass only
covered DuckDB. `run_clustering(hdbscan)` was reported as timed out after 300 s, but the
call lasted 697 s and still wrote its artifacts; the agent repeated it and lost another
704 s.

**Cause, verified.** A model fit is one long call into compiled code. While it holds the
interpreter lock the thread that enforces the timeout cannot run: in a test,
`thread.join(1.0)` on a worker executing one C call returned after 4.0 s (when the work
ended), and after 1.0 s when the worker ran a Python loop.

| Change | Where | Effect |
| --- | --- | --- |
| Model fits in a worker process | `runtime/isolated.py`; `outlier_scores`, `cluster_labels`, `reduce_matrix` in `skills/_shared/detection.py` | Above `limits.isolate_above_rows` (20,000) rows the fit runs in a child process, with the same function, arguments and seed (tests check the results are identical). The caller only waits, so the timeout fires on time and kills the process. Data goes through `.npy` files mapped by the child, not through a pipe. |
| A timed-out call leaves nothing | `ArtifactStore.create`, `patch_spec`, `save_finding` | They check for cancellation before writing (and `create` removes the file if the call was cancelled while writing). Work that could not be interrupted and finishes later is discarded. |
| Per-tool timeouts | `limits.tool_timeouts` | For example `{run_clustering: 900}`. |
| HDBSCAN sample | `detection.cluster_max_rows` 200,000 -> 75,000; `params={"max_rows": n}` | See the measurements. |
| LOF reference sample | `_LOF_MAX_PAIRS` 25 -> 6 billion pairs | At 592k entities the sample goes from 42k to 10k rows: about a quarter of the 237 s measured in that run. |

**HDBSCAN fit time** (10 PCA components, `min_cluster_size=10`, development machine):

| Rows | Time |
| --- | --- |
| 25,000 | 3.8 s |
| 50,000 | 16.3 s |
| 100,000 | 72.9 s |
| 200,000 | 321.7 s |

Time grows with the square of the rows. The run above took about 700 s for 200,000 rows,
a little over twice these figures, so 75,000 rows should take about 90 s there and
100,000 about 160 s. 75,000 is the default: it stays inside a 300 s limit with room for
loading and plotting. A smaller sample finds fewer small clusters (a group needs
`min_cluster_size` members inside the sample); raise `cluster_max_rows` together with
`limits.tool_timeouts.run_clustering` when that matters.

**Not verified here:** the 592,282-entity table is not on the development machine.

## Medium and low findings (follow-ups)

| Risk | File | Line | Function | Problem | Proposed fix |
| --- | --- | --- | --- | --- | --- |
| medium | `skills/profiler/tools.py` | 196 | `profile_table` | Up to three extra scans per column (negative counts, category variants, histogram). | Fold them into single statements. |
| medium | `skills/profiler/tools.py` | 503 | `suggest_entities` | One grouped scan per candidate column. | Rank on the cached statistics; query the top few. |
| medium | `skills/profiler/tools.py` | 583 | `check_temporal_coverage` | Five scans where one hourly aggregate is enough. | Build one hourly table and derive the rest. |
| low | `skills/profiler/tools.py` | 583 | `check_temporal_coverage` | One garbage timestamp stretches the day range to millions of dates. | Clip the range or compute the gaps in SQL. |
| low | `skills/profiler/tools.py` | 1214 | `normalize_key` | Uncapped distinct table, several statements per strategy. | One statement for all strategies; refuse very large columns. |
| low | `skills/profiler/tools.py` | 956 | `test_join` | The list of duplicated right keys is capped at 1,000, so `duplicate_keys` is too. | Count them separately from the listed examples. |
| medium | `skills/preparer/tools.py` | 207 | `_apply` (normalize, impute, winsorize) | Mapping subquery evaluated three times; exact medians and quantiles. | Join the mapping once; approximate quantiles. |
| medium | `skills/preparer/tools.py` | 903 | `validate_constraints` | One scan per rule; `DISTINCT *` for duplicate rows. | One statement for the scalar rules; `count_duplicate_rows`. |
| medium | `skills/preparer/tools.py` | 846 | `diff_versions` | Two full statistics runs. | Now cached per artifact; compute only the reported fields. |
| medium | `skills/_shared/data_tools.py` | 142 | `run_sql` | The agent's query runs twice (preview and count). | Fetch one row more than the preview; count only when it is full. |
| medium | `skills/_shared/data_tools.py` | 198 | `sample_rows` | Hashes every column of every row to pick at most 20. | Order by a hash of the row position. |
| low | `skills/_shared/data_tools.py` | 142, 59 | `run_sql` (`save_as`), `list_datasets` | No row cap on saved results; one count per dataset. | Warn above a row count; cache the counts. |
| medium | `skills/_shared/catalog.py` | 82, 111 | `discover_datasets`, `connect` | Every tool call walks the data root three times and creates a view per dataset and artifact. | Cache the discovery per run; create views on demand. |
| low | `skills/_shared/artifacts.py` | 68, 87 | `_LOCKS`, `_load` | Manifest re-read on every lookup; one lock entry per run never removed. | Cache on file modification time; weak references. |
| low | `runtime/tool_registry.py` | 256 | `enforce_limits` | Re-serialises the result on each of up to 200 shrink steps. | Cut the largest list to its target length directly. |
| low | `entity.py` | 45 | `validate_entity` | Two scans, one with an exact distinct count. | One grouped subquery. |
| medium | `skills/_shared/detection.py` | 23, 102 | `outlier_scores` (isolation forest), `cluster_labels` (kmeans, gmm) | Every row scored; `n_init=10`. | `float32` input; mini-batch k-means above a row threshold. |
| medium | `skills/_shared/detection.py` | 225 | `_baseline` | Python lambdas per group. | Grouped rolling operations. |
| medium | `skills/detector/tools.py` | 314 | `run_timeseries_anomaly` | Three scans; groups not capped. | One aggregate into a temp table; a group cap. |
| medium | `skills/detector/tools.py` | 167, 556 | `run_clustering`, `consolidate_findings` | Python call per row; whole run tables read. | Vectorise; read only the needed columns. |
| medium | `skills/critico/tools.py` | 248, 280, 904 | `_rebuild_series`, `ensemble_agreement`, `rerun_on_version` | Series re-aggregated per finding; every run table read per finding; full refits. | Save the series as an artifact; aggregate in DuckDB. |
| low | `skills/critico/tools.py` | 989 | `compare_to_known_labels` | All label ids fetched into a set. | Count the overlap in DuckDB. |
| medium | `skills/feature_engineer/tools.py` | 67 | `build_entity_features` | Exact quantiles per measure; one scan per category and counterparty column; unbounded fetch of category levels. | Combined or approximate quantiles; `LIMIT` on the level check. |
| medium | `skills/feature_engineer/tools.py` | 707, 647 | `feature_summary`, `reduce_dimensions` (pca) | Correlation and PCA on every row. | Seeded sample. |
| medium | `skills/feature_engineer/models.py` | 82, 160 | `_extend`, `build_category_features` | Full feature table rewritten per extension; self-join to pick 12 pairs. | Join in DuckDB and write Parquet; semi-join first. |
| medium | `skills/_shared/plots.py` | 31, 194 | `projection_2d`, `cluster_run_plot` | PCA fallback on every row. | Seeded sample; store the projection. |
| low | `reporting.py` | 153, 176 | `trace_stats`, `_join_result` | Whole trace parsed into a list; a scan of the enriched table per report. | Stream the trace; store the counts at join time. |
| low | `evaluation.py`, `evaluation_topup.py` | 58, 50, 86 | `evaluate`, `_query`, `_reconciliation` | Whole run tables read; a connection per query. Offline tools on the dummy data. | Read the needed columns; one connection. |

## Checked and found sound

- **Connections** are closed in a `finally` everywhere (the one exception, `_open_events`, is fixed).
- **Trace listeners** are removed in a `finally` by the CLI; none accumulates.
- **Per-file scans:** datasets are always read through one DuckDB glob with projection pushdown; no Python loop over files.
- **Module-level state:** the new statistics cache is capped (64 sources, 4,096 column entries) and emptied when full.
- **No pandas loads of source data** in the profiler, preparer, shared data tools or runtime.

## Also noticed

- `tests/test_cli.py::test_generate_data_and_list_datasets` failed when it ran after a test
  that used the session fixture `phase4_pipeline`. The fixture yielded inside its active
  run, so that run stayed the current one for the rest of the session and the CLI's
  `datasets` command listed its data. The fixture now yields outside the run.
