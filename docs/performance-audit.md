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
- **Not measured:** everything after the profiler (features, detection, critic, analyst,
  report). Their memory figures below are estimates from reading the code:
  one copy of a 2.3M x 200 `float64` feature matrix is about 3.7 GB.

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
scikit-learn in the same process is not capped by anything, and pure-Python work cannot be
interrupted (the agent gets its error at the timeout, but the thread runs on).

## Measurements (profiler stage, 6M rows)

| Tool | Before | After |
| --- | --- | --- |
| `discover_arithmetic_invariants` | timed out at 300 s, 9.2 GB peak | 6.6 s, 2.9 GB peak |
| All profiler tools, DuckDB limited to 600 MB | `profile_table`, `infer_semantic_types`, `infer_grain` failed (out of memory) | all complete in 3 to 6 s each, under 0.6 GB peak |
| `discover_join_candidates`, `test_join` against a 1.7M-row dimension, 600 MB | not measured | 6.1 s and 0.5 s, 0.8 GB peak |

The 600 MB limit stands in for a dataset about seven times larger under the default 4 GB limit.

## High-risk findings fixed in this pass

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

## High-risk findings deferred

**Why they are deferred, for all rows of this table:**

- They are in the stages after the profiler, which the reported run never reached, and they
  could not be exercised on real data here.
- Their fixes change what the models see (sampling, approximations, fewer columns), so they
  change the detection results. The acceptance tests pin those results; each fix needs its
  own validation against them.
- Since Task 5 the DuckDB part of each fails with a clear error instead of hanging. The
  pandas part is still unbounded: these tools can still exhaust memory at 2.3M entities.

| File | Line | Function | Problem | Proposed fix |
| --- | --- | --- | --- | --- |
| `skills/_shared/findings.py` | 134 | `feature_tables` | Loads raw and scaled tables in full (about 7.4 GB) on every call; 11 call sites, most need one table, a few columns or only the members. | Separate loaders with column and member filters pushed into the Parquet read; store `float32`. |
| `skills/_shared/findings.py` | 176 | `effect_sizes` | Copies the population and recomputes its medians for every finding. | Compute population statistics once per feature table and store them; read only the members per finding. |
| `skills/_shared/findings.py` | 236 | `signature` | Callers load both tables to get the median of the members' rows. | Read the members only. |
| `skills/_shared/detection.py` | 23 | `outlier_scores` (ecod, hbos) | The libraries keep several full-size matrices (an estimated 15 to 22 GB for ECOD). | Both are column-separable: accumulate the score column by column. |
| `skills/_shared/detection.py` | 23 | `outlier_scores` (lof) | Fit is capped at 50,000 rows, but all rows are scored by brute force. | Score on the PCA embedding, or against a smaller reference. |
| `skills/_shared/detection.py` | 23 | `outlier_scores` (autoencoder) | Trains on every row for 20 epochs. | Fit on a seeded sample, score in batches. |
| `skills/_shared/detection.py` | 138, 173 | `context_defining_features`, `contextual_scores` | Six to seven full-size temporary matrices. | Compute on a seeded sample or column by column. |
| `skills/_shared/detection.py` | 330 | `grouped_series_anomalies` | A Python loop builds one frame per group; groups are not capped. | Build the grid vectorised; cap the groups with a warning. |
| `skills/detector/tools.py` | 67, 89, 440 | `_load_matrix`, `run_outlier_detector`, `run_contextual_outliers` | Full tables read, every method run on every row. | Read only the needed columns as `float32`; apply the fixes above. |
| `skills/detector/tools.py` | 257 | `run_graph_analysis` | Every edge into pandas, then networkx; pure-Python community detection. | Sparse connected components on integer codes; cap edges for community detection. |
| `skills/detector/tools.py` | 556 | `consolidate_findings` | Holds both feature tables; pairwise merge loop over uncapped candidates. | Load the columns of the current family only; cap candidates per type before merging. |
| `skills/critico/tools.py` | 99, 168 | `_redetect`, `stability_check` | Matrices re-read inside the repetition loop; every method refitted on 80% of the entities per repetition and per finding, although the subsamples are the same for all findings. | Load once; compute the re-detection once per repetition and reuse it for all findings. |
| `skills/critico/tools.py` | 379 | `artifact_check` | 50 to 90 scans of the source per finding (one or two per column), plus both feature tables loaded twice. | Materialise the finding's rows once; one statement for all columns; cache the population side per run. |
| `skills/critico/tools.py` | 636, 888, 829 | `_rule_on_windows`, `_rule_relation`, `_attribution` | Full-width `SELECT DISTINCT *` of the enriched table, self-joined, repeated per rule and per reference key. | Deduplicate only the key and the referenced columns; materialise the violating rows once per rule. |
| `skills/critico/tools.py` | 934 | `inject_synthetic_anomalies` | Full extra copy of the matrix; every method refitted on all rows. | Fit on a seeded sample plus the planted rows. |
| `skills/critico/tools.py` | 672, 1033 | `explain_by_reference`, `match_pattern_registry` | Both feature tables loaded for a few columns or one centroid. | Members-only and column-only reads. |
| `skills/analista/tools.py` | 125 | `fit_surrogate_rules` | Uncapped one-hot encoding and two trees on every row (an estimated 16 GB). | Train on the members plus a seeded sample; cap the encoded levels. |
| `skills/analista/tools.py` | 61, 203, 367 | `describe_group`, `get_exemplars`, `register_pattern` | Both feature tables loaded to describe a few rows. | Members-only reads and cached population statistics. |
| `skills/feature_engineer/tools.py` | 67 | `build_entity_features` | Copies the whole event table, all columns, into a temp table. | Copy only the columns the statements use. Not done here because the column list of every later statement has to be derived first; the copy now spills to disk. |
| `skills/feature_engineer/tools.py` | 565 | `scale_encode` | Three to four full-size frames alive at once. | Do the scaling in DuckDB and write Parquet directly. |
| `skills/feature_engineer/tools.py` | 647 | `reduce_dimensions` (umap) | UMAP on every row, single-threaded. | Fit on a seeded sample and transform the rest, or refuse above a cap. |
| `skills/feature_engineer/tools.py` | 406 | `build_graph` | Uncapped edges into pandas and networkx; Python loop over all entities. | Degrees in DuckDB; sparse components; an edge cap. |
| `skills/feature_engineer/models.py` | 113 | `_open_events` | Same full copy of the events, on every model tool call. | Project the columns; conformity rules reference columns chosen by the agent, so they must be parsed first. The copy now spills to disk. |
| `skills/feature_engineer/models.py` | 543 | `build_conformity_features` | Full-width `SELECT DISTINCT *` self-join. | Deduplicate the key and the referenced columns only. |
| `skills/feature_engineer/models.py` | 267 | `build_linked_event_features` | `SELECT DISTINCT` over all rows; the sibling join is quadratic per referenced row. | Deduplicate on the id; collapse siblings to (id, type) before joining. |
| `skills/feature_engineer/models.py` | 430 | `build_sequence_features` | Three window functions over all events, materialised beside the copy. | Project the needed columns. Now spills to disk. |
| `skills/feature_engineer/models.py` | 667 | `build_interaction_graph` | Every entity pair fetched as Python tuples, then networkx. | Partner counts in DuckDB; sparse components; an edge cap. |
| `skills/preparer/tools.py` | 207 | `_apply` (dedupe, log) | Full-width `DISTINCT *` or window for dedupe; `EXCEPT ALL` and one log row per affected source row. | Dedupe only repeated hashes or keys; derive removed rows from the operation's own predicate; aggregate the log. These now spill to disk. |
| `skills/profiler/tools.py` | 1331 | `correlation_summary` | Up to 105 exact `count(DISTINCT (a, b))` and 435 `corr()` aggregates in one statement. | Test dependencies with a grouped `min = max` check; compute correlations on a seeded sample. Deferred: the columns are capped (15 categories of at most 5,000 values, 30 measures), and it ran in 4.8 s under 600 MB on the test dataset. |

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
| medium | `skills/critico/tools.py` | 379 | `artifact_check` (cleaning logs) | Each cleaning log read whole into pandas. | Aggregate in DuckDB. |
| low | `skills/critico/tools.py` | 989 | `compare_to_known_labels` | All label ids fetched into a set. | Count the overlap in DuckDB. |
| medium | `skills/analista/tools.py` | 255 | `make_plot` | Loads both feature tables before branching; a figure leaks if drawing fails. | Load in the branch that needs it; close the figure on failure. |
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

- `tests/test_cli.py::test_generate_data_and_list_datasets` fails when it runs straight after
  `tests/test_phase6.py` and passes otherwise. It does so on the code before these changes
  too, so it is a test-order dependency, not a regression. Not fixed here.
