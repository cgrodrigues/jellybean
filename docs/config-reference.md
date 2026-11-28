# `config.yaml` reference

`config.yaml` holds the limits and thresholds of a run and the external sources.
It is part of the experiment definition and is versioned with the code; secrets
and machine-specific paths live in `.env` (see the README). The file is
optional: every parameter has a default in
[src/jellybean/settings.py](../src/jellybean/settings.py), and a file only needs
the values it changes. Unknown keys are rejected, so a typo stops the command
instead of being ignored. Another file can be used with `JELLYBEAN_CONFIG_FILE`.

`null` means "no limit" or "use the default of the library", as said per
parameter.

## `run`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `default_seed` | `42` | Seed of a run when `--seed` is not given. Every step with randomness uses it. |

## `limits`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_iterations_per_agent` | `25` | Tool-use turns an agent may take before it is stopped. A skill can override it in its `SKILL.md`. |
| `token_budget_per_run` | `4000000` | LLM tokens per run. When it is used up the run stops cleanly and writes a partial report. |
| `tool_timeout_s` | `300` | Seconds a tool call may take. At the timeout its queries are interrupted and its worker process is killed. |
| `tool_timeouts` | `{}` | Timeout of single tools, in seconds, overriding `tool_timeout_s`: `{run_clustering: 900}`. |
| `max_rework_cycles` | `2` | Times the critic may send the work back for rework. |
| `max_consecutive_tool_errors` | `6` | An agent whose tool calls fail this many times in a row is stopped and reports what blocked it (minimum 2). |
| `duckdb_memory_limit` | `4GB` | Memory of each DuckDB connection. Above it queries spill to `workspace/<run_id>/tmp`. `null` = 80% of the RAM. |
| `duckdb_threads` | `4` | Threads of each DuckDB connection. `null` = all cores. |
| `duckdb_max_temp_size` | `50GB` | Most data a connection may spill to disk. `null` = DuckDB default. |
| `isolate_above_rows` | `20000` | Model fits (outlier methods, clustering, dimensionality reduction) on more rows run in a worker process that the timeout can kill. `null` = always in-process. |

## `tool_output`

What a tool may return to the model. These caps keep data out of the LLM context.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_output_chars` | `4000` | Hard cap on the size of a serialised tool result (minimum 500). |
| `max_raw_rows` | `20` | Hard cap on the raw data rows any tool returns. |

## `profiling`

Sizes above which the profiler's checks approximate instead of being exact.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `distinct_max_rows` | `5000000` | Above this many rows the integrity checks run on the raw rows; exact duplicate rows are then not removed first. |
| `exact_stats_max_rows` | `5000000` | Above this many rows column quantiles are approximate. Distinct counts stay exact. |
| `sample_rows` | `200000` | Candidate relations between datasets are screened on a sample of about this size first. |
| `candidate_batch` | `60` | Candidate relations tested per SQL statement (1 to 500). |

## `features`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `window_max_rows` | `200000000` | Rows of the intermediate (entity, hour) or (entity, day) table of the per-level window features. Above it the window is skipped with a warning. |
| `window_max_levels` | `60` | Category levels expanded into window features, over all columns. |

## `llm`

Parameters of the model calls. The provider, the model and the key are in `.env`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_tokens` | `4096` | Most output tokens per model call. |
| `temperature` | `null` | `null` = the provider's default (some reasoning models reject a temperature). |
| `request_timeout_s` | `180` | Seconds to wait for a model answer. |
| `max_retries` | `3` | Retries of a failed model call. |
| `prompt_caching` | `true` | Cache the stable prefix of the prompt between the turns of an agent (anthropic, bedrock; OpenAI caches by itself). |

## `join_confidence`

A join between two datasets is confirmed without asking only if **all** of these
hold; otherwise it goes to the first control point.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_coverage` | `0.95` | Least share of primary rows that find a match. |
| `min_right_uniqueness` | `0.999` | Least uniqueness of the key on the enrichment side. |
| `max_fanout` | `1.01` | Most rows after the join divided by rows before it. |
| `min_score_margin` | `0.15` | Least margin of the best candidate key over the second best. |

## `pattern_registry`

How a finding is compared with the patterns confirmed in earlier runs.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `known_threshold` | `0.85` | Similarity at or above this: the finding is a known pattern. |
| `variant_threshold` | `0.60` | Similarity at or above this: a variant of a known pattern. Below: new. |

## `detection`

Thresholds of the detectors.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `contamination` | `0.01` | Share of entities each outlier method flags. |
| `contextual_contamination` | `0.03` | Share flagged inside each context group. Higher than the global rate because globally extreme entities also take contextual slots. |
| `min_votes` | `2` | Outlier methods that must agree before an entity is pooled. |
| `min_group_size` | `3` | Smallest group of entities that becomes a finding. |
| `group_distance` | `0.5` | Cosine distance used to group the pooled outliers. |
| `max_cluster_share` | `0.02` | Clusters with more than this share of the entities are "normal", not findings. |
| `max_findings` | `12` | Most candidate findings sent to the critic. |
| `findings_per_family` | `3` | Feature-family groups: the best of each family that are kept. |
| `max_family_findings` | `16` | Cap on the family findings, on top of `max_findings`. |
| `merge_overlap` | `0.8` | Two findings are merged when this share of the smaller one is in the other. |
| `ts_threshold` | `6.0` | Robust z-score above which a point of a time series is an anomaly. |

Size caps of the steps whose cost grows faster than the data. Above a cap the
step samples, approximates or refuses, and says so; below it the result is exact.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `lof_max_rows` | `50000` | Quadratic methods (LOF) work on a sample above this size. |
| `cluster_max_rows` | `75000` | HDBSCAN is fitted on a seeded sample above this many entities (its time grows with the square of the rows). |
| `group_max_pool` | `10000` | Most pooled outliers grouped by pairwise distance; the most deviating ones are kept. |
| `plot_max_points` | `50000` | Population points drawn in a scatter plot. The members of a finding are always drawn. |
| `fit_max_rows` | `500000` | Above this, outlier models are fitted on a seeded sample and score the rest in batches. |
| `exact_population_max_rows` | `500000` | Above this, a group is compared with statistics of the whole population (computed once), not of the population outside the group. |
| `surrogate_max_rows` | `200000` | Non-members sampled to fit the analyst's surrogate rules. |
| `max_candidates_per_run` | `500` | Largest groups of one detection run kept before merging. |
| `graph_max_edges` | `2000000` | Community detection and centrality refuse graphs with more edges. |

## `critic`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_stability` | `0.6` | A finding below this stability cannot be `confirmed`. |
| `stability_runs` | `5` | Repetitions with other seeds and subsamples (2 to 20). |
| `stability_budget_share` | `0.6` | Share of the tool timeout `stability_check` may use; it then returns what it has. |
| `stability_max_rows` | `50000` | Entities used to re-detect a finding: a seeded sample plus the finding's members. |
| `artifact_min_share` | `0.5` | Share of a finding explained by one source (a file, nulls, a cleaning operation) to suspect an artifact... |
| `artifact_min_lift` | `5.0` | ...together with this lift of that share over the population. |

## `masking`

Which columns are personal data. Their values are replaced by stable tokens in
everything an agent sees.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `apply_to_run_sql` | `true` | Also mask those columns in the rows `run_sql` previews. |
| `value_match_ratio` | `0.5` | A text column is personal data when this share of its sampled values matches a value pattern. |
| `column_name_patterns` | phone, e-mail, name, address, IBAN, IMEI/IMSI, IP, `customer_id`... | Case-insensitive regular expressions on column names. Setting the list replaces the default one. |
| `value_patterns` | phone-like numbers, e-mail addresses, IBAN-like strings | Regular expressions on values. Setting the list replaces the default one. |

## `sources`

External databases whose tables are copied to a local snapshot before a run
(`jellybean snapshot <source>`, `jellybean run --source <source>`). The section
is a mapping from a source name to its definition; it is empty by default and
then the datasets come from the data root only. Names of sources, tables and
columns may only have letters, digits and `_`.

```yaml
sources:
  ch_prod:                           # used as --source ch_prod
    type: clickhouse
    host: clickhouse.internal
    port: 8443
    secure: true
    database: telco
    tables:
      subscribers_events:            # an events table
        time_column: event_ts        # --from/--to filter on this column
        entity_column: msisdn        # --sample picks entities by this column
      subscribers:                   # a dimension table
        entity_column: msisdn        # with --sample, only the sampled subscribers
      plans:                         # a small table, copied whole
        table: tariff_plans          # its name in ClickHouse
        columns: [plan_id, name, monthly_fee]
        where: "active = 1"
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `type` | `clickhouse` | Kind of source. Only `clickhouse` exists. |
| `host` | `localhost` | Server address (HTTP interface). |
| `port` | `null` | `null` = the default of the interface (8123, or 8443 with `secure`). |
| `secure` | `false` | Use HTTPS. |
| `database` | `default` | Database of the tables. |
| `tables` | *(required)* | The tables to copy: at least one. Each key is the **dataset name** the run sees. |

Per table (`{}` or nothing = copy every column and every row):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `table` | the key | Table name in the source, when it differs from the dataset name. |
| `time_column` | `null` | A `Date` or `DateTime` column. The window (`--from`, `--to`, end excluded) filters on it and is then required; the table is copied one day per file. Without it the table ignores the window. |
| `entity_column` | `null` | The column identifying the entity. With `--sample` only the rows of the sampled entities are copied. Use the same values in every table (the same `msisdn`), so the same entities are chosen. Without it the table ignores the sample. |
| `columns` | all | Columns to copy. |
| `where` | `null` | A fixed row filter, in ClickHouse SQL, on the table's columns. |

The user and the password are **not** configured here. They come from the
environment (or `.env`), named after the source in upper case:

| Variable | Meaning |
| --- | --- |
| `JELLYBEAN_SOURCE_<NAME>_USER` | User name. |
| `JELLYBEAN_SOURCE_<NAME>_PASSWORD` | Password. |
| `JELLYBEAN_SOURCE_<NAME>_HOST`, `_PORT`, `_SECURE`, `_DATABASE` | Optional: override the values of `config.yaml` on this machine. |

How a snapshot is taken and what is stored is in the README, section
"External sources (ClickHouse)".
