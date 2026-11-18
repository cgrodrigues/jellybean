# `spec.yaml` reference

`workspace/<run_id>/spec.yaml` is the source of truth of a run. It is written by the
orchestrator and the profiler, read by every agent, and validated on every change
(`src/jellybean/spec.py`). Agents change it with `propose_spec` (the profiler's whole
proposal) or `update_spec` (single fields).

## Fields set during discovery

| Field | Shape | Meaning |
| --- | --- | --- |
| `datasets` | `[{name, path, role}]` | `role` is `primary` (exactly one) or `enrichment`. |
| `joins` | `[{left, right, keys, key_normalization, status, ...}]` | `keys` is a list of `[left_column, right_column]` pairs; `status` is `proposed`, `confirmed` or `rejected`. May be empty. |
| `grain` | `{dataset, columns, unique}` | What identifies a row of the primary dataset. See below. |
| `entity` | `{column, alternatives, source, dataset}` | The column the analysis is about. `source: user` when set with `--entity`. |
| `time_column` | string or null | Event time. Without it there are no timestamp features and no time series. |
| `columns` | `{"<column>": {dataset, semantic_type}}` | `semantic_type` is one of `identifier`, `timestamp`, `measure`, `category`, `boolean`, `free_text`, `geo`, `unknown`. |
| `references`, `invariants`, `structural_nulls` | lists / mapping | Results of the integrity checks. |
| `evidence_artifacts` | `{"grain": id, "entity": id, ...}` | The artifacts behind the proposal. |

### `grain`

```yaml
grain:                      # a key: one row per combination of the columns
  dataset: orders
  columns: [order_id]
  unique: true              # default; may be left out

grain:                      # no unique key: event-level rows
  dataset: subscribers_events
  columns: []
  unique: false
```

`unique: false` means the table has no key: several rows per entity and timestamp are
legitimate (for example the originating and the terminating record of one call). It is a
valid outcome of discovery, not an error:

- `infer_grain` reports `unique_key: false` and `propose_spec` records the grain above;
- control 1 shows it as "no unique key", with the number of exact duplicate rows, and
  always asks a person to confirm it;
- the columns of a non-unique grain are never used to remove duplicates. The preparer
  refuses such a dedupe, and offers the removal of exact duplicate rows only as an
  operation that needs explicit approval;
- grain columns are analysed like any other column (the other party, the type, the
  device). Only a technical row id is excluded from the features: the single column of a
  unique grain, or a grain identifier with no repeated value.

`columns` may hold the closest combination found together with `unique: false`; it is then
informative only. Specs written before the field existed load with `unique: true`.

To change the grain by hand (the "Change something" answer at control 1), one
`update_spec` call is enough: `{"grain": {"columns": [], "unique": false}}`.

## Fields set later

| Field | Meaning |
| --- | --- |
| `cleaning_plan` | Plan artifact, status (`pending`, `proposed`, `approved`, `rejected`, `applied`) and approved operations. |
| `prepared` | Cleaned version of each dataset. |
| `enriched_artifact` | Primary dataset with the confirmed joins; absent when there is no join. |
| `features` | `table`, `dictionary`, `scaled`, `embedding`, `graph` artifacts. |
| `phase` | `discovery`, `cleaning`, `features`, `detection`, `critique`, `analysis`, `report`, `done`. |
| `decisions` | Human and automatic decisions with their evidence. Only the runtime writes them. |

## Feature windows

`build_entity_features` takes `windows`:

| Window | Features |
| --- | --- |
| `all` | Aggregates over the whole period (always). |
| `day` | Events per day (maximum, mean) and the maximum daily sum of each measure. |
| `hour` | Activity per category level, for categories with at most 15 values: the count of each level, its maximum count in one clock hour, its mean count per active hour and, with `day`, its maximum and mean count per day. |

The default is `["all", "day"]`. `hour` is for event logs with a column that says what kind
of event a row is: a burst of one type inside one hour shows in the hourly count and not
in the overall share. It is opt-in because on tables that are not event logs these
features repeat the shares and dilute the other features (measured on the top-up
acceptance scenario, where known groups stopped being found).

The window features are aggregated in DuckDB in two steps: counts per (entity, hour) with
one column per level, then one row per entity. The intermediate table has at most one
row per event and spills to disk under the DuckDB limits; above
`features.window_max_rows` (200 million) the window is skipped with a warning, and at
most `features.window_max_levels` (60) levels are expanded.
