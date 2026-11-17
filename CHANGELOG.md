# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Apache-2.0 license, `NOTICE`, `AUTHORS`, `CITATION.cff` and this changelog;
  SPDX license headers in the source, configuration and SVG files.
- GitHub Actions CI: secret scan (gitleaks), lint (ruff), tests (pytest),
  dependency audit (pip-audit) and vulnerability check (bandit); Dependabot for
  the Python dependencies and the actions.
- Tool errors are always shown: the message is printed on the console, recorded
  in `trace.jsonl` (masked) and the traceback goes to `logs/errors.log`.
- DuckDB memory, thread and spill limits (`limits.duckdb_*`), and real tool
  timeouts that interrupt the queries and close the connections; Ctrl+C returns
  the prompt.
- Size thresholds (`profiling.*`, `detection.*`) above which a step samples,
  approximates or refuses, with a warning. Below them results are exact.
- `docs/performance-audit.md`: memory and CPU review of the whole pipeline.
- `pytz`, so that DuckDB can return `TIMESTAMP WITH TIME ZONE` values.

### Changed

- The profiler's integrity checks, statistics and join tests scan the data a
  fixed number of times and cache column statistics per run.
- Feature tables are read by column and by entity instead of being loaded
  whole; graph features use sparse matrices instead of networkx.
- The critic computes the re-detection of a subsample once per run and reads
  the source dataset once per run in `artifact_check`.

### Fixed

- `check_temporal_coverage` failed on data covering a single day.
- `discover_arithmetic_invariants` timed out and exhausted memory on large tables.
- Join-test artifacts (raw key values) are no longer readable through
  `read_json_artifact`.
- A session test fixture left its run active for the following tests.

## [0.1.0] - 2026-11-12

### Added

- First version of the proof of concept: orchestrator, profiler, preparer,
  feature engineer, detector, critic and analyst agents; deterministic tools
  over DuckDB; data generators (telecom, top-up ledger, machine telemetry);
  CLI (`jellybean run`, `resume`, `generate-data`, `datasets`); report and
  evaluation against a ground truth.

[Unreleased]: https://github.com/cgrodrigues/jellybean/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cgrodrigues/jellybean/releases/tag/v0.1.0
