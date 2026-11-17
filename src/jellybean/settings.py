# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Application settings.

Two sources are combined:

* **Environment / ``.env``** (``EnvSettings``): provider, model, credentials and
  filesystem roots. These differ between machines and may contain secrets.
* **``config.yaml``** (``AppConfig``): limits and thresholds. These are part of
  the experiment definition and are versioned with the code.

Every field has a documented default so the system works without a
``config.yaml``; the file only overrides values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Supported LLM providers. The runtime (phase 3) maps each to an LLMClient.
Provider = Literal["anthropic", "openai", "bedrock"]


class EnvSettings(BaseSettings):
    """Settings read from environment variables prefixed with ``JELLYBEAN_``.

    Values are also loaded from a ``.env`` file in the current directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="JELLYBEAN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Provider = "anthropic"
    # No default model on purpose: the model must be chosen explicitly.
    model: str | None = None
    # SecretStr keeps the key out of logs and reprs.
    api_key: SecretStr | None = None
    data_root: Path = Path("./data")
    workspace_root: Path = Path("./workspace")
    ground_truth_root: Path = Path("./ground_truth")
    config_file: Path = Path("./config.yaml")


# --------------------------------------------------------------------------- #
# config.yaml sections                                                        #
# --------------------------------------------------------------------------- #
class _Section(BaseModel):
    """Base for config sections: unknown keys are rejected to catch typos."""

    model_config = ConfigDict(extra="forbid")


class RunConfig(_Section):
    default_seed: int = 42


class LimitsConfig(_Section):
    max_iterations_per_agent: int = Field(25, ge=1)
    token_budget_per_run: int = Field(4_000_000, ge=1)
    tool_timeout_s: float = Field(300, gt=0)
    max_rework_cycles: int = Field(2, ge=0)
    # DuckDB resources, per connection (a tool normally opens one). Above the memory
    # limit DuckDB spills to `<run>/tmp` instead of exhausting the machine.
    duckdb_memory_limit: str | None = "4GB"  # null = DuckDB default (80% of RAM)
    duckdb_threads: int | None = Field(4, ge=1)  # null = DuckDB default (all cores)
    duckdb_max_temp_size: str | None = "50GB"  # cap on spilled data; null = DuckDB default
    # Timeout of single tools, in seconds, overriding tool_timeout_s: {"run_clustering": 900}.
    tool_timeouts: dict[str, float] = Field(default_factory=dict)
    # Model fits (outlier methods, clustering, dimensionality reduction) on more rows than
    # this run in a worker process that is killed at the timeout; null = always in-process.
    isolate_above_rows: int | None = Field(20_000, ge=0)


class ProfilingConfig(_Section):
    """Limits that keep the profiler's checks cheap on large datasets."""

    # Above this many rows the integrity checks run on the raw rows instead of a
    # materialised copy of the distinct rows (exact duplicates are then not removed).
    distinct_max_rows: int = Field(5_000_000, ge=0)
    # Above this many rows column statistics use approximate quantiles, and distinct counts
    # (still exact) are computed column by column so they spill to disk.
    exact_stats_max_rows: int = Field(5_000_000, ge=0)
    # Candidate relations are screened on a sample of about this many rows first.
    sample_rows: int = Field(200_000, ge=100)
    # Candidate relations tested per statement (two filtered aggregates each).
    candidate_batch: int = Field(60, ge=1, le=500)


class FeaturesConfig(_Section):
    """Size caps of the feature engineer."""

    # Rows of the intermediate (entity, hour) or (entity, day) table of the per-level
    # window features; above it the window is skipped with a warning.
    window_max_rows: int = Field(200_000_000, ge=1000)
    # Category levels expanded into window features, over all columns.
    window_max_levels: int = Field(60, ge=1)


class ToolOutputConfig(_Section):
    max_output_chars: int = Field(4000, ge=500)
    max_raw_rows: int = Field(20, ge=1)


class JoinConfidenceConfig(_Section):
    min_coverage: float = 0.95
    min_right_uniqueness: float = 0.999
    max_fanout: float = 1.01
    min_score_margin: float = 0.15


class PatternRegistryConfig(_Section):
    known_threshold: float = 0.85
    variant_threshold: float = 0.60


class LLMConfig(_Section):
    """Model call parameters (provider/model/key live in ``.env``)."""

    max_tokens: int = Field(4096, ge=256)
    # None = provider default (some reasoning models reject a temperature).
    temperature: float | None = None
    request_timeout_s: float = Field(180, gt=0)
    max_retries: int = Field(3, ge=0)
    # Prompt caching (anthropic, bedrock): the stable prefix (tools, system prompt,
    # earlier turns) is cached between the turns of an agent. OpenAI caches automatically.
    prompt_caching: bool = True


class DetectionConfig(_Section):
    lof_max_rows: int = Field(50_000, ge=100)
    contamination: float = Field(0.01, gt=0, lt=0.5)  # share of entities flagged by an outlier method
    # Share flagged inside each context group. Higher than the global rate because
    # globally extreme entities also use contextual slots (they are excluded later).
    contextual_contamination: float = Field(0.03, gt=0, lt=0.5)
    min_votes: int = Field(2, ge=1)  # outlier methods that must agree to pool an entity
    min_group_size: int = Field(3, ge=2)  # smallest group that becomes a finding
    group_distance: float = Field(0.5, gt=0, lt=2)  # cosine distance threshold to group outliers
    max_cluster_share: float = Field(0.02, gt=0, le=0.5)  # clusters larger than this are "normal"
    max_findings: int = Field(12, ge=1)  # cap on candidate findings (critic cost)
    # Feature-family groups (run_outlier_detector families=...): the best few per family,
    # with their own cap on top of max_findings.
    findings_per_family: int = Field(3, ge=1)
    max_family_findings: int = Field(16, ge=0)
    merge_overlap: float = Field(0.8, gt=0, le=1)  # merge when this share of the smaller group is shared
    ts_threshold: float = Field(6.0, gt=0)  # robust z for time-series anomalies
    # Size caps of the steps whose cost grows faster than the number of entities.
    # HDBSCAN is fitted on a seeded sample above this. Its time grows with the square of the
    # rows (16 s at 50k, 73 s at 100k, 322 s at 200k on 10 dimensions): 75k keeps it well
    # inside the default tool timeout.
    cluster_max_rows: int = Field(75_000, ge=1000)
    group_max_pool: int = Field(10_000, ge=100)  # pooled outliers grouped by pairwise distance (n^2 memory)
    plot_max_points: int = Field(50_000, ge=1000)  # population points drawn in a scatter plot
    # Above this many entities: outlier models are fitted on a seeded sample and score
    # the rest in batches; contextual features are chosen on a sample.
    fit_max_rows: int = Field(500_000, ge=1000)
    # Above this many entities a group is compared with the statistics of the whole
    # population (computed once), not with the population outside the group.
    exact_population_max_rows: int = Field(500_000, ge=0)
    surrogate_max_rows: int = Field(200_000, ge=1000)  # non-members used to fit surrogate rules
    max_candidates_per_run: int = Field(500, ge=10)  # largest groups of a detection run kept for merging
    graph_max_edges: int = Field(2_000_000, ge=1000)  # community detection and centrality refuse more edges


class CriticConfig(_Section):
    min_stability: float = Field(0.6, ge=0, le=1)  # below this a finding cannot be "confirmed"
    stability_runs: int = Field(5, ge=2, le=20)
    artifact_min_share: float = Field(0.5, gt=0, le=1)  # share of the finding explained by a source
    artifact_min_lift: float = Field(5.0, ge=1)  # lift of that share over the population


class MaskingConfig(_Section):
    apply_to_run_sql: bool = True
    value_match_ratio: float = Field(0.5, gt=0, le=1)
    column_name_patterns: list[str] = Field(
        default_factory=lambda: [
            "msisdn",
            "phone",
            "mobile",
            "imei",
            "imsi",
            "iccid",
            "e_?mail",
            "(^|_)(first_|last_|full_|customer_)?name$",
            "address",
            "iban",
            "passport",
            "(^|_)nif$",
            "(^|_)ssn$",
            "(^|_)ip(_address)?$",
            "destination_number",
            "customer_id",
        ]
    )
    value_patterns: list[str] = Field(
        default_factory=lambda: [
            r"^\+?\d{9,15}$",
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$",
        ]
    )


class AppConfig(_Section):
    """Root of ``config.yaml``."""

    run: RunConfig = Field(default_factory=RunConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    tool_output: ToolOutputConfig = Field(default_factory=ToolOutputConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    join_confidence: JoinConfidenceConfig = Field(default_factory=JoinConfidenceConfig)
    pattern_registry: PatternRegistryConfig = Field(default_factory=PatternRegistryConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)


def load_config(path: Path) -> AppConfig:
    """Load ``config.yaml``; fall back to defaults when the file is absent."""
    if not path.exists():
        return AppConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)


class Settings(BaseModel):
    """Resolved settings: environment plus configuration, with absolute paths."""

    model_config = ConfigDict(frozen=True)

    env: EnvSettings
    config: AppConfig

    @property
    def data_root(self) -> Path:
        return self.env.data_root.resolve()

    @property
    def workspace_root(self) -> Path:
        return self.env.workspace_root.resolve()

    @property
    def ground_truth_root(self) -> Path:
        return self.env.ground_truth_root.resolve()

    @classmethod
    def build(cls, env: EnvSettings | None = None, config: AppConfig | None = None) -> Settings:
        """Build settings, loading anything that was not supplied explicitly."""
        env = env or EnvSettings()
        config = config or load_config(env.config_file)
        return cls(env=env, config=config)


def load_env_file(path: Path = Path(".env")) -> None:
    """Export ``.env`` into ``os.environ`` without overriding real variables.

    pydantic-settings only reads ``JELLYBEAN_*`` fields from ``.env``; libraries
    such as boto3 (Bedrock) need ``AWS_*`` variables in the process environment.
    """
    from dotenv import load_dotenv

    if path.exists():
        load_dotenv(path, override=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached after the first call)."""
    return Settings.build()


def reset_settings() -> None:
    """Drop the cached settings (used by tests and after editing ``.env``)."""
    get_settings.cache_clear()
