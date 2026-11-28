# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""External data sources, copied to local Parquet snapshots before a run.

The agents and tools never connect to a database: ``jellybean snapshot`` (or
``jellybean run --source``) copies the configured tables to a folder of Parquet
datasets, and the run reads that folder exactly as it reads ``./data``.
"""
