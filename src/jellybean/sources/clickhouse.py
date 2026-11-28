# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse client of the snapshot step (optional extra ``clickhouse``).

The server writes the Parquet itself; the bytes are streamed to disk, so the
size of a table never passes through Python objects.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from typing import Any

from jellybean.settings import SourceConfig
from jellybean.sources.snapshot import SnapshotError

_CHUNK = 1 << 20
# Strings as Parquet strings (not bytes), whatever the server default is.
_QUERY_SETTINGS = {"output_format_parquet_string_as_string": 1}
_TRUE = {"1", "true", "yes", "on"}


def connection_params(name: str, source: SourceConfig, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Connection of a source: ``config.yaml`` values, overridden and completed by the environment.

    ``JELLYBEAN_SOURCE_<NAME>_USER`` and ``_PASSWORD`` hold the credentials;
    ``_HOST``, ``_PORT``, ``_SECURE`` and ``_DATABASE`` override the configuration.
    """
    environ = os.environ if environ is None else environ
    prefix = f"JELLYBEAN_SOURCE_{name.upper()}_"

    def env(key: str) -> str | None:
        return environ.get(prefix + key) or None

    port = env("PORT")
    if port is not None and not port.isdigit():
        raise SnapshotError(f"{prefix}PORT must be a number.")
    secure = env("SECURE")
    return {
        "host": env("HOST") or source.host,
        "port": int(port) if port else source.port,
        "secure": secure.lower() in _TRUE if secure else source.secure,
        "database": env("DATABASE") or source.database,
        "username": env("USER"),
        "password": env("PASSWORD") or "",
    }


class ClickHouseClient:
    """Thin wrapper over ``clickhouse-connect`` with the three calls the snapshot needs."""

    def __init__(self, name: str, source: SourceConfig) -> None:
        try:
            import clickhouse_connect
        except ImportError as exc:
            raise SnapshotError("The ClickHouse source needs its client: run `uv sync --extra clickhouse`.") from exc
        # The driver logs every failure as well; the error raised here already says it.
        logging.getLogger("clickhouse_connect").setLevel(logging.CRITICAL)
        params = connection_params(name, source)
        self._where = f"{params['host']}:{params['port'] or 'default port'}"
        try:
            self._client = clickhouse_connect.get_client(
                **{k: v for k, v in params.items() if v is not None},
                send_receive_timeout=3600,
                client_name="jellybean",
            )
        except Exception as exc:
            raise SnapshotError(f"Cannot connect to ClickHouse at {self._where}: {_brief(exc)}") from exc

    def columns(self, database: str, table: str) -> list[tuple[str, str]]:
        try:
            rows = self._client.query(f"DESCRIBE TABLE `{database}`.`{table}`").result_rows
        except Exception as exc:
            raise SnapshotError(f"Cannot read the columns of {database}.{table}: {_brief(exc)}") from exc
        return [(str(row[0]), str(row[1])) for row in rows]

    def stream_parquet(self, sql: str) -> Iterator[bytes]:
        try:
            stream = self._client.raw_stream(sql, fmt="Parquet", settings=_QUERY_SETTINGS)
            try:
                while chunk := stream.read(_CHUNK):
                    yield chunk
            finally:
                stream.close()
        except SnapshotError:
            raise
        except Exception as exc:
            raise SnapshotError(f"ClickHouse query failed: {_brief(exc)}") from exc

    def close(self) -> None:
        self._client.close()


def _brief(exc: Exception) -> str:
    """First line of an error, without the stack trace the server appends."""
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0][:400]


def connect(name: str, source: SourceConfig) -> ClickHouseClient:
    return ClickHouseClient(name, source)
