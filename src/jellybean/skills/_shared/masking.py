"""Masking of personal data in row previews (``sample_rows``, exemplars).

A column is masked when its *name* matches a configured pattern or when most of
its sampled string *values* look like personal data (phone numbers, e-mails,
IBANs). Masked values are replaced by a stable salted hash, so the same value
maps to the same token inside a run (useful to see repetition) without
revealing it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

from jellybean.settings import MaskingConfig


class Masker:
    """Detect and mask personal-data columns."""

    def __init__(self, config: MaskingConfig, salt: str = "") -> None:
        self.config = config
        self.salt = salt
        self._name_res = [re.compile(p, re.IGNORECASE) for p in config.column_name_patterns]
        self._value_res = [re.compile(p) for p in config.value_patterns]

    def is_sensitive_name(self, column: str) -> bool:
        return any(r.search(column) for r in self._name_res)

    def is_sensitive_values(self, values: Sequence[Any]) -> bool:
        """True when enough non-null string values match a PII pattern."""
        strings = [v for v in values if isinstance(v, str) and v]
        if not strings:
            return False
        hits = sum(1 for v in strings if any(r.match(v.strip()) for r in self._value_res))
        return hits / len(strings) >= self.config.value_match_ratio

    def mask_value(self, value: Any) -> Any:
        if value is None:
            return None
        digest = hashlib.sha256(f"{self.salt}|{value}".encode()).hexdigest()[:10]
        return f"masked:{digest}"

    def mask_rows(self, columns: list[str], rows: list[list[Any]]) -> tuple[list[list[Any]], list[str]]:
        """Mask sensitive columns in a list of rows.

        Returns:
            The masked rows and the names of the masked columns.
        """
        sensitive: list[int] = []
        for i, col in enumerate(columns):
            if self.is_sensitive_name(col) or self.is_sensitive_values([r[i] for r in rows]):
                sensitive.append(i)
        if not sensitive:
            return rows, []
        masked = [[self.mask_value(v) if i in sensitive else v for i, v in enumerate(row)] for row in rows]
        return masked, [columns[i] for i in sensitive]
