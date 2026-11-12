"""Join-key normalisation strategies.

A strategy is a named, deterministic SQL transformation applied to a key column
(always after casting to VARCHAR) on **both** sides of a join. Using the same
transformation on both sides keeps joins symmetric: e.g. stripping ``+351``
leaves values without the prefix unchanged.

Strategy names:

* ``none``                  value as text
* ``trim``                  remove surrounding spaces
* ``lower_trim``            lower case + trim
* ``strip_leading_zeros``   remove leading ``0`` characters
* ``digits_only``           keep only digits
* ``strip_prefix_<code>``   remove a literal prefix when present; ``<code>`` is
  the prefix with ``+`` written as ``plus`` (``strip_prefix_plus351`` -> ``+351``)
* ``strip_prefix:<literal>`` same, for prefixes with separators (``strip_prefix:MCH-``)
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

BASE_STRATEGIES = ("none", "trim", "lower_trim", "strip_leading_zeros", "digits_only")
_PREFIX_RE = re.compile(r"^strip_prefix_([A-Za-z0-9]+)$")  # legacy form: strip_prefix_plus351
_PREFIX_LITERAL_RE = re.compile(r"^strip_prefix:([+A-Za-z0-9_\-.:/#]{1,12})$")  # general form: strip_prefix:MCH-
_LEGACY_CHARS_RE = re.compile(r"^[+A-Za-z0-9]{1,8}$")
_PREFIX_CHARS_RE = re.compile(r"^[+A-Za-z0-9_\-.:/#]{1,12}$")


class UnknownStrategyError(ValueError):
    """Raised for an unknown normalisation name."""


def prefix_strategy(prefix: str) -> str:
    """Strategy name that strips ``prefix``.

    Letters, digits and "+" keep the compact legacy name (``+351`` ->
    ``strip_prefix_plus351``); prefixes with separators use the literal form
    (``MCH-`` -> ``strip_prefix:MCH-``).
    """
    if _LEGACY_CHARS_RE.match(prefix):
        return "strip_prefix_" + prefix.replace("+", "plus")
    if not _PREFIX_CHARS_RE.match(prefix):
        raise ValueError(f"Unsupported prefix {prefix!r}")
    return f"strip_prefix:{prefix}"


def _decode_prefix(code: str) -> str:
    return code.replace("plus", "+")


def _prefix_of(name: str) -> str | None:
    """Literal prefix of a strip_prefix strategy (legacy or literal form), else None."""
    match = _PREFIX_RE.match(name)
    if match:
        return _decode_prefix(match.group(1))
    match = _PREFIX_LITERAL_RE.match(name)
    return match.group(1) if match else None


def is_valid_strategy(name: str) -> bool:
    return name in BASE_STRATEGIES or _prefix_of(name) is not None


def strategy_sql(name: str, expr: str) -> str:
    """SQL expression applying strategy ``name`` to the SQL expression ``expr``."""
    text = f"CAST({expr} AS VARCHAR)"
    if name == "none":
        return text
    if name == "trim":
        return f"trim({text})"
    if name == "lower_trim":
        return f"lower(trim({text}))"
    if name == "strip_leading_zeros":
        return f"ltrim({text}, '0')"
    if name == "digits_only":
        return f"regexp_replace({text}, '[^0-9]', '', 'g')"
    prefix = _prefix_of(name)
    if prefix is not None:
        literal = prefix.replace("'", "''")
        return f"CASE WHEN starts_with({text}, '{literal}') THEN substr({text}, {len(prefix) + 1}) ELSE {text} END"
    raise UnknownStrategyError(
        f"Unknown normalization '{name}'. Use one of {list(BASE_STRATEGIES)}, strip_prefix_<code> "
        "or strip_prefix:<literal prefix>."
    )


def detect_prefixes(values: Iterable[str], targets: set[str], max_len: int = 6, min_share: float = 0.3) -> list[str]:
    """Find literal prefixes that, once removed from ``values``, produce ``targets``.

    For every value and every prefix length up to ``max_len``, checks whether
    the remaining suffix exists in ``targets``. Prefixes explaining at least
    ``min_share`` of the values are returned, most frequent first.
    """
    values = [v for v in values if v]
    if not values or not targets:
        return []
    counts: Counter[str] = Counter()
    for value in values:
        for k in range(1, min(max_len, len(value) - 1) + 1):
            if value[k:] in targets:
                counts[value[:k]] += 1
                break
    result = []
    for prefix, hits in counts.most_common(3):
        if hits / len(values) >= min_share and _PREFIX_CHARS_RE.match(prefix):
            result.append(prefix)
    return result
