"""Generate the "unknown domain" dataset: industrial machine telemetry.

Same as ``jellybean generate-data --type telemetry``; the generator lives in
``jellybean.datagen.telemetry`` (see its docstring for what is planted):

    uv run python scripts/generate_unknown_domain.py --out /tmp/unknown/data --truth /tmp/unknown/truth
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jellybean.datagen.telemetry import TRUTH_FILE, generate_telemetry

# Kept under its original name: tests/test_unknown_domain.py loads this script.
generate = generate_telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="data folder (datasets are written here)")
    parser.add_argument("--truth", type=Path, required=True, help="ground-truth folder (outside --out)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--machines", type=int, default=1500)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--with-readme", action="store_true", help="write a README.md per dataset")
    parser.add_argument("--overwrite", action="store_true", help="replace existing datasets")
    args = parser.parse_args()
    result = generate_telemetry(
        args.out, args.truth, args.seed, args.machines, args.days, args.overwrite, args.with_readme, progress=print
    )
    print(f"telemetry rows: {result['rows']:,}; machines: {args.machines:,}; truth: {args.truth / TRUTH_FILE}")


if __name__ == "__main__":
    main()
