# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Generate the dummy datasets (spec section 9) or the unknown-domain dataset.

Thin wrapper around ``jellybean generate-data`` (same options):

    uv run python scripts/generate_dummy_data.py --out ./data --truth ./ground_truth \
        --seed 42 --customers 20000 --days 30 \
        --datasets traffic,cust_segmentation,orders,topup \
        --topup-days 90 [--opaque-codes] [--with-readme] [--scale 1.0]
    uv run python scripts/generate_dummy_data.py --type telemetry --out /tmp/unknown/data \
        --truth /tmp/unknown/truth [--machines 1500] [--days 14]
"""

from jellybean.cli import generate_data

if __name__ == "__main__":
    import typer

    typer.run(generate_data)
