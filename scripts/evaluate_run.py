"""Evaluate a run against the ground truth (spec section 10).

    uv run python scripts/evaluate_run.py <run_id> [--json]

Thin wrapper around ``jellybean evaluate``.
"""

from jellybean.cli import evaluate

if __name__ == "__main__":
    import typer

    from jellybean.settings import load_env_file

    load_env_file()
    typer.run(evaluate)
