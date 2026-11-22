"""CLI entry for fasterVLA training recipes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fastervla.training.policy.train_recipe import format_recipe_summary
from fastervla.training.policy.train_recipe import load_train_recipe


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fastervla-train")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run training from a recipe YAML")
    train.add_argument("--recipe", type=Path, required=True, help="Path to train recipe YAML")
    train.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Checkpoint base directory for openpi runs",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print recipe summary without calling openpi",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "train":
        recipe = load_train_recipe(args.recipe)
        if args.dry_run:
            print(format_recipe_summary(recipe))
            print(f"checkpoint_dir={args.checkpoint_dir}")
            return 0

        from fastervla.training.policy.openpi_jax_backend import run_openpi_debug_train

        result = run_openpi_debug_train(
            config_name=recipe.openpi_config,
            num_steps=recipe.num_steps,
            checkpoint_base_dir=args.checkpoint_dir,
            skip_cfg=recipe.skip_cfg,
        )
        print(
            f"done steps={result.steps} losses={result.losses} "
            f"params_updated={result.params_updated}"
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
