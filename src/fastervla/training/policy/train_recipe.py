"""Load YAML train recipes: velocity-skip config + LeRobot window kwargs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fastervla.training.data.limits import load_joint_velocity_limits
from fastervla.training.policy.dataset_assembly import build_lerobot_dataset_kwargs
from fastervla.training.policy.openpi_bridge import VelocitySkipConfig


@dataclass(frozen=True)
class TrainRecipe:
    """Resolved training recipe ready for CLI / openpi debug runs."""

    openpi_config: str
    skip_cfg: VelocitySkipConfig
    fps: float
    repo_id: str
    action_keys: tuple[str, ...]
    num_steps: int
    limits_path: Path


def load_train_recipe(path: str | Path) -> TrainRecipe:
    """Load a train recipe YAML and build ``VelocitySkipConfig`` from limits."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        data: dict[str, Any] = yaml.safe_load(fp) or {}

    required = (
        "openpi_config",
        "skip_frames",
        "horizon",
        "window",
        "fps",
        "limits_path",
        "action_keys",
        "repo_id",
        "num_steps",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing recipe keys: {missing}")

    limits_path = Path(data["limits_path"])
    if not limits_path.is_absolute():
        limits_path = (path.parent / limits_path).resolve()

    limits = load_joint_velocity_limits(limits_path)
    horizon = int(data["horizon"])
    window = int(data["window"])
    skip_frames = bool(data["skip_frames"])

    skip_cfg = VelocitySkipConfig.from_limits(
        limits,
        horizon=horizon,
        window=window,
        skip_frames=skip_frames,
        action_key=str(list(data["action_keys"])[0]),
        dt=1.0 / float(horizon),
        fps=float(data["fps"]),
    )

    return TrainRecipe(
        openpi_config=str(data["openpi_config"]),
        skip_cfg=skip_cfg,
        fps=float(data["fps"]),
        repo_id=str(data["repo_id"]),
        action_keys=tuple(str(k) for k in data["action_keys"]),
        num_steps=int(data["num_steps"]),
        limits_path=limits_path,
    )


def build_lerobot_kwargs_from_recipe(recipe: TrainRecipe) -> dict[str, Any]:
    """Build LeRobot constructor kwargs from a resolved recipe."""
    return build_lerobot_dataset_kwargs(
        repo_id=recipe.repo_id,
        fps=recipe.fps,
        action_keys=recipe.action_keys,
        skip_cfg=recipe.skip_cfg,
    )


def format_recipe_summary(recipe: TrainRecipe) -> str:
    """Human-readable one-liner summary for CLI dry-run."""
    cfg = recipe.skip_cfg
    return (
        f"openpi_config={recipe.openpi_config} "
        f"repo_id={recipe.repo_id} "
        f"skip_frames={cfg.skip_frames} "
        f"horizon={cfg.horizon} "
        f"window={cfg.window} "
        f"fps={recipe.fps} "
        f"num_steps={recipe.num_steps}"
    )
