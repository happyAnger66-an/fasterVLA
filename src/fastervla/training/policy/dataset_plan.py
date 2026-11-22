"""Turn a TrainRecipe into a LeRobot/openpi window-loading plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastervla.training.policy.openpi_bridge import VelocitySkipConfig
from fastervla.training.policy.train_recipe import TrainRecipe
from fastervla.training.policy.train_recipe import build_lerobot_kwargs_from_recipe


@dataclass(frozen=True)
class DatasetPlan:
    """Resolved load plan: nominal timestamps + skip config."""

    repo_id: str
    delta_timestamps: dict[str, list[float]]
    skip_cfg: VelocitySkipConfig
    requires_real_dataset: bool


def build_dataset_plan(recipe: TrainRecipe) -> DatasetPlan:
    """Build window-sized ``delta_timestamps`` and flag FakeDataset limits."""
    kwargs = build_lerobot_kwargs_from_recipe(recipe)
    skip_cfg = recipe.skip_cfg
    needs_real = skip_cfg.window > skip_cfg.horizon
    return DatasetPlan(
        repo_id=recipe.repo_id,
        delta_timestamps=kwargs["delta_timestamps"],
        skip_cfg=skip_cfg,
        requires_real_dataset=needs_real,
    )


def validate_dataset_plan_for_openpi_fake(plan: DatasetPlan) -> None:
    """Reject plans that need a long action window but target openpi FakeDataset.

    openpi's ``FakeDataset`` only yields ``action_horizon`` steps, so ``window > H``
    cannot be satisfied when ``repo_id == "fake"``.
    """
    if plan.repo_id != "fake":
        return
    if plan.requires_real_dataset or plan.skip_cfg.window > plan.skip_cfg.horizon:
        raise ValueError(
            "openpi FakeDataset cannot provide window > horizon; "
            f"got window={plan.skip_cfg.window}, horizon={plan.skip_cfg.horizon}. "
            "Use a real LeRobot repo_id (or set window == horizon for debug)."
        )


def plan_to_debug_kwargs(plan: DatasetPlan) -> dict[str, Any]:
    """Extras useful for logging / future LeRobot construction."""
    return {
        "repo_id": plan.repo_id,
        "delta_timestamps": plan.delta_timestamps,
        "requires_real_dataset": plan.requires_real_dataset,
    }
