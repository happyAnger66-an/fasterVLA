"""Assemble training datasets with velocity-constrained action skip."""

from __future__ import annotations

from typing import Any
from typing import Mapping
from typing import Sequence

from fastervla.training.policy.openpi_bridge import VelocitySkipActions
from fastervla.training.policy.openpi_bridge import VelocitySkipConfig
from fastervla.training.policy.openpi_bridge import make_action_delta_timestamps


class VelocitySkipDataset:
    """Random-access wrapper: apply ``VelocitySkipActions`` on each sample."""

    def __init__(self, base: Any, config: VelocitySkipConfig) -> None:
        self._base = base
        self._transform = VelocitySkipActions(config)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._base[index]
        if not isinstance(sample, Mapping):
            raise TypeError(
                f"base dataset sample must be a mapping, got {type(sample).__name__}"
            )
        return self._transform(sample)


def wrap_dataset_with_velocity_skip(
    base: Any,
    skip_cfg: VelocitySkipConfig,
) -> VelocitySkipDataset:
    """Wrap a window-yielding dataset so each sample's actions become ``[H, D]``."""
    return VelocitySkipDataset(base, skip_cfg)


def build_lerobot_dataset_kwargs(
    *,
    repo_id: str,
    fps: float,
    action_keys: Sequence[str],
    skip_cfg: VelocitySkipConfig,
) -> dict[str, Any]:
    """Build kwargs for LeRobot dataset construction with window-sized timestamps.

    ``delta_timestamps`` length follows ``skip_cfg.window`` (equals horizon when
    skip is off and window == horizon). Offsets are nominal ``k / fps``.
    """
    return {
        "repo_id": repo_id,
        "delta_timestamps": make_action_delta_timestamps(
            action_keys,
            fps=fps,
            window=skip_cfg.window,
        ),
    }
