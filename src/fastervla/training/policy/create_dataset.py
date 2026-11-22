"""Build a velocity-skip-wrapped dataset from a DatasetPlan."""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Protocol

from fastervla.training.policy.dataset_assembly import wrap_dataset_with_velocity_skip
from fastervla.training.policy.dataset_plan import DatasetPlan


class DatasetFactory(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        delta_timestamps: dict[str, list[float]],
    ) -> Any:
        """Construct a random-access dataset that yields window-length actions."""


def _default_lerobot_factory(
    *,
    repo_id: str,
    delta_timestamps: dict[str, list[float]],
) -> Any:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "lerobot is required to construct a dataset from plan when "
                "no dataset_factory is injected; pip install lerobot or pass "
                "dataset_factory=..."
            ) from exc
    return LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)


def create_dataset_from_plan(
    plan: DatasetPlan,
    *,
    dataset_factory: Callable[..., Any] | None = None,
) -> Any:
    """Create a window-yielding dataset and wrap it with velocity skip.

    ``dataset_factory`` is injectable for unit tests. When omitted, constructs a
    LeRobot dataset with ``plan.delta_timestamps``.
    """
    factory = dataset_factory or _default_lerobot_factory
    base = factory(
        repo_id=plan.repo_id,
        delta_timestamps=plan.delta_timestamps,
    )
    return wrap_dataset_with_velocity_skip(base, plan.skip_cfg)
