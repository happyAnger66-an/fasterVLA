"""模块: fastervla.training.policy.create_dataset
功能: 按 DatasetPlan 构造带 window 时间戳的数据集，再 wrap 速度跳帧。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from fastervla.training.policy.dataset_plan import build_dataset_plan
from fastervla.training.policy.train_recipe import load_train_recipe
from fastervla.training.resample.velocity_skip import build_skip_chunk


def _write_limits(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "joint_names": ["j0", "j1", "j2"],
                "v_max": [1.0, 1.0, 1.0],
                "velocity_scale": 1.0,
                "ignore_velocity_dims": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_recipe(
    path: Path,
    *,
    limits_path: Path,
    repo_id: str,
    horizon: int,
    window: int,
    skip_frames: bool = True,
) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "openpi_config": "debug_pi05",
                "skip_frames": skip_frames,
                "horizon": horizon,
                "window": window,
                "fps": 30.0,
                "limits_path": str(limits_path),
                "action_keys": ["actions"],
                "repo_id": repo_id,
                "num_steps": 1,
            }
        ),
        encoding="utf-8",
    )
    return path


class _FakeLeRobotWindowDataset:
    """Stand-in for LeRobotDataset: yields action windows of length ``window``."""

    def __init__(self, traj: np.ndarray, window: int, delta_timestamps: dict[str, list[float]]) -> None:
        self._traj = np.asarray(traj, dtype=np.float64)
        self._window = window
        self.delta_timestamps = delta_timestamps

    def __len__(self) -> int:
        return max(0, len(self._traj) - self._window + 1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        i = int(index)
        return {
            "actions": self._traj[i : i + self._window].copy(),
            "state": self._traj[i].copy(),
        }


# 用例: 注入假 LeRobot 工厂时，按 plan 的 delta_timestamps 拉窗并跳帧得到 [H,D]
# 参数: window=20；horizon=5；每跳跨 2 帧
def test_create_dataset_from_plan_wraps_skip_with_injected_factory(
    tmp_path: Path,
) -> None:
    from fastervla.training.policy.create_dataset import create_dataset_from_plan

    horizon, window = 5, 20
    dt = 1.0 / horizon
    # v_lim = 2/dt，匀速 step=1 → 每跳最远跨 2 帧
    limits_path = tmp_path / "limits.yaml"
    limits_path.write_text(
        yaml.safe_dump(
            {
                "joint_names": ["j0", "j1", "j2"],
                "v_max": [float(2.0 / dt)] * 3,
                "velocity_scale": 1.0,
                "ignore_velocity_dims": [],
            }
        ),
        encoding="utf-8",
    )
    recipe = load_train_recipe(
        _write_recipe(
            tmp_path / "recipe.yaml",
            limits_path=limits_path,
            repo_id="local/demo",
            horizon=horizon,
            window=window,
            skip_frames=True,
        )
    )
    plan = build_dataset_plan(recipe)
    assert plan.requires_real_dataset is True

    traj = np.arange(window + 5, dtype=np.float64)[:, None] * np.ones((1, 3))

    def factory(*, repo_id: str, delta_timestamps: dict[str, list[float]]) -> Any:
        assert repo_id == "local/demo"
        assert len(delta_timestamps["actions"]) == window
        return _FakeLeRobotWindowDataset(traj, window=window, delta_timestamps=delta_timestamps)

    ds = create_dataset_from_plan(plan, dataset_factory=factory)
    sample = ds[0]
    assert sample["actions"].shape == (horizon, 3)
    expected = build_skip_chunk(
        traj[:window],
        h0=0,
        horizon=horizon,
        v_lim=plan.skip_cfg.v_lim,
        dt=dt,
    )
    np.testing.assert_allclose(sample["actions"], expected)


# 用例: 未注入工厂且 repo 非 fake 时，缺少 lerobot 应给出明确 ImportError
# 参数: repo_id=local/demo
def test_create_dataset_from_plan_requires_lerobot_or_factory(tmp_path: Path) -> None:
    from fastervla.training.policy.create_dataset import create_dataset_from_plan

    limits = _write_limits(tmp_path / "limits.yaml")
    recipe = load_train_recipe(
        _write_recipe(
            tmp_path / "recipe.yaml",
            limits_path=limits,
            repo_id="local/demo",
            horizon=10,
            window=40,
        )
    )
    plan = build_dataset_plan(recipe)

    # 若环境已装 lerobot，此用例改为跳过（真构造会去拉数据）
    import importlib.util

    if importlib.util.find_spec("lerobot") is not None:
        pytest.skip("lerobot installed; default factory path not asserted here")

    with pytest.raises(ImportError, match="lerobot"):
        create_dataset_from_plan(plan)


# 用例: 仓库内示例 recipe / limits 可被 load_train_recipe 解析
# 参数: configs/training/*.yaml
def test_example_training_configs_load() -> None:
    root = Path(__file__).resolve().parents[2]
    recipe_path = root / "configs" / "training" / "pi05_velocity_skip_debug.yaml"
    limits_path = root / "configs" / "training" / "joint_velocity_limits.yaml"
    assert limits_path.is_file()
    assert recipe_path.is_file()
    recipe = load_train_recipe(recipe_path)
    assert recipe.openpi_config == "debug_pi05"
    assert recipe.skip_cfg.window >= recipe.skip_cfg.horizon
    plan = build_dataset_plan(recipe)
    assert len(plan.delta_timestamps["actions"]) == recipe.skip_cfg.window
