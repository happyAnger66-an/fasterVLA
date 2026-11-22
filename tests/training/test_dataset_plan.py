"""模块: fastervla.training.policy.dataset_plan
功能: TrainRecipe → LeRobot/openpi 拉窗计划；校验 FakeDataset 无法提供 window>H。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fastervla.training.policy.dataset_plan import (
    DatasetPlan,
    build_dataset_plan,
    validate_dataset_plan_for_openpi_fake,
)
from fastervla.training.policy.train_recipe import load_train_recipe


def _write_limits(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "joint_names": ["j0", "j1", "j2"],
                "v_max": [1.0, 2.0, 0.5],
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
    repo_id: str = "local/demo",
    skip_frames: bool = True,
    horizon: int = 10,
    window: int = 40,
    fps: float = 30.0,
) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "openpi_config": "debug_pi05",
                "skip_frames": skip_frames,
                "horizon": horizon,
                "window": window,
                "fps": fps,
                "limits_path": str(limits_path),
                "action_keys": ["actions"],
                "repo_id": repo_id,
                "num_steps": 1,
            }
        ),
        encoding="utf-8",
    )
    return path


# 用例: skip 开、window>H 时计划里时间戳长度为 window，并标记需要真实数据集
# 参数: window=40；horizon=10；fps=30；repo_id=local/demo
def test_build_dataset_plan_window_requires_real(tmp_path: Path) -> None:
    limits = _write_limits(tmp_path / "limits.yaml")
    recipe = load_train_recipe(
        _write_recipe(tmp_path / "recipe.yaml", limits_path=limits)
    )
    plan = build_dataset_plan(recipe)
    assert isinstance(plan, DatasetPlan)
    assert plan.repo_id == "local/demo"
    assert len(plan.delta_timestamps["actions"]) == 40
    assert plan.delta_timestamps["actions"][1] == pytest.approx(1.0 / 30.0)
    assert plan.requires_real_dataset is True
    assert plan.skip_cfg.window == 40


# 用例: window==horizon 时不要求真实数据集（可用 openpi FakeDataset）
# 参数: window=horizon=10；repo_id=fake
def test_build_dataset_plan_dense_ok_for_fake(tmp_path: Path) -> None:
    limits = _write_limits(tmp_path / "limits.yaml")
    recipe = load_train_recipe(
        _write_recipe(
            tmp_path / "recipe.yaml",
            limits_path=limits,
            repo_id="fake",
            skip_frames=False,
            horizon=10,
            window=10,
        )
    )
    plan = build_dataset_plan(recipe)
    assert plan.requires_real_dataset is False
    assert len(plan.delta_timestamps["actions"]) == 10
    validate_dataset_plan_for_openpi_fake(plan)  # 不应抛错


# 用例: repo_id=fake 且 window>H 时校验失败，提示 FakeDataset 拉不出长窗
# 参数: repo_id=fake；window=40；horizon=10
def test_validate_fake_rejects_long_window(tmp_path: Path) -> None:
    limits = _write_limits(tmp_path / "limits.yaml")
    recipe = load_train_recipe(
        _write_recipe(
            tmp_path / "recipe.yaml",
            limits_path=limits,
            repo_id="fake",
            window=40,
            horizon=10,
        )
    )
    plan = build_dataset_plan(recipe)
    assert plan.requires_real_dataset is True
    with pytest.raises(ValueError, match="FakeDataset|window"):
        validate_dataset_plan_for_openpi_fake(plan)


# 用例: run_train_from_recipe 在 dry 校验路径上，fake+长窗直接失败且不调 openpi
# 参数: repo_id=fake；window>horizon
def test_run_train_from_recipe_rejects_fake_long_window(tmp_path: Path) -> None:
    from fastervla.training.policy.openpi_jax_backend import run_train_from_recipe

    limits = _write_limits(tmp_path / "limits.yaml")
    recipe = load_train_recipe(
        _write_recipe(
            tmp_path / "recipe.yaml",
            limits_path=limits,
            repo_id="fake",
            window=40,
            horizon=10,
        )
    )
    with pytest.raises(ValueError, match="FakeDataset|window"):
        run_train_from_recipe(recipe, checkpoint_base_dir=tmp_path / "ckpt")
