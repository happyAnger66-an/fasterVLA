"""模块: fastervla.training.policy.train_recipe
功能: 从 YAML 组装训练配方（skip/window/fps/limits）并生成 LeRobot kwargs；供 CLI 调用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from fastervla.training.policy.train_recipe import (
    TrainRecipe,
    build_lerobot_kwargs_from_recipe,
    load_train_recipe,
)


def _write_limits(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "joint_names": ["j0", "j1", "j2"],
                "v_max": [1.0, 2.0, 0.5],
                "velocity_scale": 0.9,
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
                "repo_id": "local/demo",
                "num_steps": 2,
            }
        ),
        encoding="utf-8",
    )
    return path


# 用例: YAML 加载后 skip 开、window>horizon，v_lim 使用 effective_v_max
# 参数: velocity_scale=0.9；window=40；horizon=10
def test_load_train_recipe_skip_window(tmp_path: Path) -> None:
    limits = _write_limits(tmp_path / "limits.yaml")
    recipe_path = _write_recipe(tmp_path / "recipe.yaml", limits_path=limits)

    recipe = load_train_recipe(recipe_path)
    assert isinstance(recipe, TrainRecipe)
    assert recipe.openpi_config == "debug_pi05"
    assert recipe.num_steps == 2
    assert recipe.fps == pytest.approx(30.0)
    assert recipe.repo_id == "local/demo"
    assert recipe.action_keys == ("actions",)
    assert recipe.skip_cfg.skip_frames is True
    assert recipe.skip_cfg.horizon == 10
    assert recipe.skip_cfg.window == 40
    np.testing.assert_allclose(
        recipe.skip_cfg.v_lim,
        np.asarray([0.9, 1.8, 0.45], dtype=np.float64),
    )


# 用例: skip 关闭时允许 window==horizon，且生成的时间戳长度等于 horizon
# 参数: skip_frames=false；window=horizon=8；fps=20
def test_load_train_recipe_dense_and_lerobot_kwargs(tmp_path: Path) -> None:
    limits = _write_limits(tmp_path / "limits.yaml")
    recipe_path = _write_recipe(
        tmp_path / "recipe.yaml",
        limits_path=limits,
        skip_frames=False,
        horizon=8,
        window=8,
        fps=20.0,
    )
    recipe = load_train_recipe(recipe_path)
    assert recipe.skip_cfg.skip_frames is False
    assert recipe.skip_cfg.window == recipe.skip_cfg.horizon

    kwargs = build_lerobot_kwargs_from_recipe(recipe)
    assert kwargs["repo_id"] == "local/demo"
    assert len(kwargs["delta_timestamps"]["actions"]) == 8
    assert kwargs["delta_timestamps"]["actions"][1] == pytest.approx(1.0 / 20.0)


# 用例: skip 开时 LeRobot kwargs 时间戳长度等于 window，步长 1/fps
# 参数: fps=30；window=40
def test_build_lerobot_kwargs_uses_window_when_skip(tmp_path: Path) -> None:
    limits = _write_limits(tmp_path / "limits.yaml")
    recipe = load_train_recipe(
        _write_recipe(tmp_path / "recipe.yaml", limits_path=limits, window=40, fps=30.0)
    )
    kwargs = build_lerobot_kwargs_from_recipe(recipe)
    stamps = kwargs["delta_timestamps"]["actions"]
    assert len(stamps) == 40
    assert stamps[1] == pytest.approx(1.0 / 30.0)
    assert stamps[-1] == pytest.approx(39.0 / 30.0)


# 用例: skip 开但 window < horizon 应在加载时报错
# 参数: window=5；horizon=10
def test_load_train_recipe_rejects_short_window(tmp_path: Path) -> None:
    limits = _write_limits(tmp_path / "limits.yaml")
    recipe_path = _write_recipe(
        tmp_path / "recipe.yaml",
        limits_path=limits,
        horizon=10,
        window=5,
    )
    with pytest.raises(ValueError, match="window"):
        load_train_recipe(recipe_path)


# 用例: CLI dry-run 打印配方摘要且不要求 openpi
# 参数: --dry-run
def test_cli_dry_run_prints_recipe(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from fastervla.training.cli import main as train_cli_main

    limits = _write_limits(tmp_path / "limits.yaml")
    recipe_path = _write_recipe(tmp_path / "recipe.yaml", limits_path=limits)

    rc = train_cli_main(
        [
            "train",
            "--recipe",
            str(recipe_path),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "debug_pi05" in out
    assert "skip_frames=True" in out or "skip_frames=true" in out
    assert "window=40" in out
