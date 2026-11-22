"""模块: fastervla.training 端到端训练流程（无 openpi）
功能: recipe → plan → create_dataset → [H,D] sample；示例配置 dry-run。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fastervla.training.cli import main as train_cli_main
from fastervla.training.policy.create_dataset import create_dataset_from_plan
from fastervla.training.policy.dataset_plan import build_dataset_plan
from fastervla.training.policy.train_recipe import load_train_recipe


# 用例: 示例 debug recipe 经 CLI dry-run 输出含 openpi_config 与 fake 计划
# 参数: configs/training/pi05_velocity_skip_debug.yaml
def test_example_debug_recipe_cli_dry_run(capsys) -> None:
    root = Path(__file__).resolve().parents[2]
    recipe = root / "configs" / "training" / "pi05_velocity_skip_debug.yaml"
    rc = train_cli_main(
        [
            "train",
            "--recipe",
            str(recipe),
            "--checkpoint-dir",
            "/tmp/fastervla_ckpt_dry",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "debug_pi05" in out
    assert "requires_real_dataset=False" in out
    assert "window=50" in out


# 用例: 非 fake plan + 注入工厂走完拉窗跳帧，得到 [H,D]（完整数据侧流程）
# 参数: window=16；horizon=4
def test_full_data_side_flow_with_injected_factory(tmp_path: Path) -> None:
    horizon, window = 4, 16
    dt = 1.0 / horizon
    limits = tmp_path / "limits.yaml"
    limits.write_text(
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
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "openpi_config": "debug_pi05",
                "skip_frames": True,
                "horizon": horizon,
                "window": window,
                "fps": 30.0,
                "limits_path": str(limits),
                "action_keys": ["actions"],
                "repo_id": "local/demo",
                "num_steps": 1,
            }
        ),
        encoding="utf-8",
    )
    recipe = load_train_recipe(recipe_path)
    plan = build_dataset_plan(recipe)
    assert plan.requires_real_dataset is True

    traj = np.arange(window + 2, dtype=np.float64)[:, None] * np.ones((1, 3))

    def factory(*, repo_id: str, delta_timestamps: dict[str, list[float]]) -> Any:
        class _DS:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> dict[str, Any]:
                return {"actions": traj[:window].copy(), "state": traj[0].copy()}

        assert len(delta_timestamps["actions"]) == window
        return _DS()

    sample = create_dataset_from_plan(plan, dataset_factory=factory)[0]
    assert sample["actions"].shape == (horizon, 3)
