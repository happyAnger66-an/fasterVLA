"""模块: fastervla.training.policy.openpi_jax_backend
功能: 适配本机 openpi JAX；debug 配置跑 1–2 步；可选注入 velocity skip。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fastervla.training.policy.openpi_bridge import VelocitySkipConfig


def _skip_cfg(*, skip_frames: bool, horizon: int = 50, action_dim: int = 32) -> VelocitySkipConfig:
    # FakeDataset actions 长度 = model.action_horizon；Pi0Config 默认 50×32
    return VelocitySkipConfig(
        skip_frames=skip_frames,
        horizon=horizon,
        window=horizon,
        v_lim=np.full(action_dim, 1e6, dtype=np.float64),
        action_key="actions",
        dt=1.0 / horizon,
    )


# 用例: 未安装 openpi 时，入口给出明确 ImportError（而不是含糊失败）
# 参数: 无；若本机已装 openpi 则跳过
def test_run_openpi_debug_train_requires_openpi(tmp_path: Path) -> None:
    if importlib.util.find_spec("openpi") is not None:
        pytest.skip("openpi is importable; import-guard N/A")
    from fastervla.training.policy.openpi_jax_backend import run_openpi_debug_train

    with pytest.raises(ImportError, match="openpi"):
        run_openpi_debug_train(
            config_name="debug_pi05",
            num_steps=1,
            checkpoint_base_dir=tmp_path,
            skip_cfg=None,
        )


# 用例: debug_pi05 跑 1 step，loss 有限且参数相对初始有更新
# 参数: config_name=debug_pi05；skip 关；num_steps=1
@pytest.mark.openpi
def test_debug_pi05_one_step_finite_loss_and_update(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("openpi")
    from fastervla.training.policy.openpi_jax_backend import (
        OpenpiDebugRunResult,
        run_openpi_debug_train,
    )

    result = run_openpi_debug_train(
        config_name="debug_pi05",
        num_steps=1,
        checkpoint_base_dir=tmp_path,
        skip_cfg=_skip_cfg(skip_frames=False),
    )
    assert isinstance(result, OpenpiDebugRunResult)
    assert result.steps == 1
    assert result.config_name == "debug_pi05"
    assert len(result.losses) == 1
    assert np.isfinite(result.losses[0])
    assert result.params_updated is True


# 用例: debug_pi05 连续 2 step，每步 loss 有限，最终参数有更新
# 参数: num_steps=2；skip 关
@pytest.mark.openpi
def test_debug_pi05_two_steps_finite_losses(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("openpi")
    from fastervla.training.policy.openpi_jax_backend import run_openpi_debug_train

    result = run_openpi_debug_train(
        config_name="debug_pi05",
        num_steps=2,
        checkpoint_base_dir=tmp_path,
        skip_cfg=_skip_cfg(skip_frames=False),
    )
    assert result.steps == 2
    assert len(result.losses) == 2
    assert all(np.isfinite(x) for x in result.losses)
    assert result.params_updated is True


# 用例: skip_frames 开/关都能各跑通 1 step（Fake 数据 window==horizon）
# 参数: 两个子目录分别跑
@pytest.mark.openpi
def test_debug_pi05_skip_toggle_one_step_each(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("openpi")
    from fastervla.training.policy.openpi_jax_backend import run_openpi_debug_train

    for skip in (False, True):
        result = run_openpi_debug_train(
            config_name="debug_pi05",
            num_steps=1,
            checkpoint_base_dir=tmp_path / f"skip_{int(skip)}",
            skip_cfg=_skip_cfg(skip_frames=skip),
        )
        assert result.steps == 1
        assert np.isfinite(result.losses[0])
        assert result.params_updated is True
