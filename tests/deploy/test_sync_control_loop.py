"""模块: fastervla.deploy.client.loops（F0 同步 1×）
功能: MockBackend + MockRobotPlatform 同步控制环；κ=1.0、均匀时间轴；无越界。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.deploy.backends.mock import MockBackend
from fastervla.deploy.client.io.mock_sim import MockRobotPlatform
from fastervla.deploy.client.loops import SyncControlLoop, SyncLoopConfig


# 用例: 同步环跑满 N 拍：每拍有 reference/command；命令维 21；状态维 24
# 参数: 无
def test_sync_loop_runs_without_crash() -> None:
    plat = MockRobotPlatform(
        state_dim=24,
        command_dim=21,
        camera_names=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        plant_tau_s=0.057,
        plant_delay_s=0.036,
        seed=0,
    )
    backend = MockBackend(horizon=30, action_dim=24, seed=0, mode="sine")
    cfg = SyncLoopConfig(
        control_hz=100.0,
        policy_fps=30.0,
        kappa=1.0,
        kappa_max=1.5,
        kappa_min=1.0,
        horizon=30,
        max_steps=200,
        action_dim=24,
        command_dim=21,
    )
    loop = SyncControlLoop(platform=plat, backend=backend, config=cfg)
    summary = loop.run()
    assert summary.steps == 200
    assert summary.commands_sent == 200
    assert summary.max_abs_command < np.inf
    plat.close()


# 用例: κ=1.0 时每模型步墙钟约 1/policy_fps；整段时长 ≈ horizon / policy_fps
# 参数: 无（允许计时误差）
def test_sync_loop_one_x_duration_matches_policy_fps() -> None:
    plat = MockRobotPlatform(
        state_dim=24,
        command_dim=21,
        camera_names=("base_0_rgb",),
        plant_tau_s=0.01,
        plant_delay_s=0.0,
        seed=1,
    )
    backend = MockBackend(horizon=30, action_dim=24, seed=1, mode="sine")
    cfg = SyncLoopConfig(
        control_hz=100.0,
        policy_fps=30.0,
        kappa=1.0,
        horizon=30,
        max_steps=None,  # 跑完一个 chunk
        action_dim=24,
        command_dim=21,
        stop_after_one_chunk=True,
    )
    loop = SyncControlLoop(platform=plat, backend=backend, config=cfg)
    summary = loop.run()
    expected_s = 30.0 / 30.0  # 1.0 s
    assert summary.elapsed_s == pytest.approx(expected_s, rel=0.25, abs=0.15)
    plat.close()


# 用例: 命令经 layout 后夹在 embodiment 限位内（mock 默认宽限位）
# 参数: 无
def test_sync_loop_commands_respect_default_limits() -> None:
    plat = MockRobotPlatform(
        state_dim=24,
        command_dim=21,
        camera_names=("base_0_rgb",),
        plant_tau_s=0.05,
        plant_delay_s=0.0,
        seed=2,
        q_min=np.full(21, -2.0),
        q_max=np.full(21, 2.0),
    )
    backend = MockBackend(horizon=30, action_dim=24, seed=2, mode="sine")
    cfg = SyncLoopConfig(
        control_hz=100.0,
        policy_fps=30.0,
        kappa=1.0,
        horizon=30,
        max_steps=100,
        action_dim=24,
        command_dim=21,
    )
    summary = SyncControlLoop(platform=plat, backend=backend, config=cfg).run()
    assert summary.max_abs_command <= 2.0 + 1e-6
    assert summary.limit_violations == 0
    plat.close()


# 用例: 未知 platform / 缺 backend 时构造失败
# 参数: 无
def test_sync_loop_requires_platform_and_backend() -> None:
    with pytest.raises((TypeError, ValueError)):
        SyncControlLoop(platform=None, backend=None, config=SyncLoopConfig())


# 用例: kappa 必须落在 [kappa_min, kappa_max]（YAML/CLI 可配上限）
# 参数: 无
def test_sync_loop_config_rejects_kappa_above_max() -> None:
    with pytest.raises((ValueError, TypeError)):
        SyncLoopConfig(
            control_hz=100.0,
            policy_fps=30.0,
            kappa=1.6,
            kappa_max=1.5,
            kappa_min=1.0,
            horizon=30,
            action_dim=24,
            command_dim=21,
        )
