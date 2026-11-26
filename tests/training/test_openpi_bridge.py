"""模块: fastervla.training.policy.openpi_bridge
功能: openpi 数据加载钩子——拉长 action 窗口后按速度约束跳帧，产出与 pi05 同形的 [H, D] actions。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from fastervla.training.data.limits import effective_v_max
from fastervla.training.data.limits import load_joint_velocity_limits
from fastervla.training.policy.openpi_bridge import (
    VelocitySkipActions,
    VelocitySkipConfig,
    make_action_delta_timestamps,
)


# 用例: skip 关闭时 delta_timestamps 窗口等于 horizon
# 参数: action_keys 是序列键；fps 用于换算秒；horizon 是模型 action_horizon
def test_make_action_delta_timestamps_dense_window() -> None:
    stamps = make_action_delta_timestamps(
        action_keys=("actions",),
        fps=30.0,
        window=10,
    )
    assert list(stamps.keys()) == ["actions"]
    assert stamps["actions"] == [t / 30.0 for t in range(10)]


# 用例: skip 开启时允许 window > horizon，时间戳长度为 window
# 参数: window=20, horizon 只影响配置对象，不直接进本函数
def test_make_action_delta_timestamps_long_window_for_skip() -> None:
    stamps = make_action_delta_timestamps(
        action_keys=("action", "actions"),
        fps=20.0,
        window=40,
    )
    assert stamps["action"] == [t / 20.0 for t in range(40)]
    assert stamps["actions"] == stamps["action"]


# 用例: VelocitySkipConfig 校验 window >= horizon
# 参数: 无
def test_velocity_skip_config_rejects_window_shorter_than_horizon() -> None:
    with pytest.raises(ValueError):
        VelocitySkipConfig(
            skip_frames=True,
            horizon=10,
            window=5,
            v_lim=np.ones(3),
        )


# 用例: skip_frames=False 时原样取窗口前 H 帧（连续 horizon）
# 参数: 恒速轨迹窗口；H=5
def test_velocity_skip_actions_dense_takes_prefix(constant_speed_traj: np.ndarray) -> None:
    horizon = 5
    window = constant_speed_traj[:20]
    cfg = VelocitySkipConfig(
        skip_frames=False,
        horizon=horizon,
        window=20,
        v_lim=np.ones(3),
        action_key="actions",
    )
    out = VelocitySkipActions(cfg)({"actions": window, "state": np.zeros(3)})
    np.testing.assert_allclose(out["actions"], window[:horizon])
    assert out["actions"].shape == (horizon, 3)
    # 其它字段保留
    np.testing.assert_allclose(out["state"], np.zeros(3))


# 用例: skip_frames=True 时输出仍是 [H, D]，且等于 velocity_skip 选出的行
# 参数: 紧约束使每跳跨 1 帧 → 结果为 window[1:H+1]
def test_velocity_skip_actions_skip_matches_farthest_jumps(
    constant_speed_traj: np.ndarray,
) -> None:
    from fastervla.training.resample.velocity_skip import build_skip_chunk

    horizon = 5
    dt = 1.0 / horizon
    window = constant_speed_traj[:30]
    one_step_vel = np.asarray([0.1, 0.05, 0.02], dtype=np.float64) / dt
    cfg = VelocitySkipConfig(
        skip_frames=True,
        horizon=horizon,
        window=30,
        v_lim=one_step_vel,
        action_key="actions",
        dt=dt,
    )
    out = VelocitySkipActions(cfg)({"actions": window})
    expected = build_skip_chunk(window, h0=0, horizon=horizon, v_lim=one_step_vel, dt=dt)
    np.testing.assert_allclose(out["actions"], expected)
    assert out["actions"].shape == (horizon, 3)


# 用例: 输入 actions 长度必须 >= window（配置约定）；过短应报错
# 参数: 短数组
def test_velocity_skip_actions_rejects_short_sequence() -> None:
    cfg = VelocitySkipConfig(
        skip_frames=True,
        horizon=4,
        window=10,
        v_lim=np.ones(2),
        action_key="actions",
    )
    with pytest.raises(ValueError):
        VelocitySkipActions(cfg)({"actions": np.zeros((3, 2))})


# 用例: 从 YAML limits 构造 config 时使用 effective_v_max
# 参数: tmp_limits_yaml；scale=0.9
def test_velocity_skip_config_from_limits_yaml(tmp_limits_yaml: Path) -> None:
    limits = load_joint_velocity_limits(tmp_limits_yaml)
    limits.velocity_scale = 0.9
    cfg = VelocitySkipConfig.from_limits(
        limits,
        horizon=8,
        window=32,
        skip_frames=True,
    )
    np.testing.assert_allclose(cfg.v_lim, effective_v_max(limits))
    assert cfg.horizon == 8
    assert cfg.window == 32
    assert cfg.skip_frames is True


# 用例: from_limits 在 skip 关闭时允许 window == horizon
# 参数: tmp_limits_yaml
def test_velocity_skip_config_from_limits_dense(tmp_limits_yaml: Path) -> None:
    limits = load_joint_velocity_limits(tmp_limits_yaml)
    cfg = VelocitySkipConfig.from_limits(
        limits,
        horizon=10,
        window=10,
        skip_frames=False,
    )
    assert cfg.skip_frames is False
    assert cfg.window == cfg.horizon


# 用例: from_limits 带上 fps 后，1.5s 窗口换成帧数
# 参数: tmp_limits_yaml；fps=30 → window_frames=45
def test_velocity_skip_config_gripper_window_frames(tmp_limits_yaml: Path) -> None:
    limits = load_joint_velocity_limits(tmp_limits_yaml)
    limits.gripper_no_skip.window_s = 1.5
    cfg = VelocitySkipConfig.from_limits(
        limits,
        horizon=8,
        window=32,
        skip_frames=True,
        fps=30.0,
    )
    assert cfg.gripper_no_skip_window_s == 1.5
    assert cfg.gripper_no_skip_window_frames == 45
    assert cfg.gripper_no_skip_enabled is True
