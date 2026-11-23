"""模块: fastervla.core.types
功能: Observation / Chunk / TimedChunk / Prefix / RobotState / RobotCommand 形状与不变量。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.core.types import (
    Chunk,
    Observation,
    Prefix,
    RobotCommand,
    RobotState,
    TimedChunk,
)


# 用例: Observation 保存三路图像键与 24 维状态
# 参数: 无
def test_observation_holds_images_and_state() -> None:
    images = {
        "base_0_rgb": b"jpeg-base",
        "left_wrist_0_rgb": b"jpeg-left",
        "right_wrist_0_rgb": b"jpeg-right",
    }
    state = np.zeros(24, dtype=np.float32)
    obs = Observation(
        images=images,
        image_ts=1.0,
        state=state,
        state_ts=1.01,
        prompt="pick ball",
    )
    assert set(obs.images) == set(images)
    assert obs.state.shape == (24,)
    assert obs.prompt == "pick ball"


# 用例: Chunk.waypoints 必须是绝对关节量，形状 [H, D]
# 参数: bad 形状应在构造时拒绝
@pytest.mark.parametrize(
    "waypoints",
    [
        np.zeros(24, dtype=np.float32),
        np.zeros((30, 21), dtype=np.float32),
    ],
)
def test_chunk_rejects_bad_waypoint_shape(waypoints: np.ndarray) -> None:
    with pytest.raises((ValueError, TypeError)):
        Chunk(waypoints=waypoints, model_dt=1.0 / 30.0, obs_ts=0.0, meta={})


# 用例: 合法 Chunk / TimedChunk：路点不变，step_dt 长度 = H
# 参数: H=30, D=24
def test_timed_chunk_step_dt_matches_horizon() -> None:
    h, d = 30, 24
    waypoints = np.linspace(0.0, 1.0, h * d, dtype=np.float32).reshape(h, d)
    chunk = Chunk(waypoints=waypoints, model_dt=1.0 / 30.0, obs_ts=0.0, meta={})
    step_dt = np.full(h, 1.0 / 30.0, dtype=np.float64)
    timed = TimedChunk(chunk=chunk, step_dt=step_dt, speed=np.ones(h, dtype=np.float64))
    assert timed.chunk.waypoints.shape == (h, d)
    assert timed.step_dt.shape == (h,)
    np.testing.assert_array_equal(timed.chunk.waypoints, waypoints)


# 用例: TimedChunk 拒绝 step_dt 长度与 horizon 不一致
# 参数: 无
def test_timed_chunk_rejects_step_dt_mismatch() -> None:
    waypoints = np.zeros((10, 24), dtype=np.float32)
    chunk = Chunk(waypoints=waypoints, model_dt=0.1, obs_ts=0.0, meta={})
    with pytest.raises((ValueError, TypeError)):
        TimedChunk(
            chunk=chunk,
            step_dt=np.full(9, 0.1, dtype=np.float64),
            speed=np.ones(10, dtype=np.float64),
        )


# 用例: Prefix 携带绝对动作与 anchor_step
# 参数: 无
def test_prefix_stores_actions_and_anchor() -> None:
    actions = np.zeros((4, 24), dtype=np.float32)
    prefix = Prefix(actions=actions, anchor_step=7)
    assert prefix.actions.shape == (4, 24)
    assert prefix.anchor_step == 7


# 用例: RobotState / RobotCommand 时间戳与维数
# 参数: 无
def test_robot_state_and_command_shapes() -> None:
    state = RobotState(
        q=np.zeros(24, dtype=np.float64),
        qd=np.zeros(24, dtype=np.float64),
        timestamp=1.23,
        extras={},
    )
    cmd = RobotCommand(
        q_des=np.zeros(21, dtype=np.float64),
        qd_des=None,
        timestamp=1.24,
    )
    assert state.q.shape == (24,)
    assert cmd.q_des.shape == (21,)
    assert cmd.qd_des is None
