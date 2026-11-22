"""模块: fastervla.training.resample.velocity_skip
功能: 按 Δt=1/H 与关节速度上限，连续 H 次单步最远合法跳，组出未来 action chunk 下标。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.training.resample.velocity_skip import (
    build_skip_chunk,
    build_skip_chunk_indices,
    farthest_legal_index,
    is_legal_jump,
)


# 用例: Δt 缺省为 1/H
# 参数: H 是 horizon；用恒速轨迹验证默认 dt 与显式 1/H 得到同一组下标
@pytest.mark.parametrize("horizon", [5, 10, 30])
def test_default_dt_is_one_over_horizon(
    constant_speed_traj: np.ndarray, horizon: int
) -> None:
    v_lim = np.asarray([3.0, 3.0, 3.0], dtype=np.float64)
    h0 = 0
    with_default = build_skip_chunk_indices(
        constant_speed_traj, h0=h0, horizon=horizon, v_lim=v_lim
    )
    with_explicit = build_skip_chunk_indices(
        constant_speed_traj,
        h0=h0,
        horizon=horizon,
        v_lim=v_lim,
        dt=1.0 / horizon,
    )
    np.testing.assert_array_equal(with_default, with_explicit)


# 用例: 单关节上，最远合法跳刚好卡在 v_lim * dt 允许的最大跨度
# 参数: 无；q[i]=i，dt=0.1，v_lim=20 → |Δq|/dt=k/0.1≤20 → 最大跨度 k=2
def test_farthest_legal_index_respects_single_joint_limit() -> None:
    q = np.arange(20, dtype=np.float64)[:, None]
    dt = 0.1
    v_lim = np.asarray([20.0], dtype=np.float64)
    assert farthest_legal_index(q, a=0, v_lim=v_lim, dt=dt) == 2
    assert is_legal_jump(q[0], q[2], dt=dt, v_lim=v_lim)
    assert not is_legal_jump(q[0], q[3], dt=dt, v_lim=v_lim)


# 用例: 多关节必须全部满足；瓶颈关节决定最远跳
# 参数: 无；j0/j1 很松，j2 只允许跨 2 帧
def test_all_joints_must_satisfy_for_jump_to_be_legal() -> None:
    t = 30
    delta = np.asarray([0.10, 0.05, 0.02], dtype=np.float64)
    q = np.arange(t, dtype=np.float64)[:, None] * delta
    dt = 0.1
    # k 帧速度 = k * delta / dt；k=2 → [2.0, 1.0, 0.4]，k=3 → [3.0, 1.5, 0.6]
    v_lim = np.asarray([10.0, 10.0, 0.4], dtype=np.float64)
    assert farthest_legal_index(q, a=0, v_lim=v_lim, dt=dt) == 2
    assert is_legal_jump(q[0], q[2], dt=dt, v_lim=v_lim)
    assert not is_legal_jump(q[0], q[3], dt=dt, v_lim=v_lim)


# 用例: 连续 H 次最远跳，返回长度 H 且在未贴末尾时严格递增
# 参数: 紧 v_lim 使每跳跨 1 帧，避免撞上轨迹终点
def test_build_skip_chunk_indices_length_and_monotonic(
    constant_speed_traj: np.ndarray,
) -> None:
    horizon = 5
    dt = 1.0 / horizon
    # 每帧 delta=[0.1,0.05,0.02]；跨 1 帧速度 = delta/dt
    # 设 v_lim 刚好只允许 k=1
    one_step_vel = np.asarray([0.1, 0.05, 0.02], dtype=np.float64) / dt
    v_lim = one_step_vel.copy()
    indices = build_skip_chunk_indices(
        constant_speed_traj, h0=0, horizon=horizon, v_lim=v_lim, dt=dt
    )
    assert indices.shape == (horizon,)
    assert indices.dtype.kind in "iu"
    np.testing.assert_array_equal(indices, np.asarray([1, 2, 3, 4, 5]))


# 用例: hx(=最后一跳下标) 是约束下本 chunk 能到的最远点
# 参数: 无；每跳固定跨 2 帧，H=5 → hx=10
def test_hx_is_farthest_under_constraint() -> None:
    q = np.arange(40, dtype=np.float64)[:, None]
    horizon = 5
    dt = 1.0 / horizon
    # k/dt <= v，要 max k=2 → v = 2/dt = 2*horizon
    v_lim = np.asarray([2.0 * horizon], dtype=np.float64)
    indices = build_skip_chunk_indices(q, h0=0, horizon=horizon, v_lim=v_lim, dt=dt)
    np.testing.assert_array_equal(indices, np.asarray([2, 4, 6, 8, 10]))
    assert int(indices[-1]) == 10


# 用例: 收紧 v_lim 后平均跨度不增
# 参数: constant_speed_traj；比较宽松与收紧上限
def test_lower_velocity_limit_reduces_average_span(
    constant_speed_traj: np.ndarray,
) -> None:
    horizon = 10
    dt = 1.0 / horizon
    base = np.asarray([2.0, 2.0, 2.0], dtype=np.float64)
    wide = build_skip_chunk_indices(
        constant_speed_traj, h0=0, horizon=horizon, v_lim=base, dt=dt
    )
    tight = build_skip_chunk_indices(
        constant_speed_traj, h0=0, horizon=horizon, v_lim=0.5 * base, dt=dt
    )

    def mean_span(idx: np.ndarray) -> float:
        prev = np.concatenate([[0], idx[:-1].astype(np.float64)])
        return float(np.mean(idx.astype(np.float64) - prev))

    assert mean_span(tight) <= mean_span(wide)


# 用例: 关闭跳帧时下标等于连续未来 H 步
# 参数: skip_frames=False 应得到 h0+1 … h0+H
def test_consecutive_mode_matches_dense_horizon(
    constant_speed_traj: np.ndarray,
) -> None:
    h0 = 3
    horizon = 7
    v_lim = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    indices = build_skip_chunk_indices(
        constant_speed_traj,
        h0=h0,
        horizon=horizon,
        v_lim=v_lim,
        skip_frames=False,
    )
    np.testing.assert_array_equal(
        indices, np.arange(h0 + 1, h0 + 1 + horizon, dtype=indices.dtype)
    )


# 用例: 轨迹末尾不够远跳时，下标不超过最后一帧，且可在终点重复
# 参数: 短轨迹；h0 靠近结尾；v_lim 很大
def test_near_episode_end_clamps_to_last_frame() -> None:
    q = np.linspace(0.0, 1.0, 6)[:, None]
    horizon = 5
    h0 = 3
    v_lim = np.asarray([100.0], dtype=np.float64)
    indices = build_skip_chunk_indices(
        q, h0=h0, horizon=horizon, v_lim=v_lim, dt=1.0 / horizon
    )
    assert indices.shape == (horizon,)
    assert int(indices.max()) == len(q) - 1
    assert int(indices[0]) > h0
    assert np.all(indices >= h0)


# 用例: 连相邻帧都超速时仍至少前进 1 帧（fallback），避免卡死
# 参数: 无；v_lim 极小，相邻帧也不合法
def test_fallback_advances_one_frame_when_even_neighbor_illegal() -> None:
    q = np.arange(10, dtype=np.float64)[:, None] * 10.0
    horizon = 3
    dt = 1.0 / horizon
    v_lim = np.asarray([1e-9], dtype=np.float64)
    indices = build_skip_chunk_indices(q, h0=0, horizon=horizon, v_lim=v_lim, dt=dt)
    np.testing.assert_array_equal(indices, np.asarray([1, 2, 3]))


# 用例: build_skip_chunk 按选出的下标取 q 行，形状 [H, D]
# 参数: constant_speed_traj
def test_build_skip_chunk_gathers_joint_rows(
    constant_speed_traj: np.ndarray,
) -> None:
    horizon = 6
    v_lim = np.asarray([3.0, 3.0, 3.0], dtype=np.float64)
    indices = build_skip_chunk_indices(
        constant_speed_traj, h0=0, horizon=horizon, v_lim=v_lim
    )
    chunk = build_skip_chunk(
        constant_speed_traj, h0=0, horizon=horizon, v_lim=v_lim
    )
    assert chunk.shape == (horizon, constant_speed_traj.shape[1])
    np.testing.assert_allclose(chunk, constant_speed_traj[indices])
