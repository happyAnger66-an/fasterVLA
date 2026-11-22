"""模块: fastervla.training.policy.dataset_assembly
功能: 组装带速度跳帧的训练 Dataset——包一层 base dataset，并用 window 生成 LeRobot delta_timestamps。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.training.policy.dataset_assembly import (
    build_lerobot_dataset_kwargs,
    wrap_dataset_with_velocity_skip,
)
from fastervla.training.policy.openpi_bridge import VelocitySkipConfig
from fastervla.training.resample.velocity_skip import (
    build_skip_chunk,
    build_skip_chunk_indices,
)


class _FakeWindowDataset:
    """Minimal random-access dataset yielding a long action window per index."""

    def __init__(self, traj: np.ndarray, window: int) -> None:
        self._traj = np.asarray(traj, dtype=np.float64)
        self._window = window

    def __len__(self) -> int:
        return max(0, len(self._traj) - self._window + 1)

    def __getitem__(self, index: int) -> dict:
        i = int(index)
        return {
            "actions": self._traj[i : i + self._window].copy(),
            "state": self._traj[i].copy(),
            "index": i,
        }


# 用例: wrap 后每个 sample 的 actions 变为 [H, D]，其它字段保留
# 参数: constant_speed_traj；skip 关闭时取窗口前缀
def test_wrap_dataset_dense_prefix(constant_speed_traj: np.ndarray) -> None:
    window, horizon = 12, 5
    base = _FakeWindowDataset(constant_speed_traj, window=window)
    cfg = VelocitySkipConfig(
        skip_frames=False,
        horizon=horizon,
        window=window,
        v_lim=np.ones(3),
        action_key="actions",
    )
    ds = wrap_dataset_with_velocity_skip(base, cfg)
    assert len(ds) == len(base)
    sample = ds[0]
    assert sample["actions"].shape == (horizon, 3)
    np.testing.assert_allclose(sample["actions"], constant_speed_traj[:horizon])
    np.testing.assert_allclose(sample["state"], constant_speed_traj[0])
    assert sample["index"] == 0


# 用例: skip 开启时 wrap 结果与直接 build_skip_chunk 一致
# 参数: 紧 v_lim 使每跳 1 帧
def test_wrap_dataset_skip_matches_velocity_skip(constant_speed_traj: np.ndarray) -> None:
    window, horizon = 20, 5
    dt = 1.0 / horizon
    v_lim = np.asarray([0.1, 0.05, 0.02], dtype=np.float64) / dt
    base = _FakeWindowDataset(constant_speed_traj, window=window)
    cfg = VelocitySkipConfig(
        skip_frames=True,
        horizon=horizon,
        window=window,
        v_lim=v_lim,
        action_key="actions",
        dt=dt,
    )
    sample = wrap_dataset_with_velocity_skip(base, cfg)[0]
    expected = build_skip_chunk(
        constant_speed_traj[:window], h0=0, horizon=horizon, v_lim=v_lim, dt=dt
    )
    np.testing.assert_allclose(sample["actions"], expected)


# 用例: 组装 LeRobot 构造参数时 delta_timestamps 长度等于 window
# 参数: fps=30；skip 开，window=40
def test_build_lerobot_dataset_kwargs_uses_window_timestamps() -> None:
    cfg = VelocitySkipConfig(
        skip_frames=True,
        horizon=10,
        window=40,
        v_lim=np.ones(7),
        action_key="actions",
    )
    kwargs = build_lerobot_dataset_kwargs(
        repo_id="local/demo",
        fps=30.0,
        action_keys=("actions",),
        skip_cfg=cfg,
    )
    assert kwargs["repo_id"] == "local/demo"
    assert len(kwargs["delta_timestamps"]["actions"]) == 40
    assert kwargs["delta_timestamps"]["actions"][1] == pytest.approx(1.0 / 30.0)


# 用例: skip 关闭时 kwargs 的时间戳长度等于 horizon（window==horizon）
# 参数: 无
def test_build_lerobot_dataset_kwargs_dense() -> None:
    cfg = VelocitySkipConfig(
        skip_frames=False,
        horizon=8,
        window=8,
        v_lim=np.ones(2),
    )
    kwargs = build_lerobot_dataset_kwargs(
        repo_id="local/demo",
        fps=20.0,
        action_keys=("action",),
        skip_cfg=cfg,
    )
    assert len(kwargs["delta_timestamps"]["action"]) == 8


# 用例: 30fps 名义拉窗 → wrap 后按 v 跳成功（每跳跨 2 帧）
# 参数: fps=30；window=20；horizon=5；整数位移避开 v_lim 卡边
def test_fps30_pull_window_then_velocity_skip_succeeds() -> None:
    fps = 30.0
    horizon = 5
    window = 20
    dt = 1.0 / horizon
    # 每帧位移；|Δq|_max = v_lim*dt = 2 → 跨 2 合法、跨 3 非法
    step = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    v_lim = np.full(3, 2.0 / dt, dtype=np.float64)

    cfg = VelocitySkipConfig(
        skip_frames=True,
        horizon=horizon,
        window=window,
        v_lim=v_lim,
        action_key="actions",
        dt=dt,
    )
    kwargs = build_lerobot_dataset_kwargs(
        repo_id="local/demo",
        fps=fps,
        action_keys=("actions",),
        skip_cfg=cfg,
    )
    stamps = kwargs["delta_timestamps"]["actions"]
    assert len(stamps) == window
    assert stamps[1] == pytest.approx(1.0 / fps)
    assert stamps[-1] == pytest.approx((window - 1) / fps)

    # 模拟按名义时间戳拉到的 window 帧（相邻帧间隔 1，对应 1/fps）
    traj = np.arange(window, dtype=np.float64)[:, None] * step
    base = _FakeWindowDataset(traj, window=window)
    sample = wrap_dataset_with_velocity_skip(base, cfg)[0]

    expected_idx = np.asarray([2, 4, 6, 8, 10], dtype=np.int64)
    np.testing.assert_array_equal(
        build_skip_chunk_indices(traj, h0=0, horizon=horizon, v_lim=v_lim, dt=dt),
        expected_idx,
    )
    np.testing.assert_allclose(sample["actions"], traj[expected_idx])
    assert sample["actions"].shape == (horizon, 3)


# 用例: 30fps 拉窗 → 非匀速轨迹上按 v 跳成功，各跳跨帧数不同（非等差）
# 参数: fps=30；|Δq|_max=20；手工摆点使最远跳为 [3,4,7,9,11]（跨度 3,1,3,2,2）
def test_fps30_pull_window_velocity_skip_varying_stride() -> None:
    fps = 30.0
    horizon = 5
    window = 16
    dt = 1.0 / horizon
    v_lim = np.full(3, 100.0, dtype=np.float64)  # |Δq| <= v_lim*dt = 20

    # 疏密不一的关节轨迹（三关节同形），避免浮点卡边
    q1 = np.asarray(
        [0, 5, 10, 15, 35, 40, 45, 55, 70, 75, 80, 95, 100, 105, 110, 115],
        dtype=np.float64,
    )
    traj = np.stack([q1, q1, q1], axis=1)
    expected_idx = np.asarray([3, 4, 7, 9, 11], dtype=np.int64)
    assert not np.all(np.diff(expected_idx) == np.diff(expected_idx)[0])

    cfg = VelocitySkipConfig(
        skip_frames=True,
        horizon=horizon,
        window=window,
        v_lim=v_lim,
        action_key="actions",
        dt=dt,
    )
    kwargs = build_lerobot_dataset_kwargs(
        repo_id="local/demo",
        fps=fps,
        action_keys=("actions",),
        skip_cfg=cfg,
    )
    assert len(kwargs["delta_timestamps"]["actions"]) == window
    assert kwargs["delta_timestamps"]["actions"][1] == pytest.approx(1.0 / fps)

    got_idx = build_skip_chunk_indices(
        traj, h0=0, horizon=horizon, v_lim=v_lim, dt=dt
    )
    np.testing.assert_array_equal(got_idx, expected_idx)

    base = _FakeWindowDataset(traj, window=window)
    sample = wrap_dataset_with_velocity_skip(base, cfg)[0]
    np.testing.assert_allclose(sample["actions"], traj[expected_idx])
    assert sample["actions"].shape == (horizon, 3)
