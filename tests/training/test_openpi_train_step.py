"""模块: fastervla.training.policy.openpi_bridge（T0b）
功能: 包一层跑通一小步训练（取样本→前向/反向/更新）；skip_frames 可开关。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Mapping

import numpy as np
import pytest

from fastervla.training.policy.dataset_assembly import wrap_dataset_with_velocity_skip
from fastervla.training.policy.openpi_bridge import (
    TrainStepResult,
    VelocitySkipConfig,
    run_one_train_step,
)
from fastervla.training.resample.velocity_skip import build_skip_chunk


class _FakeWindowDataset:
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


@dataclass
class _FakeOpenpiBackend:
    """Stand-in for openpi JAX: one step updates a scalar param from batch actions."""

    param: float = 0.0
    last_batch: dict[str, Any] | None = None
    calls: int = 0

    def train_step(self, batch: Mapping[str, Any]) -> float:
        actions = np.asarray(batch["actions"], dtype=np.float64)
        self.last_batch = dict(batch)
        self.calls += 1
        loss = float(np.mean(actions * actions))
        # 模拟「前向 → 反向 → 更新」：参数依赖 batch，保证有变化
        self.param = self.param + 1.0 + float(np.sum(actions))
        return loss


# 用例: skip 关闭时跑 1 step，loss 有限且参数更新；喂给 backend 的 actions 为窗口前缀
# 参数: horizon=5；window=12
def test_one_step_dense_finite_loss_and_param_update(
    constant_speed_traj: np.ndarray,
) -> None:
    window, horizon = 12, 5
    cfg = VelocitySkipConfig(
        skip_frames=False,
        horizon=horizon,
        window=window,
        v_lim=np.ones(3),
        action_key="actions",
    )
    base = _FakeWindowDataset(constant_speed_traj, window=window)
    backend = _FakeOpenpiBackend(param=0.0)
    before = backend.param

    result = run_one_train_step(
        dataset=base,
        skip_cfg=cfg,
        backend=backend,
        index=0,
    )

    assert isinstance(result, TrainStepResult)
    assert np.isfinite(result.loss)
    assert result.params_updated is True
    assert backend.calls == 1
    assert backend.param != before
    assert backend.last_batch is not None
    np.testing.assert_allclose(
        backend.last_batch["actions"],
        constant_speed_traj[:horizon],
    )
    assert backend.last_batch["actions"].shape == (horizon, 3)


# 用例: skip 开启时跑 1 step，loss 有限且参数更新；actions 与 build_skip_chunk 一致
# 参数: 紧 v_lim 使每跳 1 帧
def test_one_step_skip_matches_velocity_chunk_and_updates(
    constant_speed_traj: np.ndarray,
) -> None:
    window, horizon = 20, 5
    dt = 1.0 / horizon
    v_lim = np.asarray([0.1, 0.05, 0.02], dtype=np.float64) / dt
    cfg = VelocitySkipConfig(
        skip_frames=True,
        horizon=horizon,
        window=window,
        v_lim=v_lim,
        action_key="actions",
        dt=dt,
    )
    base = _FakeWindowDataset(constant_speed_traj, window=window)
    backend = _FakeOpenpiBackend(param=1.0)
    before = backend.param

    result = run_one_train_step(
        dataset=base,
        skip_cfg=cfg,
        backend=backend,
        index=0,
    )

    assert np.isfinite(result.loss)
    assert result.params_updated is True
    assert backend.param != before
    expected = build_skip_chunk(
        constant_speed_traj[:window],
        h0=0,
        horizon=horizon,
        v_lim=v_lim,
        dt=dt,
    )
    assert backend.last_batch is not None
    np.testing.assert_allclose(backend.last_batch["actions"], expected)


# 用例: 同一 backend 上 skip 开关都能各跑 1 step（T0b 验收：可开关跳帧）
# 参数: 无
def test_one_step_skip_toggle_both_paths(
    constant_speed_traj: np.ndarray,
) -> None:
    window, horizon = 16, 4
    dt = 1.0 / horizon
    v_lim = np.asarray([0.1, 0.05, 0.02], dtype=np.float64) / dt
    base = _FakeWindowDataset(constant_speed_traj, window=window)
    backend = _FakeOpenpiBackend()

    for skip in (False, True):
        cfg = VelocitySkipConfig(
            skip_frames=skip,
            horizon=horizon,
            window=window,
            v_lim=v_lim,
            action_key="actions",
            dt=dt,
        )
        result = run_one_train_step(
            dataset=base,
            skip_cfg=cfg,
            backend=backend,
            index=0,
        )
        assert np.isfinite(result.loss)
        assert result.params_updated is True
        assert backend.last_batch is not None
        assert backend.last_batch["actions"].shape == (horizon, 3)

    assert backend.calls == 2


# 用例: run_one_train_step 与手工 wrap 后取样本再 train_step 等价
# 参数: skip 开
def test_one_step_equivalent_to_manual_wrap(
    constant_speed_traj: np.ndarray,
) -> None:
    window, horizon = 20, 5
    dt = 1.0 / horizon
    v_lim = np.asarray([0.1, 0.05, 0.02], dtype=np.float64) / dt
    cfg = VelocitySkipConfig(
        skip_frames=True,
        horizon=horizon,
        window=window,
        v_lim=v_lim,
        action_key="actions",
        dt=dt,
    )
    base = _FakeWindowDataset(constant_speed_traj, window=window)

    wrapped_sample = wrap_dataset_with_velocity_skip(base, cfg)[0]
    backend_a = _FakeOpenpiBackend(param=0.0)
    loss_a = backend_a.train_step(wrapped_sample)

    backend_b = _FakeOpenpiBackend(param=0.0)
    result_b = run_one_train_step(
        dataset=base,
        skip_cfg=cfg,
        backend=backend_b,
        index=0,
    )

    assert result_b.loss == pytest.approx(loss_a)
    assert backend_b.param == pytest.approx(backend_a.param)
    np.testing.assert_allclose(
        backend_b.last_batch["actions"],
        backend_a.last_batch["actions"],
    )
