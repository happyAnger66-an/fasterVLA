"""模块: fastervla.deploy.client.plan_buffer
功能: 模型步尺子；advance / prefix(d̂) / merge；晚到丢弃；非均匀 step_dt。

F2 测试（先测后码）。实现须等 review 通过。
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fastervla.core.types import Chunk, Prefix, TimedChunk
from fastervla.deploy.client.plan_buffer import PlanBuffer


def _waypoints(h: int = 30, d: int = 24, base: float = 0.0) -> np.ndarray:
    # Distinct per-step values so merge splice points are observable.
    t = np.arange(h, dtype=np.float64)[:, None]
    j = np.arange(d, dtype=np.float64)[None, :]
    return (base + 0.01 * t + 0.001 * j).astype(np.float64)


def _timed(
    *,
    h: int = 30,
    step_dt: np.ndarray | None = None,
    base: float = 0.0,
    plan_generation: int | None = None,
) -> TimedChunk:
    wp = _waypoints(h=h, base=base)
    if step_dt is None:
        step_dt = np.full(h, 1.0 / 30.0, dtype=np.float64)
    speed = 1.0 / np.maximum(step_dt, 1e-9)
    chunk = Chunk(waypoints=wp.astype(np.float32), model_dt=float(step_dt[0]), obs_ts=0.0)
    timed = TimedChunk(chunk=chunk, step_dt=step_dt, speed=speed)
    if plan_generation is not None:
        timed.chunk.meta["plan_generation"] = int(plan_generation)
    return timed


def _buffer(timed: TimedChunk | None = None, *, origin: float = 0.0) -> PlanBuffer:
    buf = PlanBuffer()
    buf.load(timed or _timed(), origin_monotonic=origin)
    return buf


# 用例: load 后 progress=0，horizon/action_dim 正确；waypoints 可读取
# 参数: 无
def test_plan_buffer_load_initial_progress() -> None:
    timed = _timed(h=30)
    buf = _buffer(timed)
    assert buf.progress == pytest.approx(0.0)
    assert buf.horizon == 30
    assert buf.action_dim == 24
    np.testing.assert_allclose(buf.waypoints, timed.chunk.waypoints, rtol=0, atol=0)


# 用例: 均匀 step_dt 下 advance(dt) 推进的模型步 ≈ dt / step_dt
# 参数: 无
def test_plan_buffer_advance_uniform_step_dt() -> None:
    dt = 1.0 / 30.0
    buf = _buffer(_timed(step_dt=np.full(30, dt)))
    buf.advance(dt, alpha=1.0)
    assert buf.progress == pytest.approx(1.0, abs=1e-9)
    buf.advance(2.5 * dt, alpha=1.0)
    assert buf.progress == pytest.approx(3.5, abs=1e-9)


# 用例: 非均匀 step_dt 上 advance 按段累加墙钟（不能当常数 1/fps）
# 参数: 无
def test_plan_buffer_advance_nonuniform_step_dt() -> None:
    step_dt = np.array([0.02, 0.05, 0.01, 0.04] + [0.03] * 26, dtype=np.float64)
    buf = _buffer(_timed(step_dt=step_dt))
    # Walk exactly through first two segments: 0.02 + 0.05 = 0.07 → progress=2.0
    buf.advance(0.07, alpha=1.0)
    assert buf.progress == pytest.approx(2.0, abs=1e-9)
    # Half of third segment (0.01): +0.5 model steps
    buf.advance(0.005, alpha=1.0)
    assert buf.progress == pytest.approx(2.5, abs=1e-9)


# 用例: alpha=0 时 advance 不推进（接触冻结）
# 参数: 无
def test_plan_buffer_advance_alpha_zero_freezes() -> None:
    buf = _buffer()
    buf.advance(0.1, alpha=1.0)
    p = buf.progress
    buf.advance(1.0, alpha=0.0)
    assert buf.progress == pytest.approx(p)


# 用例: prefix(d_hat) 返回从 progress 起未来 d_hat 个模型步（绝对量）+ anchor_step
# 参数: 无
def test_plan_buffer_prefix_shape_and_anchor() -> None:
    timed = _timed()
    buf = _buffer(timed)
    buf.advance(4.0 / 30.0, alpha=1.0)  # → progress≈4 with uniform 1/30
    # Snap to exact for contract clarity if implementation uses continuous progress.
    # Require prefix length == d_hat and values match waypoints near ceil(progress).
    d_hat = 5
    prefix = buf.prefix(d_hat)
    assert isinstance(prefix, Prefix)
    assert prefix.actions.shape == (d_hat, 24)
    assert np.isfinite(prefix.actions).all()
    assert prefix.anchor_step == int(np.floor(buf.progress))
    # Absolute waypoints: first prefix row ≈ waypoint at floor(progress)
    start = prefix.anchor_step
    np.testing.assert_allclose(
        prefix.actions[0],
        timed.chunk.waypoints[start],
        rtol=0,
        atol=1e-5,
    )


# 用例: estimate_delay_steps(L_hat) 沿 step_dt 从 progress 累加到覆盖 L_hat（上取整）
# 参数: 无
def test_plan_buffer_estimate_delay_steps_ceil() -> None:
    step_dt = np.full(30, 0.05, dtype=np.float64)  # 50 ms / step
    buf = _buffer(_timed(step_dt=step_dt))
    # L_hat = 0.12 s → covers 0.05+0.05+0.02 into 3rd step → ceil = 3
    assert buf.estimate_delay_steps(0.12) == 3
    assert buf.estimate_delay_steps(0.05) == 1
    assert buf.estimate_delay_steps(0.051) == 2


# 用例: merge 按实际 progress 相对 anchor 算 d，从新 chunk[d:] 接入；返回 True
# 参数: 无
def test_plan_buffer_merge_splices_at_actual_d() -> None:
    old = _timed(base=0.0)
    buf = _buffer(old)
    # Freeze prefix at progress=0, d_hat conceptually 4; then advance 2 steps worth.
    prefix = buf.prefix(4)
    assert prefix.anchor_step == 0
    buf.advance(2.0 / 30.0, alpha=1.0)  # actual d ≈ 2
    new = _timed(base=10.0)  # clearly different values
    ok = buf.merge(new, anchor_step=prefix.anchor_step)
    assert ok is True
    # After merge, waypoint at current progress should come from new chunk index ≈ d.
    # progress≈2 → new.waypoints[2] near current reference sample.
    ref = buf.waypoints
    np.testing.assert_allclose(ref[2], new.chunk.waypoints[2], rtol=0, atol=1e-5)
    # Meta: last merge records d_hat/d if available; at least d_actual ≈ 2
    assert abs(float(buf.last_merge["d_actual"]) - 2.0) <= 1.0 + 1e-6


# 用例: 比当前计划旧的结果 merge 返回 False 且不改 waypoints
# 参数: 无
def test_plan_buffer_merge_discards_stale() -> None:
    buf = _buffer(_timed(base=0.0))
    buf.merge(_timed(base=1.0), anchor_step=0)  # gen bump
    before = buf.waypoints.copy()
    plan_id_before = buf.plan_id
    # Stale: older generation / older than current plan.
    stale = _timed(base=99.0, plan_generation=0)
    ok = buf.merge(stale, anchor_step=0)
    assert ok is False
    np.testing.assert_allclose(buf.waypoints, before)
    assert buf.plan_id == plan_id_before


# 用例: d_hat 与实际 d 在模拟延迟下误差 ≤ 1 模型步
# 参数: 无
def test_plan_buffer_d_hat_vs_d_actual_within_one_step() -> None:
    step_dt = np.full(30, 1.0 / 30.0, dtype=np.float64)
    buf = _buffer(_timed(step_dt=step_dt))
    latency_s = 0.10  # 100 ms
    d_hat = buf.estimate_delay_steps(latency_s)
    prefix = buf.prefix(d_hat)
    # Simulate control advancing for exactly latency_s while "inferring".
    buf.advance(latency_s, alpha=1.0)
    new = _timed(base=3.0, step_dt=step_dt)
    assert buf.merge(new, anchor_step=prefix.anchor_step) is True
    d_actual = float(buf.last_merge["d_actual"])
    assert abs(d_actual - d_hat) <= 1.0 + 1e-6


# 用例: reference 在真实时间上返回有限插值（长度 = action_dim）
# 参数: 无
def test_plan_buffer_reference_returns_finite_vector() -> None:
    buf = _buffer(_timed())
    buf.advance(0.05, alpha=1.0)
    q = buf.reference(t_now=buf.origin_monotonic + 0.05, horizon_s=0.0)
    assert np.asarray(q).shape == (24,)
    assert np.isfinite(q).all()


# 用例: 控制环 advance 与推理线程 prefix/merge 可并发不崩（冒烟）
# 参数: 无
def test_plan_buffer_thread_smoke() -> None:
    buf = _buffer(_timed())
    err: list[BaseException] = []

    def control() -> None:
        try:
            for _ in range(200):
                buf.advance(0.01, alpha=1.0)
                _ = buf.reference(t_now=0.0, horizon_s=0.0)
                time.sleep(0.0005)
        except BaseException as e:  # noqa: BLE001 — collect for main thread
            err.append(e)

    def infer() -> None:
        try:
            for i in range(20):
                d_hat = max(1, buf.estimate_delay_steps(0.05))
                pref = buf.prefix(d_hat)
                time.sleep(0.002)
                buf.merge(_timed(base=float(i)), anchor_step=pref.anchor_step)
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t1 = threading.Thread(target=control)
    t2 = threading.Thread(target=infer)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not err, f"thread errors: {err}"
