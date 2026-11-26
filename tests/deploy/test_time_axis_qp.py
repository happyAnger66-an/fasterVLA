"""模块: fastervla.deploy.postprocess.retime.TimeAxisQP
功能: L2 时间轴 QP；路点不变；dt 盒约束 + 逐关节 v_max；prefix 锁定；失败退 Uniform。

F3 测试（先测后码）。行为对齐老栈 arm_timeline.py / test_arm_timeline.py。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.core.types import Chunk
from fastervla.deploy.postprocess.retime import (
    TimeAxisQP,
    TimeAxisQPConfig,
    UniformRetimer,
    box_qp,
    contact_mask,
    rate_hessian,
)


def _motion(t: np.ndarray) -> np.ndarray:
    phases = np.linspace(0.0, 2.0, 14)
    return 0.3 * np.sin(2 * np.pi * 0.7 * np.asarray(t)[..., None] + phases)


def _chunk(*, fps: float = 30.0, origin: float = 0.0) -> Chunk:
    h, d = 30, 24
    wp = np.zeros((h, d), dtype=np.float32)
    wp[:, 8:22] = _motion(origin + np.arange(1, h + 1) / fps).astype(np.float32)
    wp[:, 22:] = 0.5
    return Chunk(waypoints=wp, model_dt=1.0 / fps, obs_ts=origin)


def _state24(*, origin: float = 0.0) -> np.ndarray:
    q = np.zeros(24, dtype=np.float64)
    q[8:22] = _motion(np.asarray([origin]))[0]
    return q


def _cfg(**kwargs: object) -> TimeAxisQPConfig:
    base = dict(
        policy_fps=30.0,
        kappa=1.2,
        kappa_max=1.5,
        kappa_min=1.0,
        dt_min_s=0.015,
        dt_max_s=0.08,
        lambda_time=0.1,
        lambda_acc=1.0,
        v_max_rad_s=None,
        buffer_window_enabled=False,
        solver="active_set",
    )
    base.update(kwargs)
    return TimeAxisQPConfig(**base)  # type: ignore[arg-type]


# 用例: kappa 必须落在 [kappa_min, kappa_max]
# 参数: 无
def test_time_axis_qp_rejects_kappa_above_max() -> None:
    with pytest.raises((ValueError, TypeError)):
        TimeAxisQPConfig(kappa=1.6, kappa_max=1.5, kappa_min=1.0)


# 用例: 1/(policy_fps·κ) 必须落在 [dt_min, dt_max]
# 参数: 无
def test_time_axis_qp_rejects_dt_ref_outside_box() -> None:
    with pytest.raises((ValueError, TypeError)):
        # κ=2 → dt_ref≈16.7ms but dt_min=20ms
        TimeAxisQPConfig(policy_fps=30.0, kappa=2.0, kappa_max=2.0, dt_min_s=0.02, dt_max_s=0.08)


# 用例: retime 不改变 waypoints；返回 TimedChunk，step_dt 长度 = H
# 参数: 无
def test_time_axis_qp_preserves_waypoints() -> None:
    src = _chunk()
    qp = TimeAxisQP(_cfg(kappa=1.2))
    timed = qp.retime(src, state=_state24())
    np.testing.assert_allclose(timed.chunk.waypoints, src.waypoints)
    assert timed.step_dt.shape == (30,)
    assert np.isfinite(timed.step_dt).all()
    assert timed.chunk.meta.get("retime_status") in ("qp", "active_set", "ok", "lbfgsb")


# 用例: 全部 step_dt 落在 [dt_min, dt_max]
# 参数: kappa
@pytest.mark.parametrize("kappa", [1.0, 1.2, 1.4])
def test_time_axis_qp_dt_box_constraints(kappa: float) -> None:
    cfg = _cfg(kappa=kappa, kappa_max=1.5)
    timed = TimeAxisQP(cfg).retime(_chunk(), state=_state24())
    assert np.all(timed.step_dt >= cfg.dt_min_s - 1e-9)
    assert np.all(timed.step_dt <= cfg.dt_max_s + 1e-9)


# 用例: κ=1.2 时总时长相对 κ=1.0 明显缩短；executed_speed ≈ κ（允许小误差）
# 参数: 无
def test_time_axis_qp_kappa_1p2_speeds_up() -> None:
    chunk = _chunk()
    state = _state24()
    t1 = TimeAxisQP(_cfg(kappa=1.0)).retime(chunk, state=state)
    t12 = TimeAxisQP(_cfg(kappa=1.2)).retime(chunk, state=state)
    assert float(np.sum(t12.step_dt)) < float(np.sum(t1.step_dt)) * 0.95
    meta = t12.chunk.meta
    # Prefer explicit executed_speed if present; else infer from total duration.
    if "executed_speed" in meta:
        assert abs(float(meta["executed_speed"]) - 1.2) <= 0.05
    else:
        # H / policy_fps / sum(dt) ≈ kappa
        speed = 30.0 / 30.0 / float(np.sum(t12.step_dt))
        assert abs(speed - 1.2) <= 0.08


# 用例: κ=1.4 总时长再短于 1.2
# 参数: 无
def test_time_axis_qp_kappa_1p4_faster_than_1p2() -> None:
    chunk = _chunk()
    state = _state24()
    t12 = TimeAxisQP(_cfg(kappa=1.2)).retime(chunk, state=state)
    t14 = TimeAxisQP(_cfg(kappa=1.4, kappa_max=1.5)).retime(chunk, state=state)
    assert float(np.sum(t14.step_dt)) < float(np.sum(t12.step_dt)) * 0.98


# 用例: 大位移帧被 v_max 限速（|Δq|/dt ≤ v_max）
# 参数: 无
def test_time_axis_qp_velocity_limit_slows_fast_frame() -> None:
    chunk = _chunk()
    wp = chunk.waypoints.copy()
    wp[5:, 8] = wp[5:, 8] + 0.1  # ~0.1 rad jump into frame 5
    chunk = Chunk(waypoints=wp, model_dt=chunk.model_dt, obs_ts=chunk.obs_ts)
    v_max = 2.0
    timed = TimeAxisQP(
        _cfg(kappa=1.2, v_max_rad_s=v_max, v_max_from_lag=False)
    ).retime(chunk, state=_state24())
    # Segment i duration is step_dt[i] between waypoint i and i+1? 
    # Contract: |wp[5,8]-wp[4,8]| / step_dt[5] ≤ v_max  (duration of arriving at frame 5)
    # Align with old stack: times[k] ends frame k-1; duration for frame i is times[i+1]-times[i] = step_dt[i].
    dp = abs(float(wp[5, 8] - wp[4, 8]))
    assert dp / float(timed.step_dt[5]) <= v_max + 1e-6
    hits = timed.chunk.meta.get("velocity_limited_frames", [])
    assert 5 in list(hits)


# 用例: lock_durations 锁定前缀帧时长，不受 κ 拉动
# 参数: 无
def test_time_axis_qp_prefix_lock_durations() -> None:
    lock = np.full(4, 1.0 / 36.0, dtype=np.float64)
    timed = TimeAxisQP(_cfg(kappa=1.2)).retime(
        _chunk(), state=_state24(), lock_durations=lock
    )
    np.testing.assert_allclose(timed.step_dt[:4], lock, rtol=0, atol=1e-12)
    assert int(timed.chunk.meta.get("locked_frames", 0)) == 4


# 用例: buffer_window_enabled 时，prefix..prefix+buffer 窗墙钟 ≥ r2r 预算
# 参数: 无
def test_time_axis_qp_optional_buffer_window_covers_r2r() -> None:
    cfg = _cfg(
        kappa=1.5,
        kappa_max=1.5,
        buffer_window_enabled=True,
        prefix_frames=4,
        buffer_frames=4,
        r2r_default_ms=100.0,
        buffer_margin_ms=20.0,
    )
    timed = TimeAxisQP(cfg).retime(
        _chunk(),
        state=_state24(),
        r2r_samples_ms=[90.0] * 30,
    )
    prefix, buffer = 4, 4
    window = float(np.sum(timed.step_dt[prefix : prefix + buffer]))
    budget = (90.0 + 20.0) / 1000.0  # p95≈90 with enough samples, + margin
    assert window + 1e-9 >= budget


# 用例: active-set box_qp 与 L-BFGS-B 在随机盒问题上目标值一致（≤）
# 参数: 无
def test_box_qp_active_set_matches_lbfgsb() -> None:
    scipy = pytest.importorskip("scipy")
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)
    for _ in range(25):
        dp = rng.normal(scale=0.02, size=(30, 14))
        hessian, wt = rate_hessian(dp, s_ref=36.0, lambda_time=0.1, lambda_acc=float(rng.uniform(0.5, 10.0)))
        target = float(rng.uniform(30, 60))
        linear = -2 * wt * target * np.ones(30)
        lower = np.full(30, 12.5)
        upper = rng.uniform(25.0, 66.0, size=30)
        upper[:3] = lower[:3] = rng.uniform(25.0, 40.0, size=3)
        x, kkt = box_qp(hessian, linear, lower, upper)
        assert kkt is True
        ref = minimize(
            lambda s: (0.5 * s @ hessian @ s + linear @ s, hessian @ s + linear),
            np.clip(np.full(30, 36.0), lower, upper),
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000},
        )
        obj = lambda s: 0.5 * s @ hessian @ s + linear @ s  # noqa: E731
        assert obj(x) <= obj(np.clip(ref.x, lower, upper)) + 1e-9


# 用例: 若安装了 osqp，active_set 与 osqp 在同一盒问题上接近
# 参数: 无
def test_box_qp_osqp_matches_active_set_when_available() -> None:
    pytest.importorskip("osqp")
    rng = np.random.default_rng(1)
    dp = rng.normal(scale=0.02, size=(30, 14))
    hessian, wt = rate_hessian(dp, s_ref=36.0, lambda_time=0.1, lambda_acc=1.0)
    linear = -2 * wt * 36.0 * np.ones(30)
    lower = np.full(30, 12.5)
    upper = np.full(30, 66.0)
    x_as, _ = box_qp(hessian, linear, lower, upper)
    qp = TimeAxisQP(_cfg(solver="osqp"))
    x_osqp = qp.solve_box(hessian, linear, lower, upper)
    np.testing.assert_allclose(x_osqp, x_as, rtol=1e-4, atol=1e-4)


# 用例: 求解失败时退回 UniformRetimer，meta.retime_status 标明 fallback
# 参数: 无
def test_time_axis_qp_falls_back_to_uniform_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    qp = TimeAxisQP(_cfg(kappa=1.2))

    def boom(*_a, **_k):
        raise RuntimeError("forced qp failure")

    monkeypatch.setattr(qp, "_solve_rates", boom)
    src = _chunk()
    timed = qp.retime(src, state=_state24())
    np.testing.assert_allclose(timed.chunk.waypoints, src.waypoints)
    status = str(timed.chunk.meta.get("retime_status", ""))
    assert "uniform" in status or "fallback" in status
    # Matches Uniform at same kappa.
    uni = UniformRetimer(policy_fps=30.0, kappa=1.2).retime(src)
    np.testing.assert_allclose(timed.step_dt, uni.step_dt, rtol=0, atol=1e-12)


def _grasp_chunk() -> Chunk:
    """Small arm steps + gripper closing at frames 12..16 (fine phase)."""
    c = _chunk()
    wp = c.waypoints.copy()
    wp[10:20, 8:22] = wp[10, 8:22]  # near-still arm around the grasp
    wp[:, 23] = -0.8
    wp[12:17, 23] = np.linspace(-0.8, 0.0, 5)
    wp[17:, 23] = 0.0
    return Chunk(waypoints=wp, model_dt=c.model_dt, obs_ts=c.obs_ts)


# 用例: contact_mask 标出夹爪变化前后 window 帧
# 参数: 无
def test_contact_mask_dilates_gripper_events() -> None:
    # Gripper moves on frames 13..16 → ±2 frames → 11..18.
    m = contact_mask(_grasp_chunk().waypoints, grip_dims=(22, 23), delta=0.02, window=2)
    assert m[11:19].all()
    assert not m[:11].any() and not m[19:].any()


# 用例: contact_window_scale_kappa 时窗按 κ 拉长（墙钟软区近似不变）
# 参数: 无
def test_time_axis_qp_contact_window_scales_with_kappa() -> None:
    chunk = _grasp_chunk()
    plain = TimeAxisQP(
        _cfg(kappa=1.5, contact_kappa=1.0, contact_window_frames=2)
    ).retime(chunk, state=_state24())
    scaled = TimeAxisQP(
        _cfg(
            kappa=1.5,
            contact_kappa=1.0,
            contact_window_frames=2,
            contact_window_scale_kappa=True,
        )
    ).retime(chunk, state=_state24())
    assert int(plain.chunk.meta["contact_window_frames_used"]) == 2
    assert int(scaled.chunk.meta["contact_window_frames_used"]) == 3  # ceil(2*1.5)
    assert int(scaled.chunk.meta["contact_frames"]) >= int(
        plain.chunk.meta["contact_frames"]
    )


# 用例: contact_kappa 把夹爪窗内速率限制在 policy_fps·contact_kappa
# 参数: 无
def test_time_axis_qp_contact_cap_limits_rate_near_grip() -> None:
    chunk = _grasp_chunk()
    base = TimeAxisQP(_cfg(kappa=1.2)).retime(chunk, state=_state24())
    capped = TimeAxisQP(
        _cfg(kappa=1.2, contact_kappa=1.0, contact_window_frames=2)
    ).retime(chunk, state=_state24())
    near = slice(11, 19)
    assert float(np.max(base.speed[near])) > 30.0 * 1.05  # unconstrained QP speeds up fine phase
    assert float(np.max(capped.speed[near])) <= 30.0 * 1.0 + 1e-6
    assert int(capped.chunk.meta["contact_frames"]) == 8


# 用例: 夹爪窗减速后，校准仍让整块平均速率对齐 κ（时间挪到搬运段）
# 参数: 无
def test_time_axis_qp_contact_cap_keeps_mean_speed() -> None:
    chunk = _grasp_chunk()
    capped = TimeAxisQP(
        _cfg(kappa=1.2, contact_kappa=1.0, contact_window_frames=2)
    ).retime(chunk, state=_state24())
    mean_speed = 30.0 / 30.0 / float(np.sum(capped.step_dt))
    assert abs(mean_speed - 1.2) <= 0.03
    far = np.r_[0:11, 19:30]
    assert float(np.mean(capped.speed[far])) > 36.0


# 用例: transport_kappa_max 限制所有帧速率
# 参数: 无
def test_time_axis_qp_transport_cap() -> None:
    timed = TimeAxisQP(_cfg(kappa=1.2, transport_kappa_max=1.4)).retime(
        _grasp_chunk(), state=_state24()
    )
    assert float(np.max(timed.speed)) <= 30.0 * 1.4 + 1e-6


# 用例: transport_kappa_max < kappa 非法
# 参数: 无
def test_time_axis_qp_rejects_transport_cap_below_kappa() -> None:
    with pytest.raises(ValueError):
        _cfg(kappa=1.2, transport_kappa_max=1.1)


def _dwell_chunk() -> Chunk:
    """Mostly stationary arms, with a mid-chunk grasp edge."""
    c = _chunk()
    wp = np.asarray(c.waypoints, dtype=np.float32).copy()
    # Freeze arms: tiny steps so dwell_mask fires.
    wp[:, 8:22] = 0.1
    wp[15:, 8:22] = 0.1005  # still < 0.005 step from previous after state
    wp[:, 22:] = 0.5
    wp[14:17, 23] = -0.5  # gripper event around frame 14
    return Chunk(waypoints=wp, model_dt=c.model_dt, obs_ts=c.obs_ts)


# 用例: dwell_kappa_max 把远离接触的静止帧钉到更高倍速；夹爪窗仍受 contact 限制
# 参数: 无
def test_time_axis_qp_dwell_compresses_stationary_far_from_grip() -> None:
    from fastervla.deploy.postprocess.retime import contact_mask, dwell_mask

    chunk = _dwell_chunk()
    st = np.zeros(24, dtype=np.float64)
    st[8:22] = 0.1
    base = TimeAxisQP(
        _cfg(kappa=1.2, contact_kappa=1.0, transport_kappa_max=1.4, contact_window_frames=2)
    ).retime(chunk, state=st)
    dwell = TimeAxisQP(
        _cfg(
            kappa=1.2,
            contact_kappa=1.0,
            transport_kappa_max=1.4,
            dwell_kappa_max=2.0,
            dwell_step_rad=0.005,
            contact_window_frames=2,
        )
    ).retime(chunk, state=st)
    pts = np.vstack([st[8:22], chunk.waypoints[:, 8:22]])
    dp = np.diff(pts, axis=0)
    near = contact_mask(chunk.waypoints, grip_dims=(22, 23), delta=0.02, window=2)
    dm = dwell_mask(dp, 0.005) & ~near
    assert dm.any()
    assert int(dwell.chunk.meta["dwell_frames"]) == int(dm.sum())
    # Dwell frames reach ~2×; transport-only run stays ≤1.4×.
    assert float(np.min(dwell.speed[dm])) >= 30.0 * 1.95
    assert float(np.max(base.speed)) <= 30.0 * 1.4 + 1e-6
    # Contact window still capped.
    assert float(np.max(dwell.speed[near])) <= 30.0 * 1.0 + 1e-6


# 用例: dwell_kappa_max < kappa 非法
# 参数: 无
def test_time_axis_qp_rejects_dwell_below_kappa() -> None:
    with pytest.raises(ValueError):
        _cfg(kappa=1.2, dwell_kappa_max=1.1)


# 用例: 与 Uniform 接口兼容 — PlanBuffer 能在 QP 的非均匀 step_dt 上 advance
# 参数: 无
def test_time_axis_qp_step_dt_works_with_plan_buffer() -> None:
    from fastervla.deploy.client.plan_buffer import PlanBuffer

    timed = TimeAxisQP(_cfg(kappa=1.2)).retime(_chunk(), state=_state24())
    buf = PlanBuffer()
    buf.load(timed, origin_monotonic=0.0)
    total = float(np.sum(timed.step_dt))
    buf.advance(total * 0.5, alpha=1.0)
    assert 0.0 < buf.progress < 30.0
    q = buf.reference(t_now=0.0, horizon_s=0.0)
    assert q.shape == (24,) and np.isfinite(q).all()
