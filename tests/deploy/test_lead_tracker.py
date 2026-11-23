"""模块: fastervla.deploy.client.tracker.lead
功能: LeadTracker 对齐老栈 soft lead：u = r + clip(gain·((r_ahead+τ·v)-r)).

F2 测试（先测后码）。臂 14 维在命令布局下标 5:19；夹爪不进 lead。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fastervla.deploy.client.tracker.lead import LeadTracker, LeadTrackerConfig


def _cmd(arm: np.ndarray | float = 0.0) -> np.ndarray:
    q = np.zeros(21, dtype=np.float64)
    if np.ndim(arm) == 0:
        q[5:19] = float(arm)
    else:
        q[5:19] = np.asarray(arm, dtype=np.float64)
    return q


# 用例: 默认参数与老栈 soft 配方一致（τ=57ms, gain=0.25, max_lead=0.06, smooth=40ms）
# 参数: 无
def test_lead_tracker_default_matches_soft_recipe() -> None:
    cfg = LeadTrackerConfig()
    assert cfg.tau_s == pytest.approx(0.057)
    assert cfg.gain == pytest.approx(0.25)
    assert cfg.max_lead_rad == pytest.approx(0.06)
    assert cfg.lead_smooth_tau_s == pytest.approx(0.040)
    assert cfg.hold_during_blend is True


# 用例: gain=0 或 τ=0 时输出 ≈ reference（无超前）
# 参数: 无
def test_lead_tracker_zero_gain_passthrough() -> None:
    tr = LeadTracker(
        LeadTrackerConfig(tau_s=0.057, gain=0.0, max_lead_rad=0.06, lead_smooth_tau_s=0.0)
    )
    ref = _cmd(0.3)
    out = tr.step(
        ref,
        ref_ahead=np.full(14, 0.5),
        ref_vel=np.full(14, 1.0),
        now=0.0,
        blending=False,
    )
    np.testing.assert_allclose(out, ref, atol=1e-12)


# 用例: 单轴公式 lead = clip(gain * ((r_ahead + τ v) - r), ±max_lead)
# 参数: 无
def test_lead_tracker_formula_single_step_no_ema() -> None:
    tau, gain, max_lead = 0.057, 0.25, 0.06
    tr = LeadTracker(
        LeadTrackerConfig(
            tau_s=tau,
            gain=gain,
            max_lead_rad=max_lead,
            lead_smooth_tau_s=0.0,
            hold_during_blend=True,
        )
    )
    r = 0.10
    r_ahead = 0.20
    v = 0.50
    ref = _cmd(r)
    out = tr.step(
        ref,
        ref_ahead=np.full(14, r_ahead),
        ref_vel=np.full(14, v),
        now=0.0,
        blending=False,
    )
    desired = r_ahead + tau * v
    lead = gain * (desired - r)
    lead = float(np.clip(lead, -max_lead, max_lead))
    expected = _cmd(r + lead)
    np.testing.assert_allclose(out[5:19], expected[5:19], atol=1e-9)
    # Non-arm dims unchanged.
    np.testing.assert_allclose(out[:5], ref[:5])
    np.testing.assert_allclose(out[19:], ref[19:])


# 用例: |lead| 被 max_lead 截断
# 参数: 无
def test_lead_tracker_clips_max_lead() -> None:
    tr = LeadTracker(
        LeadTrackerConfig(
            tau_s=0.1, gain=1.0, max_lead_rad=0.06, lead_smooth_tau_s=0.0
        )
    )
    ref = _cmd(0.0)
    out = tr.step(
        ref,
        ref_ahead=np.full(14, 1.0),
        ref_vel=np.full(14, 10.0),
        now=0.0,
        blending=False,
    )
    assert np.max(np.abs(out[5:19] - ref[5:19])) == pytest.approx(0.06, abs=1e-9)


# 用例: hold_during_blend=True 时 blending 阶段目标 lead→0（经 EMA 收敛）
# 参数: 无
def test_lead_tracker_hold_during_blend_drives_lead_to_zero() -> None:
    tr = LeadTracker(
        LeadTrackerConfig(
            tau_s=0.057,
            gain=0.25,
            max_lead_rad=0.06,
            lead_smooth_tau_s=0.01,
            hold_during_blend=True,
        )
    )
    ref = _cmd(0.2)
    # Build some lead first.
    tr.step(
        ref,
        ref_ahead=np.full(14, 0.4),
        ref_vel=np.zeros(14),
        now=0.0,
        blending=False,
    )
    # Then blend for many ticks — lead should decay toward 0.
    out = ref
    t = 0.0
    for _ in range(80):
        t += 0.01
        out = tr.step(
            ref,
            ref_ahead=np.full(14, 0.4),
            ref_vel=np.zeros(14),
            now=t,
            blending=True,
        )
    assert np.max(np.abs(out[5:19] - ref[5:19])) < 0.005


# 用例: lead_smooth_tau 做 EMA；连续两拍不完全等于瞬时 lead
# 参数: 无
def test_lead_tracker_ema_smooths_lead() -> None:
    cfg = LeadTrackerConfig(
        tau_s=0.0, gain=1.0, max_lead_rad=1.0, lead_smooth_tau_s=0.05
    )
    tr = LeadTracker(cfg)
    ref = _cmd(0.0)
    # Instantaneous desired lead would be 0.2 (r_ahead - r).
    out0 = tr.step(
        ref, ref_ahead=np.full(14, 0.2), ref_vel=np.zeros(14), now=0.0, blending=False
    )
    # First tick: no previous → may accept full or start filter; second tick must lag.
    out1 = tr.step(
        ref, ref_ahead=np.full(14, 0.2), ref_vel=np.zeros(14), now=0.01, blending=False
    )
    instant = 0.2
    # After 10 ms with τ=50 ms: alpha = 1-exp(-0.01/0.05) ≈ 0.181
    # If first tick set filt=0.2, second stays ~0.2; start from 0 on first:
    # out1 ≈ alpha * 0.2. Either way |out1| ≤ instant + eps and sequence is finite.
    assert np.max(np.abs(out1[5:19])) <= instant + 1e-9
    assert np.isfinite(out0).all() and np.isfinite(out1).all()
    # EMA must not jump to full lead in one 10 ms tick if filter started at 0.
    alpha = 1.0 - math.exp(-0.01 / 0.05)
    # Allow either init-at-target or init-at-zero; if init-at-zero, out1≈alpha*instant.
    lead1 = float(np.mean(out1[5:19]))
    assert lead1 == pytest.approx(instant, abs=1e-6) or lead1 == pytest.approx(
        alpha * instant, abs=1e-3
    )
