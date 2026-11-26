"""模块: fastervla.deploy.analysis.control_quality
功能: 跟随时滞 / 轮廓误差 / 频响 / 夹爪事件；用已知纯延时的合成轨迹验证各估计量。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.deploy.analysis.control_quality import (
    control_headline,
    control_quality_metrics,
    path_lag,
    sparc,
)
from fastervla.deploy.client.recorder import STAGE_NAMES, STAGE_STEP

DT = 0.01


def _synthetic_ctl(delay_ticks: int, lead_ticks: int, n: int = 2400) -> dict[str, np.ndarray]:
    t = np.arange(n) * DT
    base = np.zeros((n + 64, 7))
    tt = np.arange(n + 64) * DT
    for j in range(7):
        base[:, j] = 0.4 * np.sin(2 * np.pi * 0.7 * tt + j) + 0.2 * np.sin(2 * np.pi * 1.3 * tt + 2 * j)
    off = 32
    cmd_arm = base[off : off + n]
    ref_arm = base[off - lead_ticks : off - lead_ticks + n]
    meas_arm = base[off - delay_ticks : off - delay_ticks + n]

    cmd21 = np.zeros((n, 21))
    ref21 = np.zeros((n, 21))
    meas24 = np.zeros((n, 24))
    cmd21[:, 12:19] = cmd_arm
    ref21[:, 12:19] = ref_arm
    meas24[:, 15:22] = meas_arm
    grip = np.where((t % 4.0) < 2.0, -1.0, 0.0)
    cmd21[:, 20] = grip
    ref21[:, 20] = grip
    return {
        "monotonic_s": t,
        "measured_state24": meas24,
        "reference21": ref21,
        "command21": cmd21,
        "plan_id": (t // 0.2).astype(np.int32),
        "stage": np.full(n, STAGE_STEP, dtype=np.int8),
        "stage_names": STAGE_NAMES,
        "tracker_status": np.ones(n, dtype=np.int8),
        "servo_tick_ms": np.full(n, 2.0, dtype=np.float32),
    }


# 用例: 纯延时 9 拍 → path_lag 恢复 9 拍、距离≈0
def test_path_lag_recovers_pure_delay() -> None:
    t = np.arange(500) * DT
    x = np.stack([np.sin(2 * np.pi * 0.8 * t), np.cos(2 * np.pi * 0.5 * t)], 1)
    y = np.roll(x, 9, axis=0)
    lag, dist, _ = path_lag(x, y, 40)
    assert np.median(lag[50:]) == pytest.approx(9.0, abs=0.3)
    assert np.median(dist[50:]) < 1e-3


# 用例: cmd→meas 90ms、ref 落后 cmd 20ms → 路径时滞 / 互相关 / 相位延迟均对得上
def test_control_quality_lags_match_ground_truth() -> None:
    cq = control_quality_metrics(_synthetic_ctl(delay_ticks=9, lead_ticks=2), ee=False)
    lag = cq["lag"]
    assert lag["path_cmd_ms"]["p50"] == pytest.approx(90.0, abs=5.0)
    assert lag["path_ref_ms"]["p50"] == pytest.approx(70.0, abs=5.0)
    assert lag["xcorr"]["cmd_ms"]["p50"] == pytest.approx(90.0, abs=5.0)
    assert lag["xcorr"]["ref_ms"]["p50"] == pytest.approx(70.0, abs=5.0)
    assert lag["xcorr"]["cmd_ms"]["rho_p50"] > 0.95
    assert cq["freq"]["arm"] == "R"
    assert cq["freq"]["cmd"]["phase_delay_1hz_ms"] == pytest.approx(90.0, abs=10.0)
    assert cq["gain"]["speed_gain"]["p50"] == pytest.approx(1.0, abs=0.05)
    assert cq["error"]["contour_cmd_mrad"]["p95"] < 5.0
    assert cq["error"]["aligned_resid_ref_mrad"]["p95"] < 0.5 * cq["error"]["err_ref_l2_mrad"]["p95"]
    head = control_headline(cq)
    assert head["lag_cmd_p50_ms"] == lag["path_cmd_ms"]["p50"]
    assert "lag_ref_p50_ms@0.5-1" in head


# 用例: 延时越大，时滞越大（单调）
def test_control_quality_lag_monotonic_in_delay() -> None:
    a = control_quality_metrics(_synthetic_ctl(delay_ticks=5, lead_ticks=0), ee=False)
    b = control_quality_metrics(_synthetic_ctl(delay_ticks=12, lead_ticks=0), ee=False)
    assert a["lag"]["path_cmd_ms"]["p50"] < b["lag"]["path_cmd_ms"]["p50"]


# 用例: 新增 aux 列存在时进 timing / effort
def test_control_quality_reads_aux_columns() -> None:
    ctl = _synthetic_ctl(delay_ticks=9, lead_ticks=2)
    n = ctl["monotonic_s"].shape[0]
    ctl["state_read_ms"] = np.full(n, 0.8, dtype=np.float32)
    ctl["send_ms"] = np.full(n, 0.3, dtype=np.float32)
    ctl["sat_excess_rad"] = np.zeros(n, dtype=np.float32)
    cq = control_quality_metrics(ctl, ee=False)
    assert cq["timing"]["state_read_p50_ms"] == pytest.approx(0.8, abs=1e-3)
    assert cq["timing"]["send_p99_ms"] == pytest.approx(0.3, abs=1e-3)
    assert cq["effort"]["sat_frac"] == 0.0


# 用例: SPARC 平滑的单峰速度比叠加高频的更接近 0
def test_sparc_prefers_smooth_profile() -> None:
    t = np.linspace(0, 1, 100)
    smooth = np.sin(np.pi * t) ** 2
    rough = smooth + 0.2 * np.sin(2 * np.pi * 8 * t) ** 2
    assert sparc(smooth, 100.0) > sparc(rough, 100.0)


# 用例: 夹爪周期切 seek/grasp/carry/place，各阶段时滞/轮廓可统计；高速子集存在
def test_control_quality_task_phases() -> None:
    from fastervla.deploy.analysis.control_quality import _phase_labels, PHASE_NAMES

    ctl = _synthetic_ctl(delay_ticks=9, lead_ticks=2)
    g = ctl["command21"][:, 20]
    t = ctl["monotonic_s"]
    phase = _phase_labels(g, t)
    for name in PHASE_NAMES:
        assert np.any(phase == name), name
    cq = control_quality_metrics(ctl, ee=False)
    ph = cq["phases"]["joint"]
    assert ph["carry"]["n"] > 0 and ph["seek"]["n"] > 0
    assert ph["grasp"]["lag_ms"]["p50"] == pytest.approx(70.0, abs=15.0)
    head = control_headline(cq)
    assert head["phase_carry_lag_p50_ms"] is not None


# 用例: URDF 限速利用率 = max_i(|q̇|/vlim)；超限 → util>1、gap<0
def test_joint_vel_limit_util_and_phase_gap() -> None:
    from fastervla.deploy.analysis.control_quality import (
        G2_ARM_R_VEL_LIMIT_RAD_S,
        VEL_LIMIT_UTIL_TARGET,
        joint_vel_limit_util,
    )

    lim = np.asarray(G2_ARM_R_VEL_LIMIT_RAD_S, dtype=np.float64)
    half = np.tile(0.5 * lim, (8, 1))
    u = joint_vel_limit_util(half)
    assert u.shape == (8,)
    assert float(np.max(u)) == pytest.approx(0.5, abs=1e-9)
    over = np.tile(1.2 * lim, (4, 1))
    u2 = joint_vel_limit_util(over)
    assert float(np.min(u2)) == pytest.approx(1.2, abs=1e-9)

    cq = control_quality_metrics(_synthetic_ctl(delay_ticks=9, lead_ticks=2), ee=False)
    carry = cq["phases"]["joint"]["carry"]
    assert carry["vel_limit_util_p50"] is not None
    assert carry["vel_limit_util_p95"] is not None
    assert carry["vel_limit_gap_p95"] == pytest.approx(1.0 - carry["vel_limit_util_p95"], abs=1e-9)
    assert cq["phases"]["definition"]["vel_limit_util_target"] == VEL_LIMIT_UTIL_TARGET
    head = control_headline(cq)
    assert head["phase_carry_vel_util_p50"] == carry["vel_limit_util_p50"]
    assert head["phase_carry_vel_util_p95"] == carry["vel_limit_util_p95"]
    assert head["phase_carry_vel_gap_p50"] == pytest.approx(1.0 - carry["vel_limit_util_p50"], abs=1e-9)
    assert head["vel_limit_util_target"] == VEL_LIMIT_UTIL_TARGET
