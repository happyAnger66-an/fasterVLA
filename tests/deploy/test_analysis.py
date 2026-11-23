"""模块: fastervla.deploy.analysis
功能: 离线分析 robot-live trace（四类指标 + legacy hf3/跟踪/r2r）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fastervla.deploy.analysis import analyze_results_dir, analyze_run
from fastervla.deploy.analysis.ee_kinematics import load_urdf_fk
from fastervla.deploy.analysis.metrics import (
    merge_sway_metrics,
    progress_wall_fps,
    wide_row,
)
from fastervla.deploy.analysis.spectrum import band_rms
from fastervla.deploy.client.recorder import (
    STAGE_STEP,
    TraceRecorder,
    make_run_dir,
)


def _synth_run(tmp_path: Path, *, n_ticks: int = 150, n_chunks: int = 6) -> Path:
    run = make_run_dir(tmp_path)
    rec = TraceRecorder(run, capacity_ticks=n_ticks + 8, horizon=30)
    rec.set_experiment(
        policy_fps=30.0,
        kappa=1.5,
        tag="analysis-unit",
        tracker="lead",
        prefix_frames=4,
        buffer_frames=4,
        arm_filter="zero_phase_dct",
        arm_interp="pchip",
    )
    t0 = 100.0
    # Two plan segments: advance then reindex-like drop on plan_id change.
    for i in range(n_ticks):
        phase = 0.02 * i
        meas = np.zeros(24, dtype=np.float64)
        # small EE-ish motion on left arm joints
        meas[8:15] = 0.05 * np.sin(phase + np.arange(7) * 0.01)
        meas[15:22] = 0.04 * np.cos(phase + np.arange(7) * 0.01)
        cmd = np.zeros(21, dtype=np.float64)
        cmd[5:19] = meas[8:22] + 0.001
        ref = cmd.copy()
        ref[5:19] = meas[8:22]  # lead-ish: cmd ahead of ref slightly already via +0.001
        plan = 1 + i // 50
        # Within each plan, progress climbs; across plan change it "resets" toward d_actual.
        local = float(i % 50) * 0.1
        if i > 0 and i % 50 == 0:
            local = 2.0
        rec.record_tick(
            monotonic_s=t0 + 0.01 * i,
            measured24=meas,
            reference21=ref,
            command21=cmd,
            progress=local,
            plan_id=plan,
            stage=STAGE_STEP,
            servo_tick_ms=1.0,
        )
    for c in range(n_chunks):
        acts = np.zeros((30, 24), dtype=np.float32)
        for h in range(30):
            acts[h, 8:22] = 0.05 * np.sin(0.1 * (c * 30 + h) + np.arange(14) * 0.01)
        origin = t0 + c * 0.5
        rec.record_chunk(
            actions=acts,
            actions_raw=acts,
            requested_monotonic_s=origin,
            ready_monotonic_s=origin + 0.08,
            accepted_monotonic_s=origin + 0.09,
            action_origin_monotonic_s=origin,
            handoff_at_monotonic_s=origin + 0.45,
            observation_monotonic_s=origin - 0.02,
            observation_age_ms=20.0,
            d_hat=3,
            d_actual=2.0,
            plan_id=c + 1,
            rtc_applied=True,
            rtc_d=3,
            request_to_ready_ms=80.0,
            previous_command_difference=0.01 * (c + 1),
            step_dt=np.full(30, 1.0 / (1.5 * 30.0)),
            merge_accepted=True,
        )
    rec.close()
    return run


def test_band_rms_silence_near_zero() -> None:
    x = np.zeros((64, 3), dtype=np.float64)
    rms = band_rms(x, fs=30.0, lo=3.0, hi=7.5)
    assert rms.shape == (3,)
    assert float(np.max(rms)) < 1e-9


def test_progress_wall_fps_sums_per_plan() -> None:
    # plan0: 0→5 over 1s; plan1: 0→5 over 1s → 10 steps / 2s = 5 fps
    t = np.linspace(0.0, 2.0, 21)
    progress = np.concatenate(
        [np.linspace(0.0, 5.0, 11), np.linspace(0.0, 5.0, 10)]
    )
    plan_id = np.concatenate([np.zeros(11, dtype=np.int32), np.ones(10, dtype=np.int32)])
    out = progress_wall_fps(t, progress, plan_id)
    assert out["progress_advance_total"] == 10.0
    assert abs(out["progress_wall_fps"] - 5.0) < 1e-6


def test_merge_sway_has_jumps_not_prog_jump() -> None:
    t = np.arange(0, 1.0, 0.01)
    plan = np.zeros(t.size, dtype=np.int32)
    plan[50:] = 1
    ref = np.zeros((t.size, 21))
    ref[:, 5:19] = 0.01
    ref[50:, 5:19] = 0.05  # jump at switch
    cmd = ref.copy()
    out = merge_sway_metrics(t, plan, reference21=ref, command21=cmd)
    assert out["n_merges"] == 1
    assert "prog_jump" not in out
    assert out["ref_jump_sw_p95_mrad"] is not None
    assert out["ref_jump_sw_p95_mrad"] > 30.0  # ~40 mrad


def test_analyze_run_four_categories(tmp_path: Path) -> None:
    run = _synth_run(tmp_path)
    summary = analyze_run(run, ee_stride=10)
    assert summary["n_chunks"] == 6
    assert summary["n_control_ticks"] == 150
    assert "speed" in summary and "sway" in summary
    assert "stability" in summary and "accuracy" in summary
    assert "prog_jump" not in summary["headline"]
    assert "prog_jump" not in summary["wide"]
    assert summary["speed"]["progress_wall_fps"] is not None
    assert summary["speed"]["progress_wall_fps"] > 0.0
    assert summary["sway"]["merge_hz"] is not None
    assert summary["sway"]["merge_hz"] > 0.0
    assert summary["accuracy"]["ref_track_p95_rad"] is not None
    assert summary["headline"]["merge_hz"] == summary["sway"]["merge_hz"]
    assert summary["headline"]["progress_wall_fps"] == summary["speed"]["progress_wall_fps"]
    assert summary["wide"]["kappa"] == 1.5
    # step_dt implies ~45 fps at κ=1.5 * 30
    assert summary["speed"]["implied_fps_from_dt_p50"] is not None
    assert abs(summary["speed"]["implied_fps_from_dt_p50"] - 45.0) < 1e-6


def test_analyze_run_ee_speed_when_urdf_available(tmp_path: Path) -> None:
    if load_urdf_fk() is None:
        return  # skip silently when URDF not on this machine
    run = _synth_run(tmp_path, n_ticks=80, n_chunks=3)
    summary = analyze_run(run, ee_stride=4)
    assert summary["speed"].get("ee_fk_available") is True
    assert summary["speed"].get("ee_speed_p50_mps") is not None
    assert summary["headline"].get("ee_speed_p50_mps") is not None


def test_wide_row_excludes_prog_jump(tmp_path: Path) -> None:
    run = _synth_run(tmp_path, n_ticks=60, n_chunks=2)
    summary = analyze_run(run)
    row = wide_row(summary)
    assert "prog_jump_p50" not in row
    assert "net_per_cycle_p50" not in row
    assert "progress_wall_fps" in row
    assert "ref_track_p95_mrad" in row


def test_analyze_results_dir_wide(tmp_path: Path) -> None:
    _synth_run(tmp_path)
    rows = analyze_results_dir(tmp_path, wide_only=True)
    assert len(rows) == 1
    assert "error" not in rows[0]
    assert rows[0]["progress_wall_fps"] is not None


def test_analyze_missing_raises(tmp_path: Path) -> None:
    empty = tmp_path / "robot-live-empty"
    empty.mkdir()
    try:
        analyze_run(empty)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
