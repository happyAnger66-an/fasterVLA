"""模块: fastervla.deploy.analysis
功能: 离线分析 robot-live trace（hf3/hf7、跟踪、r2r）；本仓自包含，不依赖 tools/jitter。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fastervla.deploy.analysis import analyze_results_dir, analyze_run
from fastervla.deploy.analysis.spectrum import band_rms
from fastervla.deploy.client.recorder import (
    STAGE_STEP,
    TraceRecorder,
    make_run_dir,
)


def _synth_run(tmp_path: Path, *, n_ticks: int = 150, n_chunks: int = 6) -> Path:
    run = make_run_dir(tmp_path)
    rec = TraceRecorder(run, capacity_ticks=n_ticks + 8, horizon=30)
    rec.set_experiment(policy_fps=30.0, kappa=1.0, tag="analysis-unit")
    t0 = 100.0
    for i in range(n_ticks):
        # Slow sine so band energy is mostly low; still finite hf3.
        phase = 0.02 * i
        meas = np.zeros(24, dtype=np.float64)
        meas[8:22] = 0.05 * np.sin(phase + np.arange(14) * 0.01)
        cmd = np.zeros(21, dtype=np.float64)
        cmd[5:19] = meas[8:22] + 0.001
        rec.record_tick(
            monotonic_s=t0 + 0.01 * i,
            measured24=meas,
            reference21=cmd,
            command21=cmd,
            progress=float(i) * 0.05,
            plan_id=1 + i // 30,
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
            step_dt=np.full(30, 1.0 / 30.0),
            merge_accepted=True,
        )
    rec.close()
    return run


def test_band_rms_silence_near_zero() -> None:
    x = np.zeros((64, 3), dtype=np.float64)
    rms = band_rms(x, fs=30.0, lo=3.0, hi=7.5)
    assert rms.shape == (3,)
    assert float(np.max(rms)) < 1e-9


def test_analyze_run_headline_fields(tmp_path: Path) -> None:
    run = _synth_run(tmp_path)
    summary = analyze_run(run)
    assert summary["n_chunks"] == 6
    assert summary["n_control_ticks"] == 150
    assert summary["headline"]["hf3_mrad"] is not None
    assert summary["headline"]["r2r_p95_ms"] == 80.0
    assert summary["tracking"]["cmd_track_p95_rad"] is not None
    assert summary["chunk_filtered"]["hf3_mrad_mean"] >= 0.0
    assert summary["exec"]["exec_seconds"] > 0.0


def test_analyze_results_dir(tmp_path: Path) -> None:
    _synth_run(tmp_path)
    rows = analyze_results_dir(tmp_path)
    assert len(rows) == 1
    assert "error" not in rows[0]
    assert rows[0]["experiment"]["tag"] == "analysis-unit"


def test_analyze_missing_raises(tmp_path: Path) -> None:
    empty = tmp_path / "robot-live-empty"
    empty.mkdir()
    try:
        analyze_run(empty)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
