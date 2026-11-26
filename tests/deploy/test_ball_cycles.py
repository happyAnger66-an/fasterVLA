"""模块: fastervla.deploy.analysis.ball_cycles
功能: 夹→放（搬运）/ 松→夹（找球）周期 + 找球停/慢/快拆分。
"""

from __future__ import annotations

import numpy as np

from fastervla.deploy.analysis.ball_cycles import (
    GRIP_CLOSED,
    ball_events,
    merge_short_release_cycles,
    seek_motion_share,
    summarize_times,
)


def test_ball_events_detects_two_balls() -> None:
    fps = 30.0
    g = np.full(300, -0.5, dtype=np.float64)
    # ball0: close 30-60, ball1: close 120-160
    g[30:60] = 0.0
    g[120:160] = 0.0
    balls = ball_events(g, fps, merge_gap_s=0.4, min_close_s=0.5, regrasp_s=1.0)
    assert len(balls) == 2
    assert balls[0]["n_attempts"] == 1
    assert abs(balls[0]["grasp_to_place_s"] - 1.0) < 1e-6


def test_seek_motion_share_per_release_to_grasp() -> None:
    t = np.arange(200, dtype=np.float64) / 30.0
    balls = [
        {"first_grasp_i": 10, "final_release_i": 40, "n_attempts": 1,
         "grasp_to_place_s": 1.0, "hold_s": 1.0, "gap_s": 0.0},
        {"first_grasp_i": 100, "final_release_i": 140, "n_attempts": 1,
         "grasp_to_place_s": 1.33, "hold_s": 1.33, "gap_s": 0.0},
    ]
    v = np.full(200, 0.2, dtype=np.float64)  # fast by default
    v[40:70] = 0.01  # dwell after release
    v[70:100] = 0.08  # slow approach
    out = seek_motion_share(t=t, balls=balls, ee_speed_mps=v)
    assert out["n"] == 1
    row = out["per_ball"][0]
    assert row["dwell_frac"] > 0.4
    assert row["slow_frac"] > 0.4
    assert out["dwell_frac_p50"] == row["dwell_frac"]


def test_summarize_times_empty() -> None:
    assert summarize_times([]) == {"n": 0}
    s = summarize_times([1.0, 2.0, 3.0])
    assert s["n"] == 3 and s["p50"] == 2.0


def test_grip_closed_threshold() -> None:
    assert GRIP_CLOSED == -0.39


def test_merge_short_release_cycles_as_retries() -> None:
    """松→松 < 3s is a regrasp, not a new ball period."""
    fps = 30.0
    # Three "balls": gaps between releases = 5s, 2s → merge last two.
    balls = [
        {
            "first_grasp_i": 0,
            "final_release_i": 30,  # t=1s
            "n_attempts": 1,
            "grasp_to_place_s": 1.0,
            "hold_s": 1.0,
            "gap_s": 0.0,
        },
        {
            "first_grasp_i": 120,  # after 3s seek
            "final_release_i": 180,  # release at 6s; r2r from prev = 5s
            "n_attempts": 1,
            "grasp_to_place_s": 2.0,
            "hold_s": 2.0,
            "gap_s": 0.0,
        },
        {
            "first_grasp_i": 200,  # quick regrasp
            "final_release_i": 240,  # release at 8s; r2r from prev = 2s < 3
            "n_attempts": 1,
            "grasp_to_place_s": 1.33,
            "hold_s": 1.33,
            "gap_s": 0.0,
        },
    ]
    out, n_merged = merge_short_release_cycles(balls, fps, min_release_to_release_s=3.0)
    assert n_merged == 1
    assert len(out) == 2
    assert out[1]["n_attempts"] == 2
    assert out[1]["final_release_i"] == 240
    assert out[1]["short_cycle_merged"] is True


def test_analyze_trace_ending_while_closed() -> None:
    """Grip held through EOF: exclusive release index == n must not IndexError."""
    from fastervla.deploy.analysis.ball_cycles import analyze_ball_cycles

    fps = 100.0
    n = 200
    t = np.arange(n, dtype=np.float64) / fps
    cmd = np.zeros((n, 21), dtype=np.float64)
    cmd[:, 20] = -0.5
    cmd[50:, 20] = 0.0  # still closed at last sample
    ctl = {
        "monotonic_s": t,
        "command21": cmd,
        "measured_state24": np.zeros((n, 24)),
        "reference21": np.zeros((n, 21)),
    }
    out = analyze_ball_cycles(ctl, ee_speed=False)
    assert out["n_balls"] == 1
    assert out["per_ball"][0]["t_release"] <= float(t[-1] - t[0]) + 1e-9


def test_analyze_excludes_short_r2r_from_summary() -> None:
    from fastervla.deploy.analysis.ball_cycles import analyze_ball_cycles

    fps = 30.0
    n = 400
    t = np.arange(n, dtype=np.float64) / fps
    cmd = np.zeros((n, 21), dtype=np.float64)
    cmd[:, 20] = -0.5
    # ball0: 30-90, ball1: 180-240 (r2r≈5s), ball2: 270-330 (r2r≈3s? 330-240=90 → 3.0s)
    # Use r2r=2.0s for ball2: release1 at 240, release2 at 300 → 2s
    cmd[30:90, 20] = 0.0
    cmd[180:240, 20] = 0.0
    # regrasp >1s after release@240 (not caught by regrasp_s), but r2r≈2.67s < 3s
    cmd[280:320, 20] = 0.0
    ctl = {
        "monotonic_s": t,
        "command21": cmd,
        "measured_state24": np.zeros((n, 24)),
        "reference21": np.zeros((n, 21)),
    }
    out = analyze_ball_cycles(ctl, ee_speed=False, min_release_to_release_s=3.0)
    assert out["n_raw_balls"] == 3
    assert out["n_short_cycles_merged"] == 1
    assert out["n_balls"] == 2
    r2r = out["release_to_release"]
    assert r2r["n"] == 1
    assert r2r["min"] >= 3.0 - 1e-6
    assert out["attempts"]["exact"].get("2") == 1
