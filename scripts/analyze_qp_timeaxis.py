#!/usr/bin/env python3
"""QP time-axis diagnostics on robot-live prediction traces.

Per accepted chunk: where does TimeAxisQP put its rates (1/step_dt) relative
to joint step size |dp| and the gripper transition? And how consistent are the
wall-times assigned to the *same* future waypoints across consecutive chunks?

  python3 scripts/analyze_qp_timeaxis.py --runs results/robot-live-* [--exec 8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARM = slice(8, 22)
GRIP_R = 23
GRIP_CLOSED = -0.39


def _q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def analyze(run_dir: Path, exec_frames: int, fps: float = 30.0) -> dict:
    z = np.load(run_dir / "prediction_trace.npz", allow_pickle=True)
    act = np.asarray(z["actions"], dtype=np.float64)
    dt = np.asarray(z["step_dt"], dtype=np.float64)
    st = np.asarray(z["inference_state"], dtype=np.float64)
    acc = np.asarray(z["merge_accepted"]).astype(bool)
    status = [str(s) for s in z["retime_status"]]
    n, h, _ = act.shape
    rate = 1.0 / dt
    s_ref_all = float(np.median(rate))

    step = np.empty((n, h))
    for i in range(n):
        pts = np.vstack([st[i, ARM], act[i, :, ARM]])
        step[i] = np.linalg.norm(np.diff(pts, axis=0), axis=1)

    ex = slice(0, exec_frames)
    r_ex = rate[:, ex].ravel()
    s_ex = step[:, ex].ravel()
    jv_ex = (step[:, ex] * rate[:, ex]).ravel()  # joint-space speed rad/s (L2)

    # Gripper-transition frames: |Δgrip| big or grip near closed threshold.
    g = act[:, :, GRIP_R]
    near_grip = np.zeros_like(g, dtype=bool)
    dg = np.abs(np.diff(np.concatenate([g[:, :1], g], axis=1), axis=1))
    near_grip |= dg > 0.05
    near_grip |= np.abs(g - GRIP_CLOSED) < 0.15
    ng = near_grip[:, ex].ravel()

    # Size quartiles of |dp| in executed window: rate per bucket.
    qs = np.quantile(s_ex, [0.25, 0.5, 0.75])
    buckets = np.digitize(s_ex, qs)
    by_bucket = {
        f"Q{b+1}": {
            "dp_mrad_p50": 1e3 * _q(s_ex[buckets == b], 50),
            "rate_over_ref_p50": _q(r_ex[buckets == b], 50) / s_ref_all,
        }
        for b in range(4)
    }

    # Handoff time-axis jump: rate the old plan had at the new plan's origin
    # vs the new plan's first-window rate (same waypoints, re-solved timing).
    origin = np.asarray(z["action_origin_monotonic_s"], dtype=np.float64)
    ft = np.asarray(z["frame_times_s"], dtype=np.float64)
    idx = np.flatnonzero(acc)
    rate_jump = []
    for a, b in zip(idx[:-1], idx[1:]):
        k = int(np.searchsorted(origin[a] + ft[a, 1:], origin[b]))
        if k >= h:
            continue
        old = rate[a, k : min(k + 3, h)].mean()
        new = rate[b, :3].mean()
        rate_jump.append(abs(new - old) / s_ref_all)

    corr = float(np.corrcoef(np.log(s_ex + 1e-6), np.log(r_ex))[0, 1])
    return {
        "run": run_dir.name.replace("robot-live-", ""),
        "n_chunks": int(n),
        "status": {s: status.count(s) for s in sorted(set(status))},
        "s_ref_median": s_ref_all,
        "exec_rate_over_ref": {
            "p05": _q(r_ex, 5) / s_ref_all,
            "p50": _q(r_ex, 50) / s_ref_all,
            "p95": _q(r_ex, 95) / s_ref_all,
            "max": float(r_ex.max()) / s_ref_all,
        },
        "frac_at_dt_min": float(np.mean(dt[:, ex] <= 0.0151)),
        "frac_at_dt_max": float(np.mean(dt[:, ex] >= 0.0799)),
        "corr_log_dp_vs_log_rate": corr,
        "by_dp_quartile": by_bucket,
        "near_grip_rate_over_ref_p50": _q(r_ex[ng], 50) / s_ref_all if ng.any() else None,
        "far_grip_rate_over_ref_p50": _q(r_ex[~ng], 50) / s_ref_all,
        "joint_speed_rad_s": {"p50": _q(jv_ex, 50), "p95": _q(jv_ex, 95)},
        "joint_speed_cv": float(np.std(jv_ex) / max(np.mean(jv_ex), 1e-9)),
        "handoff_rate_jump_over_ref": {
            "p50": _q(rate_jump, 50),
            "p95": _q(rate_jump, 95),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--exec", type=int, default=8, dest="exec_frames")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = [analyze(Path(r), a.exec_frames) for r in a.runs]
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
