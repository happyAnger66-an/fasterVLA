"""Tracking error summaries from control_trace."""

from __future__ import annotations

from typing import Any

import numpy as np

from fastervla.deploy.analysis.jitter import ARM_ACT, ARM_CMD


def tracking_metrics(ctl: dict[str, np.ndarray]) -> dict[str, Any]:
    """|cmd−meas| and |ref−meas| on arm dims (command layout)."""
    if "command21" not in ctl or "measured_state24" not in ctl:
        return {}
    cmd = np.asarray(ctl["command21"], dtype=np.float64)
    meas = np.asarray(ctl["measured_state24"], dtype=np.float64)
    if cmd.shape[0] < 1:
        return {"n": 0}
    meas_arm = meas[:, ARM_ACT]
    cmd_arm = cmd[:, ARM_CMD]
    err_cmd = np.max(np.abs(cmd_arm - meas_arm), axis=1)
    out: dict[str, Any] = {
        "n": int(err_cmd.shape[0]),
        "cmd_track_p50_rad": float(np.percentile(err_cmd, 50)),
        "cmd_track_p95_rad": float(np.percentile(err_cmd, 95)),
        "cmd_track_max_rad": float(np.max(err_cmd)),
    }
    if "reference21" in ctl:
        ref_arm = np.asarray(ctl["reference21"], dtype=np.float64)[:, ARM_CMD]
        err_ref = np.max(np.abs(ref_arm - meas_arm), axis=1)
        out["ref_track_p50_rad"] = float(np.percentile(err_ref, 50))
        out["ref_track_p95_rad"] = float(np.percentile(err_ref, 95))
        out["ref_track_max_rad"] = float(np.max(err_ref))
    if "stage" in ctl and "stage_names" in ctl:
        names = [str(x) for x in np.asarray(ctl["stage_names"]).tolist()]
        stage = np.asarray(ctl["stage"])
        if "buffer_empty_hold" in names:
            empty_id = names.index("buffer_empty_hold")
            out["empty_frac"] = float(np.mean(stage == empty_id))
        if "step" in names:
            step_id = names.index("step")
            step_mask = stage == step_id
            if np.any(step_mask):
                out["cmd_track_p95_step_rad"] = float(
                    np.percentile(err_cmd[step_mask], 95)
                )
    return out
