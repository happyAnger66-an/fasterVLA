"""Four-category offline metrics for robot-live runs (no prog_jump).

Categories
----------
1. speed     — kappa / progress_wall_fps / cmd speed / EE / step_dt
2. sway      — merge rate / spatial jumps at switch / hf3 / rev
3. stability — empty / starved / rem / r2r / duty / servo
4. accuracy  — ref/cmd track / lead / frac>100mrad
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fastervla.deploy.analysis.ee_kinematics import ee_speed_metrics
from fastervla.deploy.analysis.jitter import ARM_ACT, ARM_CMD

_TRACK_WARN_RAD = 0.100


def _finite(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)[np.isfinite(a)]


def _pct(a: np.ndarray, q: float) -> float | None:
    v = _finite(a)
    if v.size == 0:
        return None
    return float(np.percentile(v, q))


def _switch_indices(plan_id: np.ndarray) -> np.ndarray:
    p = np.asarray(plan_id).reshape(-1)
    if p.size < 2:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.diff(p.astype(np.int64)) != 0) + 1


def progress_wall_fps(
    t: np.ndarray,
    progress: np.ndarray,
    plan_id: np.ndarray,
) -> dict[str, Any]:
    """Net model-step rate on wall clock, summing advance within each plan_id."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    prog = np.asarray(progress, dtype=np.float64).reshape(-1)
    pid = np.asarray(plan_id).reshape(-1)
    if t.size < 2 or prog.size != t.size or pid.size != t.size:
        return {"progress_wall_fps": None, "progress_advance_total": None}
    dur = float(t[-1] - t[0])
    if dur <= 1e-9:
        return {"progress_wall_fps": None, "progress_advance_total": 0.0, "dur_s": dur}

    advance = 0.0
    # Segment by plan_id runs.
    cuts = np.concatenate([[0], _switch_indices(pid), [pid.size]])
    for s, e in zip(cuts[:-1], cuts[1:]):
        if e - s < 1:
            continue
        seg = prog[s:e]
        # Prefer endpoint delta (monotonic within plan); fall back to positive diffs.
        delta = float(seg[-1] - seg[0])
        if delta < -1e-6:
            d = np.diff(seg)
            delta = float(np.sum(d[d > 0.0]))
        else:
            delta = max(0.0, delta)
        advance += delta
    return {
        "progress_wall_fps": float(advance / dur),
        "progress_advance_total": float(advance),
        "dur_s": dur,
    }


def merge_sway_metrics(
    t: np.ndarray,
    plan_id: np.ndarray,
    *,
    reference21: np.ndarray | None = None,
    command21: np.ndarray | None = None,
) -> dict[str, Any]:
    """Merge density and spatial jumps at plan switches (not progress reindex)."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    pid = np.asarray(plan_id).reshape(-1)
    if t.size < 2:
        return {}
    dur = float(t[-1] - t[0])
    sw = _switch_indices(pid)
    out: dict[str, Any] = {
        "n_merges": int(sw.size),
        "merge_hz": float(sw.size / dur) if dur > 1e-9 else None,
        "dur_s": dur,
    }
    if sw.size >= 2:
        gaps_ms = np.diff(t[sw]) * 1e3
        out["sw_ms_p50"] = _pct(gaps_ms, 50)
        out["sw_ms_p95"] = _pct(gaps_ms, 95)
    elif sw.size == 1 and dur > 1e-9:
        out["sw_ms_p50"] = float(dur * 1e3)
        out["sw_ms_p95"] = float(dur * 1e3)

    def _jump_p95(arr: np.ndarray | None, key: str) -> None:
        if arr is None or sw.size == 0:
            return
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] != t.size:
            return
        arm = a[:, ARM_CMD]
        # Jump from tick before switch → switch tick.
        prev = sw - 1
        ok = prev >= 0
        if not np.any(ok):
            return
        delta = np.max(np.abs(arm[sw[ok]] - arm[prev[ok]]), axis=1)
        out[key] = _pct(delta, 95)
        out[key.replace("p95", "p50")] = _pct(delta, 50)

    _jump_p95(reference21, "ref_jump_sw_p95_rad")
    _jump_p95(command21, "cmd_jump_sw_p95_rad")
    # Convenience mrad mirrors for tables.
    for src, dst in (
        ("ref_jump_sw_p95_rad", "ref_jump_sw_p95_mrad"),
        ("cmd_jump_sw_p95_rad", "cmd_jump_sw_p95_mrad"),
        ("ref_jump_sw_p50_rad", "ref_jump_sw_p50_mrad"),
        ("cmd_jump_sw_p50_rad", "cmd_jump_sw_p50_mrad"),
    ):
        if src in out and out[src] is not None:
            out[dst] = float(out[src]) * 1e3
    return out


def cmd_speed_metrics(
    t: np.ndarray,
    command21: np.ndarray,
    *,
    stage: np.ndarray | None = None,
    stage_names: list[str] | None = None,
) -> dict[str, Any]:
    """Peak/mean arm command speed (rad/s) on step segments."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    cmd = np.asarray(command21, dtype=np.float64)
    if t.size < 2 or cmd.shape[0] != t.size:
        return {}
    arm = cmd[:, ARM_CMD]
    dt = np.diff(t)
    ok = dt > 1e-6
    if not np.any(ok):
        return {}
    spd = np.max(np.abs(np.diff(arm, axis=0)[ok] / dt[ok, None]), axis=1)
    if stage is not None and stage_names and "step" in stage_names:
        st = np.asarray(stage).reshape(-1)
        step_id = stage_names.index("step")
        # speed sample i corresponds to interval between i and i+1
        mask = (st[:-1] == step_id)[ok]
        if np.any(mask):
            spd = spd[mask]
    if spd.size == 0:
        return {}
    return {
        "cmd_spd_p50_rad_s": float(np.percentile(spd, 50)),
        "cmd_spd_p95_rad_s": float(np.percentile(spd, 95)),
        "cmd_spd_mean_rad_s": float(np.mean(spd)),
    }


def step_dt_metrics(step_dt: np.ndarray | None) -> dict[str, Any]:
    if step_dt is None:
        return {}
    a = np.asarray(step_dt, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return {}
    return {
        "step_dt_p50_s": float(np.percentile(a, 50)),
        "step_dt_p95_s": float(np.percentile(a, 95)),
        "step_dt_mean_s": float(np.mean(a)),
        "implied_fps_from_dt_p50": float(1.0 / np.percentile(a, 50)),
    }


def remaining_metrics(
    progress: np.ndarray,
    *,
    horizon: int = 30,
) -> dict[str, Any]:
    p = np.asarray(progress, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return {}
    rem = float(horizon) - p
    return {
        "rem_min": float(np.min(rem)),
        "rem_p10": float(np.percentile(rem, 10)),
        "rem_p50": float(np.percentile(rem, 50)),
    }


def accuracy_metrics(ctl: dict[str, np.ndarray]) -> dict[str, Any]:
    """ref/cmd tracking + lead + large-error fraction on arm dims."""
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
        "frac_track_gt_100mrad": float(np.mean(err_cmd > _TRACK_WARN_RAD)),
    }
    if "reference21" in ctl:
        ref_arm = np.asarray(ctl["reference21"], dtype=np.float64)[:, ARM_CMD]
        err_ref = np.max(np.abs(ref_arm - meas_arm), axis=1)
        lead = np.max(np.abs(cmd_arm - ref_arm), axis=1)
        out.update(
            {
                "ref_track_p50_rad": float(np.percentile(err_ref, 50)),
                "ref_track_p95_rad": float(np.percentile(err_ref, 95)),
                "ref_track_max_rad": float(np.max(err_ref)),
                "frac_ref_gt_100mrad": float(np.mean(err_ref > _TRACK_WARN_RAD)),
                "lead_p50_rad": float(np.percentile(lead, 50)),
                "lead_p95_rad": float(np.percentile(lead, 95)),
            }
        )
    # mrad mirrors
    for k in list(out):
        if k.endswith("_rad") and isinstance(out[k], float):
            out[k.replace("_rad", "_mrad")] = float(out[k]) * 1e3
    return out


def stability_metrics(
    *,
    report: dict[str, Any],
    ctl: dict[str, np.ndarray],
    r2r_p50_ms: float | None,
    r2r_p95_ms: float | None,
    n_chunks: int,
    horizon: int = 30,
) -> dict[str, Any]:
    t = np.asarray(ctl.get("monotonic_s", []), dtype=np.float64)
    dur = float(t[-1] - t[0]) if t.size >= 2 else 0.0
    empty = report.get("buffer_empty_cycles")
    starved = report.get("starved_events")
    out: dict[str, Any] = {
        "empty": int(empty) if empty is not None else None,
        "starved": int(starved) if starved is not None else None,
        "r2r_p50_ms": r2r_p50_ms,
        "r2r_p95_ms": r2r_p95_ms,
        "servo_tick_p95_ms": report.get("servo_tick_p95_ms"),
        "tracking_warning_cycles": report.get("tracking_warning_cycles"),
    }
    if "progress" in ctl:
        out.update(remaining_metrics(ctl["progress"], horizon=horizon))
    if "stage" in ctl and "stage_names" in ctl:
        names = [str(x) for x in np.asarray(ctl["stage_names"]).tolist()]
        stage = np.asarray(ctl["stage"])
        if "buffer_empty_hold" in names:
            out["empty_frac"] = float(np.mean(stage == names.index("buffer_empty_hold")))
    if dur > 1e-9 and r2r_p50_ms is not None and n_chunks > 0:
        # Approximate infer wall duty: n_chunks * r2r_p50 / duration.
        out["infer_duty"] = float(n_chunks * (float(r2r_p50_ms) * 1e-3) / dur)
    out["dur_s"] = dur if dur > 0 else None
    return out


def compute_run_metrics(
    *,
    ctl: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    report: dict[str, Any],
    experiment: dict[str, Any],
    chunk_filtered: dict[str, Any],
    exec_m: dict[str, Any],
    r2r_p50_ms: float | None,
    r2r_p95_ms: float | None,
    n_chunks: int,
    control_hz: float = 100.0,
    urdf_hint: str | None = None,
    ee_stride: int = 5,
) -> dict[str, Any]:
    """Build the four-category metrics block for one run."""
    t = np.asarray(ctl["monotonic_s"], dtype=np.float64)
    horizon = int(experiment.get("horizon") or report.get("horizon") or 30)
    kappa = experiment.get("kappa")
    policy_fps = float(experiment.get("policy_fps") or report.get("policy_fps") or 30.0)

    stage = ctl.get("stage")
    stage_names = (
        [str(x) for x in np.asarray(ctl["stage_names"]).tolist()]
        if "stage_names" in ctl
        else None
    )

    speed: dict[str, Any] = {
        "kappa": kappa,
        "policy_fps": policy_fps,
        "kappa_x_policy_fps": (
            float(kappa) * policy_fps if kappa is not None else None
        ),
    }
    if "progress" in ctl and "plan_id" in ctl:
        speed.update(progress_wall_fps(t, ctl["progress"], ctl["plan_id"]))
    if "command21" in ctl:
        speed.update(
            cmd_speed_metrics(
                t, ctl["command21"], stage=stage, stage_names=stage_names
            )
        )
    if "step_dt" in pred:
        # prediction may be (K, H)
        sd = np.asarray(pred["step_dt"], dtype=np.float64)
        speed.update(step_dt_metrics(sd))
    if "measured_state24" in ctl:
        speed.update(
            ee_speed_metrics(
                t,
                ctl["measured_state24"],
                urdf_hint=urdf_hint,
                stride=ee_stride,
            )
        )

    sway: dict[str, Any] = {}
    if "plan_id" in ctl:
        sway.update(
            merge_sway_metrics(
                t,
                ctl["plan_id"],
                reference21=ctl.get("reference21"),
                command21=ctl.get("command21"),
            )
        )
    sway["cmd_hf3_mrad"] = exec_m.get("cmd_hf3_mrad_mean")
    sway["meas_hf3_mrad"] = exec_m.get("meas_hf3_mrad_mean")
    sway["rev_per_s"] = chunk_filtered.get("rev_per_s")
    sway["handoff_jump_p95_mrad"] = chunk_filtered.get("handoff_jump_p95_mrad")

    stability = stability_metrics(
        report=report,
        ctl=ctl,
        r2r_p50_ms=r2r_p50_ms,
        r2r_p95_ms=r2r_p95_ms,
        n_chunks=n_chunks,
        horizon=horizon,
    )

    accuracy = accuracy_metrics(ctl)

    return {
        "speed": speed,
        "sway": sway,
        "stability": stability,
        "accuracy": accuracy,
        "meta": {
            "control_hz": control_hz,
            "horizon": horizon,
            "exclude": ["prog_jump", "net_per_cycle"],
        },
    }


def wide_row(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten analyze_run summary into one dashboard row (no prog_jump)."""
    exp = dict(summary.get("experiment") or {})
    speed = dict(summary.get("speed") or {})
    sway = dict(summary.get("sway") or {})
    stab = dict(summary.get("stability") or {})
    acc = dict(summary.get("accuracy") or {})

    def g(*dicts: dict, key: str):
        for d in dicts:
            if key in d and d[key] is not None:
                return d[key]
        return None

    return {
        "run_id": summary.get("run_id"),
        "tag": exp.get("tag"),
        "kappa": g(exp, speed, key="kappa"),
        "tracker": exp.get("tracker"),
        "prefix": exp.get("prefix_frames") or exp.get("prefix"),
        "buffer": exp.get("buffer_frames") or exp.get("buffer"),
        "filter": exp.get("arm_filter") or exp.get("filter"),
        "interp": exp.get("arm_interp") or exp.get("interp"),
        "retimer": exp.get("retimer"),
        "hold_gripper": exp.get("hold_gripper"),
        # speed
        "progress_wall_fps": speed.get("progress_wall_fps"),
        "cmd_spd_p95": speed.get("cmd_spd_p95_rad_s"),
        "ee_speed_p50": speed.get("ee_speed_p50_mps"),
        "ee_speed_p95": speed.get("ee_speed_p95_mps"),
        "step_dt_p50": speed.get("step_dt_p50_s"),
        "step_dt_p95": speed.get("step_dt_p95_s"),
        # sway
        "merge_hz": sway.get("merge_hz"),
        "sw_ms_p50": sway.get("sw_ms_p50"),
        "ref_jump_sw_p95_mrad": sway.get("ref_jump_sw_p95_mrad"),
        "cmd_jump_sw_p95_mrad": sway.get("cmd_jump_sw_p95_mrad"),
        "cmd_hf3_mrad": sway.get("cmd_hf3_mrad"),
        "rev_per_s": sway.get("rev_per_s"),
        "handoff_jump_p95_mrad": sway.get("handoff_jump_p95_mrad"),
        # stability
        "empty": stab.get("empty"),
        "starved": stab.get("starved"),
        "empty_frac": stab.get("empty_frac"),
        "rem_min": stab.get("rem_min"),
        "r2r_p95_ms": stab.get("r2r_p95_ms"),
        "infer_duty": stab.get("infer_duty"),
        "servo_tick_p95_ms": stab.get("servo_tick_p95_ms"),
        "dur_s": stab.get("dur_s") or speed.get("dur_s"),
        # accuracy
        "ref_track_p50_mrad": acc.get("ref_track_p50_mrad"),
        "ref_track_p95_mrad": acc.get("ref_track_p95_mrad"),
        "cmd_track_p50_mrad": acc.get("cmd_track_p50_mrad"),
        "cmd_track_p95_mrad": acc.get("cmd_track_p95_mrad"),
        "lead_p95_mrad": acc.get("lead_p95_mrad"),
        "frac_ref_gt_100mrad": acc.get("frac_ref_gt_100mrad"),
        "frac_track_gt_100mrad": acc.get("frac_track_gt_100mrad"),
    }
