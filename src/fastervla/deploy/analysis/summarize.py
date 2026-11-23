"""Load a robot-live run directory and emit a JSON-serializable summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fastervla.deploy.analysis.jitter import chunk_metrics, exec_metrics
from fastervla.deploy.analysis.metrics import compute_run_metrics, wide_row
from fastervla.deploy.analysis.tracking import tracking_metrics


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _as_float_list(arr: np.ndarray | None) -> list[float]:
    if arr is None:
        return []
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return [float(x) for x in a if np.isfinite(x)]


def analyze_run(
    run_dir: str | Path,
    *,
    min_chunks: int = 1,
    control_hz: float = 100.0,
    urdf: str | Path | None = None,
    ee_stride: int = 5,
) -> dict[str, Any]:
    """Analyze one ``robot-live-*`` directory.

    Accepts fasterVLA traces (``mode=fastervla`` / shadow) as well as legacy
    report shapes when fields are present.
    """
    d = Path(run_dir)
    ctl_path = d / "control_trace.npz"
    pred_path = d / "prediction_trace.npz"
    rep_path = d / "robot_bridge_report.json"
    if not ctl_path.is_file() or not pred_path.is_file():
        raise FileNotFoundError(f"missing control/prediction trace under {d}")

    ctl = _load_npz(ctl_path)
    pred = _load_npz(pred_path)
    report: dict[str, Any] = {}
    if rep_path.is_file():
        report = json.loads(rep_path.read_text(encoding="utf-8"))

    experiment = dict(report.get("experiment") or {})
    fps = float(experiment.get("policy_fps") or report.get("policy_fps") or 30.0)

    actions = np.asarray(pred["actions"])
    if actions.ndim != 3 or actions.shape[0] < min_chunks:
        raise ValueError(
            f"need actions (K,H,D) with K>={min_chunks}, got {getattr(actions, 'shape', None)}"
        )
    k = int(actions.shape[0])
    raw = (
        np.asarray(pred["actions_raw"])
        if "actions_raw" in pred
        else actions
    )

    if "action_origin_monotonic_s" in pred:
        origin = np.asarray(pred["action_origin_monotonic_s"], dtype=np.float64)
    elif "accepted_monotonic_s" in pred:
        origin = np.asarray(pred["accepted_monotonic_s"], dtype=np.float64)
    else:
        origin = np.arange(k, dtype=np.float64)

    if "handoff_at_monotonic_s" in pred:
        handoff = np.asarray(pred["handoff_at_monotonic_s"], dtype=np.float64)
    elif "accepted_monotonic_s" in pred:
        acc = np.asarray(pred["accepted_monotonic_s"], dtype=np.float64)
        handoff = np.concatenate([acc[1:], [np.nan]])
    else:
        handoff = np.full(k, np.nan)

    # r2r from prediction columns or inference_records
    r2r_vals = _as_float_list(pred.get("request_to_ready_ms"))
    if not r2r_vals:
        for rec in report.get("inference_records") or []:
            v = rec.get("request_to_ready_ms")
            if v is not None:
                r2r_vals.append(float(v))

    prev_diff = _as_float_list(pred.get("previous_command_difference"))
    obs_age = _as_float_list(pred.get("observation_age_ms"))

    r2r_p50 = float(np.median(r2r_vals)) if r2r_vals else report.get("r2r_p50_ms")
    r2r_p95 = (
        float(np.percentile(r2r_vals, 95))
        if len(r2r_vals) >= 2
        else (float(r2r_vals[0]) if r2r_vals else report.get("r2r_p95_ms"))
    )

    chunk_raw = chunk_metrics(raw, origin, handoff, fps)
    chunk_filt = chunk_metrics(actions, origin, handoff, fps)
    exec_m = exec_metrics(ctl, handoff, fs=control_hz)
    track_m = tracking_metrics(ctl)

    cats = compute_run_metrics(
        ctl=ctl,
        pred=pred,
        report=report,
        experiment=experiment,
        chunk_filtered=chunk_filt,
        exec_m=exec_m,
        r2r_p50_ms=float(r2r_p50) if r2r_p50 is not None else None,
        r2r_p95_ms=float(r2r_p95) if r2r_p95 is not None else None,
        n_chunks=k,
        control_hz=control_hz,
        urdf_hint=str(urdf) if urdf is not None else None,
        ee_stride=int(ee_stride),
    )

    summary: dict[str, Any] = {
        "run_dir": str(d),
        "run_id": d.name.replace("robot-live-", ""),
        "mode": report.get("mode"),
        "experiment": experiment,
        "fps": fps,
        "control_hz": control_hz,
        "n_control_ticks": int(np.asarray(ctl["monotonic_s"]).shape[0]),
        "n_chunks": k,
        "trace_drops": report.get("trace_drops"),
        "buffer_empty_cycles": report.get("buffer_empty_cycles"),
        "starved_events": report.get("starved_events"),
        "tracking_warning_cycles": report.get("tracking_warning_cycles"),
        "r2r_p50_ms": r2r_p50,
        "r2r_p95_ms": r2r_p95,
        "obs_age_p95_ms": float(np.percentile(obs_age, 95))
        if obs_age
        else report.get("image_age_p95_ms"),
        "handoff_delta_p95_rad": float(np.percentile(prev_diff, 95))
        if len(prev_diff) >= 2
        else (float(prev_diff[0]) if prev_diff else None),
        "chunk_raw": chunk_raw,
        "chunk_filtered": chunk_filt,
        "exec": exec_m,
        "tracking": track_m,
        "report_track_err_p95_rad": report.get("track_err_p95_rad"),
        "speed": cats["speed"],
        "sway": cats["sway"],
        "stability": cats["stability"],
        "accuracy": cats["accuracy"],
        "metrics_meta": cats["meta"],
    }
    summary["wide"] = wide_row(summary)

    # Compact headline for CLI / dashboards (four categories, no prog_jump).
    summary["headline"] = {
        # legacy + speed
        "kappa": experiment.get("kappa"),
        "progress_wall_fps": cats["speed"].get("progress_wall_fps"),
        "ee_speed_p50_mps": cats["speed"].get("ee_speed_p50_mps"),
        "ee_speed_p95_mps": cats["speed"].get("ee_speed_p95_mps"),
        "cmd_spd_p95": cats["speed"].get("cmd_spd_p95_rad_s"),
        # sway
        "merge_hz": cats["sway"].get("merge_hz"),
        "ref_jump_sw_p95_mrad": cats["sway"].get("ref_jump_sw_p95_mrad"),
        "cmd_jump_sw_p95_mrad": cats["sway"].get("cmd_jump_sw_p95_mrad"),
        "cmd_hf3_mrad": cats["sway"].get("cmd_hf3_mrad") or exec_m.get("cmd_hf3_mrad_mean"),
        "hf3_mrad": chunk_filt.get("hf3_mrad_mean"),
        "hf7_mrad": chunk_filt.get("hf7_mrad_mean"),
        "rev_per_s": chunk_filt.get("rev_per_s"),
        "handoff_jump_p95_mrad": chunk_filt.get("handoff_jump_p95_mrad"),
        # stability
        "empty": cats["stability"].get("empty"),
        "starved": cats["stability"].get("starved"),
        "rem_min": cats["stability"].get("rem_min"),
        "infer_duty": cats["stability"].get("infer_duty"),
        "empty_frac": exec_m.get("empty_frac")
        if exec_m.get("empty_frac") is not None
        else track_m.get("empty_frac"),
        "r2r_p95_ms": r2r_p95,
        # accuracy
        "ref_track_p95_mrad": cats["accuracy"].get("ref_track_p95_mrad"),
        "cmd_track_p95_mrad": cats["accuracy"].get("cmd_track_p95_mrad"),
        "lead_p95_mrad": cats["accuracy"].get("lead_p95_mrad"),
        "frac_ref_gt_100mrad": cats["accuracy"].get("frac_ref_gt_100mrad"),
        "track_p95_rad": track_m.get("cmd_track_p95_rad")
        or track_m.get("cmd_track_p95_step_rad"),
        "meas_hf3_mrad": exec_m.get("meas_hf3_mrad_mean"),
    }
    return summary


def analyze_results_dir(
    results_dir: str | Path,
    *,
    pattern: str = "robot-live-*",
    min_chunks: int = 1,
    urdf: str | Path | None = None,
    ee_stride: int = 5,
    wide_only: bool = False,
) -> list[dict[str, Any]]:
    """Analyze every matching run under ``results_dir``."""
    root = Path(results_dir)
    rows: list[dict[str, Any]] = []
    for d in sorted(root.glob(pattern)):
        if not d.is_dir():
            continue
        try:
            summary = analyze_run(
                d,
                min_chunks=min_chunks,
                urdf=urdf,
                ee_stride=ee_stride,
            )
            rows.append(summary["wide"] if wide_only else summary)
        except Exception as exc:  # noqa: BLE001
            rows.append({"run_dir": str(d), "error": str(exc)})
    return rows
