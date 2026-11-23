"""Load a robot-live run directory and emit a JSON-serializable summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fastervla.deploy.analysis.jitter import chunk_metrics, exec_metrics
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
        "r2r_p50_ms": float(np.median(r2r_vals)) if r2r_vals else report.get("r2r_p50_ms"),
        "r2r_p95_ms": float(np.percentile(r2r_vals, 95))
        if len(r2r_vals) >= 2
        else (float(r2r_vals[0]) if r2r_vals else report.get("r2r_p95_ms")),
        "obs_age_p95_ms": float(np.percentile(obs_age, 95)) if obs_age else report.get("image_age_p95_ms"),
        "handoff_delta_p95_rad": float(np.percentile(prev_diff, 95))
        if len(prev_diff) >= 2
        else (float(prev_diff[0]) if prev_diff else None),
        "chunk_raw": chunk_metrics(raw, origin, handoff, fps),
        "chunk_filtered": chunk_metrics(actions, origin, handoff, fps),
        "exec": exec_metrics(ctl, handoff, fs=control_hz),
        "tracking": tracking_metrics(ctl),
        "report_track_err_p95_rad": report.get("track_err_p95_rad"),
    }
    # Compact headline metrics for CLI printing
    ch = summary["chunk_filtered"]
    ex = summary["exec"]
    tr = summary["tracking"]
    summary["headline"] = {
        "hf3_mrad": ch.get("hf3_mrad_mean"),
        "hf7_mrad": ch.get("hf7_mrad_mean"),
        "rev_per_s": ch.get("rev_per_s"),
        "handoff_jump_p95_mrad": ch.get("handoff_jump_p95_mrad"),
        "cmd_hf3_mrad": ex.get("cmd_hf3_mrad_mean"),
        "meas_hf3_mrad": ex.get("meas_hf3_mrad_mean"),
        "track_p95_rad": tr.get("cmd_track_p95_rad") or tr.get("cmd_track_p95_step_rad"),
        "empty_frac": ex.get("empty_frac") if ex.get("empty_frac") is not None else tr.get("empty_frac"),
        "r2r_p95_ms": summary["r2r_p95_ms"],
    }
    return summary


def analyze_results_dir(
    results_dir: str | Path,
    *,
    pattern: str = "robot-live-*",
    min_chunks: int = 1,
) -> list[dict[str, Any]]:
    """Analyze every matching run under ``results_dir``."""
    root = Path(results_dir)
    rows: list[dict[str, Any]] = []
    for d in sorted(root.glob(pattern)):
        if not d.is_dir():
            continue
        try:
            rows.append(analyze_run(d, min_chunks=min_chunks))
        except Exception as exc:  # noqa: BLE001
            rows.append({"run_dir": str(d), "error": str(exc)})
    return rows
