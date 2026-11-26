"""One-shot post-run summary: ball cycles + phases + control headline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fastervla.deploy.analysis.ball_cycles import analyze_ball_cycles
from fastervla.deploy.analysis.summarize import analyze_run


def _load_ctl(run_dir: Path) -> dict[str, np.ndarray]:
    z = np.load(run_dir / "control_trace.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def _stat(d: dict[str, Any] | None, key: str = "p50") -> float | None:
    if not d or not isinstance(d, dict) or not d.get("n"):
        return None
    v = d.get(key)
    return float(v) if v is not None else None


def report_run(
    run_dir: str | Path,
    *,
    control: bool = True,
    urdf: str | Path | None = None,
) -> dict[str, Any]:
    d = Path(run_dir)
    summary = analyze_run(d, control=control, urdf=urdf)
    ctl = _load_ctl(d)
    balls = analyze_ball_cycles(ctl)
    ch = summary.get("control_headline") or {}
    phases = (summary.get("control") or {}).get("phases") or {}

    ee_ph = phases.get("ee") or {}
    frac = phases.get("frac") or {}
    phase_ee: dict[str, Any] = {}
    for name in ("seek", "grasp", "carry", "place"):
        pe = ee_ph.get(name) or {}
        phase_ee[name] = {
            "ee_total_p95_mm": (pe.get("err") or {}).get("p95"),
            "ee_contour_p95_mm": (pe.get("contour") or {}).get("p95"),
            "ee_lag_p50_ms": (pe.get("lag_ms") or {}).get("p50"),
            "frac": frac.get(name),
        }

    exp = dict(summary.get("experiment") or {})
    out = {
        "run_dir": str(d),
        "run": d.name.replace("robot-live-", ""),
        "tag": exp.get("tag"),
        "experiment": exp or (summary.get("metrics_meta") or {}),
        "headline": summary.get("headline") or {},
        "balls": {
            "n_balls": balls.get("n_balls"),
            "dur_s": balls.get("dur_s"),
            "balls_per_min": balls.get("balls_per_min"),
            "grasp_to_place_p50": _stat(balls.get("grasp_to_place")),
            "release_to_grasp_p50": _stat(balls.get("release_to_grasp")),
            "release_to_release_p50": _stat(balls.get("release_to_release")),
            "first_try": (balls.get("attempts") or {}).get("first_try"),
            "seek_dwell_frac_p50": (balls.get("seek_motion") or {}).get("dwell_frac_p50"),
            "seek_dwell_s_p50": (balls.get("seek_motion") or {}).get("dwell_s_p50"),
        },
        "control": {
            "lag_ref_p50_ms": ch.get("lag_ref_p50_ms"),
            "lag_ref_p95_ms": ch.get("lag_ref_p95_ms"),
            "xcorr_ref_p50_ms": ch.get("xcorr_ref_p50_ms"),
            "ee_contour_ref_p95_mm": ch.get("ee_contour_ref_p95_mm"),
            "grip_close_err_p50_mm": ch.get("grip_close_err_p50_mm"),
            "seek_vel_util_p50": ch.get("phase_seek_vel_util_p50"),
            "seek_vel_util_p95": ch.get("phase_seek_vel_util_p95"),
            "carry_vel_util_p50": ch.get("phase_carry_vel_util_p50"),
            "carry_vel_util_p95": ch.get("phase_carry_vel_util_p95"),
            "seek_vel_gap_p50": ch.get("phase_seek_vel_gap_p50"),
            "seek_vel_gap_p95": ch.get("phase_seek_vel_gap_p95"),
            "carry_vel_gap_p50": ch.get("phase_carry_vel_gap_p50"),
            "carry_vel_gap_p95": ch.get("phase_carry_vel_gap_p95"),
        },
        "phases_ee": phase_ee,
        "ball_detail": balls,
        "definition": {
            "grasp_to_place": "夹→放（搬运）",
            "release_to_grasp": "松→夹（找球）",
            "release_to_release": "松→松（单球周期）",
        },
    }
    return out


def format_report_line(rep: dict[str, Any]) -> str:
    b = rep.get("balls") or {}
    c = rep.get("control") or {}

    def fmt(v: Any, nd: int = 2) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return str(v)

    def pct(v: Any, nd: int = 0) -> str:
        return "—" if v is None else fmt(float(v) * 100.0, nd)

    tag = rep.get("tag")
    tag_s = f" tag={tag}" if tag else ""
    ft = b.get("first_try")
    return (
        f"{rep.get('run')}{tag_s}  "
        f"bpm={fmt(b.get('balls_per_min'))} n={b.get('n_balls')}  "
        f"夹→放(搬运)p50={fmt(b.get('grasp_to_place_p50'))}s  "
        f"松→夹(找球)p50={fmt(b.get('release_to_grasp_p50'))}s  "
        f"松→松p50={fmt(b.get('release_to_release_p50'))}s  "
        f"找球停p50={fmt(b.get('seek_dwell_frac_p50'), 2)}  "
        f"lag_ref_p50={fmt(c.get('lag_ref_p50_ms'), 0)}ms  "
        f"grip_close_ee_p50={fmt(c.get('grip_close_err_p50_mm'), 1)}mm  "
        f"first_try={pct(ft)}%  "
        f"找球限速util_p50/p95={pct(c.get('seek_vel_util_p50'))}/{pct(c.get('seek_vel_util_p95'))}%  "
        f"搬运限速util_p50/p95={pct(c.get('carry_vel_util_p50'))}/{pct(c.get('carry_vel_util_p95'))}%"
    )


def write_report(
    run_dir: str | Path,
    *,
    out: str | Path | None = None,
    control: bool = True,
    urdf: str | Path | None = None,
) -> dict[str, Any]:
    rep = report_run(run_dir, control=control, urdf=urdf)
    line = format_report_line(rep)
    print(line, flush=True)
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"wrote {p}", flush=True)
    return rep
