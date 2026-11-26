#!/usr/bin/env python3
"""Batch control-quality metrics (lag / contour / FRF / grip-event EE error).

  python scripts/analyze_control_quality.py results/robot-live-2026112*-* \
      --out results/control_quality_20261124.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastervla.deploy.analysis.control_quality import (  # noqa: E402
    control_headline,
    control_quality_metrics,
)


def _fmt(v: float | None, nd: int = 0) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--urdf", default=None)
    args = ap.parse_args()

    rows = []
    for d in args.runs:
        ctl_path = d / "control_trace.npz"
        if not ctl_path.is_file():
            continue
        data = np.load(ctl_path, allow_pickle=True)
        ctl = {k: data[k] for k in data.files}
        rep_path = d / "robot_bridge_report.json"
        exp = {}
        if rep_path.is_file():
            exp = json.loads(rep_path.read_text(encoding="utf-8")).get("experiment") or {}
        cq = control_quality_metrics(ctl, urdf_hint=args.urdf)
        if "lag" not in cq:
            continue
        head = control_headline(cq)
        run_id = d.name.replace("robot-live-", "")
        rows.append(
            {
                "run_id": run_id,
                "kappa": exp.get("kappa"),
                "retimer": exp.get("retimer"),
                "tracker": exp.get("tracker"),
                "prefix": exp.get("execution_prefix_frames"),
                "buffer": exp.get("execution_buffer_frames"),
                "qp_prefix_lock": exp.get("qp_prefix_lock"),
                "headline": head,
                "control": cq,
            }
        )
        print(
            f"{run_id[-6:]} κ={exp.get('kappa')} {exp.get('retimer')} "
            f"lag cmd/ref p50 {_fmt(head['lag_cmd_p50_ms'])}/{_fmt(head['lag_ref_p50_ms'])} "
            f"xcorr ref {_fmt(head['xcorr_ref_p50_ms'])} "
            f"contour p95 {_fmt(head['contour_ref_p95_mrad'])}mrad "
            f"EE contour p95 {_fmt(head.get('ee_contour_ref_p95_mm'), 1)}mm "
            f"grip close {_fmt(head.get('grip_close_err_p50_mm'), 1)}mm "
            f"bw {_fmt(head['bw_cmd_hz'], 1)}Hz f95 {_fmt(head['cmd_f95_hz'], 2)}Hz"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
