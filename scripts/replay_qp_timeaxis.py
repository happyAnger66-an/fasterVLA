#!/usr/bin/env python3
"""Offline replay of TimeAxisQP variants on recorded robot-live chunks.

Re-solves every accepted chunk in merge order (trace anchor_step / d_actual),
optionally locking the first d_hat+extra durations to the previous *replayed*
plan, then stitches the executed frames into one wall-clock rate sequence.

  python3 scripts/replay_qp_timeaxis.py --runs results/robot-live-* --kappa 1.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fastervla.core.types import Chunk
from fastervla.deploy.postprocess.retime import (
    TimeAxisQP,
    TimeAxisQPConfig,
    UniformRetimer,
    contact_mask,
)

ARM = slice(8, 22)
FPS = 30.0


def _variants(kappa: float) -> dict[str, dict]:
    return {
        "uniform": {"uniform": True},
        "qp_old": {},
        "qp_lock": {"lock": True},
        "qp_contact1.0": {"contact_kappa": 1.0},
        "qp_contact1.0_lock": {"contact_kappa": 1.0, "lock": True},
        "qp_contact1.0_tr1.5_lock": {
            "contact_kappa": 1.0,
            "transport_kappa_max": 1.5,
            "lock": True,
        },
        "qp_contact1.1_tr1.5_lock": {
            "contact_kappa": 1.1,
            "transport_kappa_max": 1.5,
            "lock": True,
        },
        "qp_c1.1_tr1.5_dwell2_lock": {
            "contact_kappa": 1.1,
            "transport_kappa_max": 1.5,
            "dwell_kappa_max": 2.0,
            "dwell_step_rad": 0.005,
            "lock": True,
        },
    }


def replay(run_dir: Path, kappa: float, variant: dict, lock_extra: int) -> dict:
    z = np.load(run_dir / "prediction_trace.npz", allow_pickle=True)
    act = np.asarray(z["actions"], dtype=np.float64)
    st = np.asarray(z["inference_state"], dtype=np.float64)
    acc = np.asarray(z["merge_accepted"]).astype(bool)
    anchor = np.asarray(z["anchor_step"], dtype=np.int64)
    d_hat = np.asarray(z["d_hat"], dtype=np.int64)
    d_act = np.asarray(z["d_actual"], dtype=np.float64)
    idx = np.flatnonzero(acc)
    h = act.shape[1]

    if variant.get("uniform"):
        retimer = UniformRetimer(policy_fps=FPS, kappa=kappa, kappa_max=2.0)
    else:
        kw = {k: v for k, v in variant.items() if k != "lock"}
        retimer = TimeAxisQP(
            TimeAxisQPConfig(policy_fps=FPS, kappa=kappa, kappa_max=2.0, **kw)
        )

    plans: list[np.ndarray] = []
    prev_dt: np.ndarray | None = None
    for i in idx:
        chunk = Chunk(waypoints=act[i].astype(np.float32), model_dt=1 / FPS, obs_ts=0.0)
        if variant.get("uniform"):
            timed = retimer.retime(chunk)
        else:
            lock = None
            if variant.get("lock") and prev_dt is not None:
                a = int(anchor[i])
                lock = prev_dt[a : a + max(0, int(d_hat[i])) + lock_extra]
                lock = lock if len(lock) else None
            timed = retimer.retime(chunk, state=st[i], lock_durations=lock)
        prev_dt = np.asarray(timed.step_dt, dtype=np.float64)
        plans.append(prev_dt)

    rates, near, steps = [], [], []
    for k, i in enumerate(idx):
        start = int(np.ceil(max(0.0, d_act[i])))
        if k + 1 < len(idx):
            j = idx[k + 1]
            stop = int(anchor[j]) + int(np.floor(max(0.0, d_act[j])))
        else:
            stop = h
        stop = min(max(stop, start), h)
        dt = plans[k][start:stop]
        rates.append(1.0 / dt)
        m = contact_mask(act[i], grip_dims=(22, 23), delta=0.02, window=10)
        near.append(m[start:stop])
        pts = np.vstack([st[i, ARM], act[i, :, ARM]])
        steps.append(np.linalg.norm(np.diff(pts, axis=0), axis=1)[start:stop])
    r = np.concatenate(rates)
    nr = np.concatenate(near)
    sp = np.concatenate(steps)
    x = r / FPS  # speed in × demo
    dx = np.abs(np.diff(x))
    q25, q75 = np.quantile(sp, [0.25, 0.75])
    jv = sp * r
    return {
        "frames": int(len(r)),
        "eff_kappa": float(len(r) / FPS / np.sum(1.0 / r)),
        "near_grip_x": float(len(r[nr]) / FPS / np.sum(1.0 / r[nr])) if nr.any() else None,
        "far_grip_x": float(len(r[~nr]) / FPS / np.sum(1.0 / r[~nr])),
        "small_step_x_p50": float(np.median(x[sp <= q25])),
        "large_step_x_p50": float(np.median(x[sp >= q75])),
        "x_p05": float(np.percentile(x, 5)),
        "x_p95": float(np.percentile(x, 95)),
        "dx_p95": float(np.percentile(dx, 95)),
        "dx_max": float(dx.max()),
        "joint_speed_p95": float(np.percentile(jv, 95)),
        "near_frac": float(nr.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--kappa", type=float, default=None, help="default: run's κ")
    ap.add_argument("--lock-extra", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    out = {}
    for run in a.runs:
        rd = Path(run)
        rep = json.loads((rd / "robot_bridge_report.json").read_text())
        kappa = a.kappa or float(rep["experiment"]["kappa"])
        out[rd.name] = {"kappa": kappa}
        for name, v in _variants(kappa).items():
            out[rd.name][name] = replay(rd, kappa, v, a.lock_extra)
    keys = ["eff_kappa", "near_grip_x", "far_grip_x", "small_step_x_p50",
            "large_step_x_p50", "x_p05", "x_p95", "dx_p95", "dx_max", "joint_speed_p95"]
    for run, res in out.items():
        print(f"== {run}  κ={res['kappa']}  near_frac={res['uniform']['near_frac']:.2f}")
        print(f"{'variant':28s}" + "".join(f"{k[:10]:>11s}" for k in keys))
        for name in _variants(res["kappa"]):
            row = res[name]
            print(f"{name:28s}" + "".join(
                f"{(row[k] if row[k] is not None else float('nan')):11.3f}" for k in keys))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
