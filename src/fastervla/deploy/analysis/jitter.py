"""Chunk- and execution-side jitter metrics (hf3 / hf7 / handoff)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fastervla.deploy.analysis.spectrum import FC_HZ, band_rms

ARM_ACT = slice(8, 22)
ARM_CMD = slice(5, 19)


def _interp_chunk(
    chunk: np.ndarray, origin: float, fps: float, t: float
) -> np.ndarray | None:
    pos = (t - origin) * fps
    if pos < 0 or pos > chunk.shape[0] - 1:
        return None
    i = int(np.floor(pos))
    j = min(i + 1, chunk.shape[0] - 1)
    w = pos - i
    return chunk[i] * (1 - w) + chunk[j] * w


def chunk_metrics(
    acts: np.ndarray,
    origin: np.ndarray,
    handoff: np.ndarray,
    fps: float,
) -> dict[str, Any]:
    """Per-run summary over prediction chunks ``(K, H, 24)``."""
    a = np.asarray(acts, dtype=np.float64)[:, :, ARM_ACT]
    k, n, d = a.shape
    if k < 1 or n < 2:
        return {"n_chunks": int(k), "chunk_len": int(n)}
    nyq = fps / 2.0
    hf = np.zeros((k, d))
    hf_hi = np.zeros((k, d))
    vel = np.zeros((k, d))
    rev = np.zeros(k)
    for c in range(k):
        hf[c] = band_rms(a[c], fps, FC_HZ, min(7.5, nyq) + 1e-9)
        if nyq > 7.5:
            hf_hi[c] = band_rms(a[c], fps, 7.5, nyq + 1e-9)
        v = np.diff(a[c], axis=0) * fps
        vel[c] = np.mean(np.abs(v), axis=0)
        s = np.sign(np.where(np.abs(v) > 0.02, v, 0))
        flips = np.sum((s[1:] * s[:-1]) < 0)
        rev[c] = flips / (d * (n - 1) / fps)
    acc = np.diff(a, n=2, axis=1) * fps**2
    jumps: list[np.ndarray] = []
    for c in range(k - 1):
        t = float(handoff[c]) if c < len(handoff) else float("nan")
        if not np.isfinite(t):
            continue
        p_old = _interp_chunk(a[c], float(origin[c]), fps, t)
        p_new = _interp_chunk(a[c + 1], float(origin[c + 1]), fps, t)
        if p_old is None or p_new is None:
            continue
        jumps.append(np.abs(p_new - p_old))
    out: dict[str, Any] = {
        "n_chunks": int(k),
        "chunk_len": int(n),
        "hf3_mrad_mean": float(np.mean(hf) * 1e3),
        "hf3_mrad": (hf.mean(0) * 1e3).tolist(),
        "hf7_mrad_mean": float(np.mean(hf_hi) * 1e3),
        "hf7_mrad": (hf_hi.mean(0) * 1e3).tolist(),
        "acc_rms_mean": float(np.sqrt(np.mean(acc**2))),
        "mean_speed_mean": float(vel.mean()),
        "rev_per_s": float(rev.mean()),
    }
    if jumps:
        j = np.asarray(jumps)
        out["handoff_jump_mrad_mean"] = float(j.mean() * 1e3)
        out["handoff_jump_p95_mrad"] = float(np.percentile(j.max(1), 95) * 1e3)
    return out


def _segments(mask: np.ndarray, min_len: int):
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    for s, e in zip(idx[::2], idx[1::2]):
        if e - s >= min_len:
            yield s, e


def exec_metrics(
    ctl: dict[str, np.ndarray],
    handoffs: np.ndarray,
    *,
    fs: float = 100.0,
) -> dict[str, Any]:
    """100 Hz command / measured jitter on step segments."""
    t = np.asarray(ctl["monotonic_s"], dtype=np.float64)
    if t.size < 2:
        return {"exec_seconds": 0.0}
    cmd = np.asarray(ctl["command21"], dtype=np.float64)[:, ARM_CMD]
    meas = np.asarray(ctl["measured_state24"], dtype=np.float64)[:, ARM_ACT]
    if "stage" in ctl and "stage_names" in ctl:
        stage = np.asarray(ctl["stage"])
        names = [str(x) for x in np.asarray(ctl["stage_names"]).tolist()]
        step_id = names.index("step") if "step" in names else 0
        empty_id = names.index("buffer_empty_hold") if "buffer_empty_hold" in names else -1
    else:
        stage = np.zeros(t.shape[0], dtype=np.int8)
        step_id, empty_id = 0, -1

    tu = np.arange(float(t[0]), float(t[-1]), 1.0 / fs)
    if tu.size < int(fs):
        return {"exec_seconds": 0.0}
    cmd_u = np.stack([np.interp(tu, t, cmd[:, i]) for i in range(cmd.shape[1])], 1)
    meas_u = np.stack([np.interp(tu, t, meas[:, i]) for i in range(meas.shape[1])], 1)
    st_u = stage[np.clip(np.searchsorted(t, tu), 0, len(t) - 1)]
    run_mask = st_u == step_id
    buckets: dict[str, list[np.ndarray]] = {
        k: [] for k in ("c3", "c8", "m3", "m8", "trk", "cv", "mv")
    }
    weights: list[int] = []
    for s, e in _segments(run_mask, int(1.0 * fs)):
        c, m = cmd_u[s:e], meas_u[s:e]
        buckets["c3"].append(band_rms(c, fs, FC_HZ, 8.0))
        buckets["c8"].append(band_rms(c, fs, 8.0, 50.0))
        buckets["m3"].append(band_rms(m, fs, FC_HZ, 8.0))
        buckets["m8"].append(band_rms(m, fs, 8.0, 50.0))
        buckets["trk"].append(np.sqrt(np.mean((c - m) ** 2, axis=0)))
        buckets["cv"].append(np.mean(np.abs(np.diff(c, axis=0)) * fs, axis=0))
        buckets["mv"].append(np.mean(np.abs(np.diff(m, axis=0)) * fs, axis=0))
        weights.append(e - s)
    if not weights:
        return {
            "exec_seconds": 0.0,
            "empty_frac": float(np.mean(st_u == empty_id)) if empty_id >= 0 else None,
        }
    w = np.asarray(weights, dtype=np.float64)[:, None]

    def avg(key: str) -> list[float]:
        return (np.sum(np.asarray(buckets[key]) * w, 0) / w.sum()).tolist()

    c3 = avg("c3")
    c8 = avg("c8")
    m3 = avg("m3")
    m8 = avg("m8")
    trk = avg("trk")
    return {
        "exec_seconds": float(w.sum() / fs),
        "empty_frac": float(np.mean(st_u == empty_id)) if empty_id >= 0 else None,
        "cmd_hf3_mrad_mean": float(np.mean(c3) * 1e3),
        "cmd_hf3_mrad": [x * 1e3 for x in c3],
        "cmd_hf8_mrad_mean": float(np.mean(c8) * 1e3),
        "meas_hf3_mrad_mean": float(np.mean(m3) * 1e3),
        "meas_hf3_mrad": [x * 1e3 for x in m3],
        "meas_hf8_mrad_mean": float(np.mean(m8) * 1e3),
        "track_rms_mrad_mean": float(np.mean(trk) * 1e3),
        "track_rms_mrad": [x * 1e3 for x in trk],
        "cmd_speed_mean": float(np.mean(avg("cv"))),
        "meas_speed_mean": float(np.mean(avg("mv"))),
    }
