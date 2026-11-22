"""Velocity-constrained farthest-jump frame skip for action chunks."""

from __future__ import annotations

import numpy as np


def is_legal_jump(
    q_a: np.ndarray,
    q_b: np.ndarray,
    dt: float,
    v_lim: np.ndarray,
) -> bool:
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    vel = np.abs(np.asarray(q_b, dtype=np.float64) - np.asarray(q_a, dtype=np.float64)) / dt
    lim = np.asarray(v_lim, dtype=np.float64)
    return bool(np.all(vel <= lim))


def farthest_legal_index(
    q_traj: np.ndarray,
    a: int,
    v_lim: np.ndarray,
    dt: float,
    end: int | None = None,
) -> int:
    """Return farthest index b > a that is velocity-legal; else min(a+1, end)."""
    q_traj = np.asarray(q_traj, dtype=np.float64)
    if q_traj.ndim == 1:
        q_traj = q_traj[:, None]
    last = int(q_traj.shape[0] - 1) if end is None else int(end)
    a = int(a)
    if a >= last:
        return last

    q_a = q_traj[a]
    for b in range(last, a, -1):
        if is_legal_jump(q_a, q_traj[b], dt=dt, v_lim=v_lim):
            return b
    return min(a + 1, last)


def build_skip_chunk_indices(
    q_traj: np.ndarray,
    h0: int,
    horizon: int,
    v_lim: np.ndarray,
    dt: float | None = None,
    skip_frames: bool = True,
) -> np.ndarray:
    """Build H future frame indices by H successive farthest legal jumps."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    q_traj = np.asarray(q_traj, dtype=np.float64)
    if q_traj.ndim == 1:
        q_traj = q_traj[:, None]

    if not skip_frames:
        return np.arange(h0 + 1, h0 + 1 + horizon, dtype=np.int64)

    if dt is None:
        dt = 1.0 / float(horizon)

    indices = np.empty(horizon, dtype=np.int64)
    a = int(h0)
    last = int(q_traj.shape[0] - 1)
    for k in range(horizon):
        hx = farthest_legal_index(q_traj, a=a, v_lim=v_lim, dt=dt, end=last)
        indices[k] = hx
        a = hx
    return indices


def build_skip_chunk(
    q_traj: np.ndarray,
    h0: int,
    horizon: int,
    v_lim: np.ndarray,
    dt: float | None = None,
    skip_frames: bool = True,
) -> np.ndarray:
    """Gather joint rows for the skipped chunk; shape [H, D]."""
    q_traj = np.asarray(q_traj, dtype=np.float64)
    if q_traj.ndim == 1:
        q_traj = q_traj[:, None]
    indices = build_skip_chunk_indices(
        q_traj, h0=h0, horizon=horizon, v_lim=v_lim, dt=dt, skip_frames=skip_frames
    )
    return q_traj[indices]
