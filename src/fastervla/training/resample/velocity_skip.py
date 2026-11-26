"""Velocity-constrained farthest-jump frame skip for action chunks."""

from __future__ import annotations

from collections.abc import Sequence

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


def gripper_no_skip_mask(
    q_traj: np.ndarray,
    *,
    grip_dims: Sequence[int],
    delta: float,
    window_frames: int,
) -> np.ndarray:
    """Protect frames within ``window_frames`` of a gripper command change.

    Same dilation idea as deploy ``contact_mask``: any |Δgrip| > ``delta`` marks
    an event, then ±``window_frames`` around each event are no-skip.
    """
    q_traj = np.asarray(q_traj, dtype=np.float64)
    if q_traj.ndim == 1:
        q_traj = q_traj[:, None]
    h = int(q_traj.shape[0])
    dims = [int(d) for d in grip_dims if 0 <= int(d) < q_traj.shape[1]]
    if not dims or h < 2 or window_frames < 0:
        return np.zeros(h, dtype=bool)
    g = q_traj[:, dims]
    step = np.max(np.abs(np.diff(g, axis=0)), axis=1)
    events = np.flatnonzero(step > float(delta)) + 1
    mask = np.zeros(h, dtype=bool)
    w = int(window_frames)
    for e in events:
        mask[max(0, e - w) : min(h, e + w + 1)] = True
    return mask


def farthest_legal_index(
    q_traj: np.ndarray,
    a: int,
    v_lim: np.ndarray,
    dt: float,
    end: int | None = None,
    no_skip_mask: np.ndarray | None = None,
) -> int:
    """Return farthest index b > a that is velocity-legal; else min(a+1, end).

    If ``no_skip_mask`` is set, jumps may not skip over any protected frame in
    ``(a, b)`` (dense walk through gripper open/close windows).
    """
    q_traj = np.asarray(q_traj, dtype=np.float64)
    if q_traj.ndim == 1:
        q_traj = q_traj[:, None]
    last = int(q_traj.shape[0] - 1) if end is None else int(end)
    a = int(a)
    if a >= last:
        return last

    mask = None if no_skip_mask is None else np.asarray(no_skip_mask, dtype=bool)
    if mask is not None and mask.shape[0] != q_traj.shape[0]:
        raise ValueError(
            f"no_skip_mask length {mask.shape[0]} != traj length {q_traj.shape[0]}"
        )

    q_a = q_traj[a]
    for b in range(last, a, -1):
        # b == a+1 never skips an interior frame; always eligible for fallback.
        if mask is not None and b > a + 1 and bool(np.any(mask[a + 1 : b])):
            continue
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
    no_skip_mask: np.ndarray | None = None,
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
        hx = farthest_legal_index(
            q_traj,
            a=a,
            v_lim=v_lim,
            dt=dt,
            end=last,
            no_skip_mask=no_skip_mask,
        )
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
    no_skip_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Gather joint rows for the skipped chunk; shape [H, D]."""
    q_traj = np.asarray(q_traj, dtype=np.float64)
    if q_traj.ndim == 1:
        q_traj = q_traj[:, None]
    indices = build_skip_chunk_indices(
        q_traj,
        h0=h0,
        horizon=horizon,
        v_lim=v_lim,
        dt=dt,
        skip_frames=skip_frames,
        no_skip_mask=no_skip_mask,
    )
    return q_traj[indices]
