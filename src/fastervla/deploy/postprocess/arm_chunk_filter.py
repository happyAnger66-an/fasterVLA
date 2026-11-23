"""Arm chunk zero-phase DCT low-pass (deploy-side; mirrors old soft ZP@3Hz).

Only policy arm dims ``[8:22]`` are filtered; waist / grippers pass through.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fastervla.deploy.client.io.layout import ARM_POLICY_SLICE

DEFAULT_CUTOFF_HZ = 3.0
DEFAULT_CONTEXT_FRAMES = 10
ARM_SLICE = slice(ARM_POLICY_SLICE[0], ARM_POLICY_SLICE[1])


def lowpass_dct(x: np.ndarray, cutoff_hz: float, fps: float) -> np.ndarray:
    """Zero-phase DCT low-pass on a known block. ``x`` shape ``(T, J)``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"lowpass_dct expects 2-D array, got shape={x.shape}")
    n = int(x.shape[0])
    if n < 3:
        return x.copy()
    k = np.arange(n, dtype=np.float64)
    basis = np.cos(np.pi * (k[:, None] + 0.5) * k[None, :] / n)
    coef = np.linalg.lstsq(basis, x, rcond=None)[0]
    freqs = k / (2.0 * n / float(fps))
    coef = coef.copy()
    coef[freqs > float(cutoff_hz)] = 0.0
    return basis @ coef


def apply_zero_phase_dct(
    actions: np.ndarray,
    *,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    fps: float,
    arm_slice: slice = ARM_SLICE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Filter one chunk. ``actions`` shape ``(T, >=22)``."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] < arm_slice.stop:
        raise ValueError(f"actions must be (T, >= {arm_slice.stop}), got {actions.shape}")
    out = actions.copy()
    raw_arm = actions[:, arm_slice].astype(np.float64)
    filtered_arm = lowpass_dct(raw_arm, cutoff_hz=float(cutoff_hz), fps=float(fps))
    out[:, arm_slice] = filtered_arm.astype(np.float32)
    delta = np.abs(filtered_arm - raw_arm)
    meta: dict[str, Any] = {
        "mode": "zero_phase_dct",
        "cutoff_hz": float(cutoff_hz),
        "fps": float(fps),
        "arm_slice": [arm_slice.start, arm_slice.stop],
        "max_delta_rad": float(delta.max()) if delta.size else 0.0,
        "mean_delta_rad": float(delta.mean()) if delta.size else 0.0,
        "context_frames": 0,
    }
    return out, meta


def apply_zero_phase_dct_cross(
    actions: np.ndarray,
    prev_filtered: np.ndarray | None,
    *,
    context_frames: int = DEFAULT_CONTEXT_FRAMES,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    fps: float,
    arm_slice: slice = ARM_SLICE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stitch previous tail + new chunk, DCT, keep new segment."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] < arm_slice.stop:
        raise ValueError(f"actions must be (T, >= {arm_slice.stop}), got {actions.shape}")
    if prev_filtered is None:
        out, meta = apply_zero_phase_dct(
            actions, cutoff_hz=cutoff_hz, fps=fps, arm_slice=arm_slice
        )
        meta["mode"] = "zero_phase_dct_cross"
        meta["cross_fallback"] = "no_prev"
        return out, meta

    prev = np.asarray(prev_filtered, dtype=np.float32)
    if prev.ndim != 2 or prev.shape[-1] < arm_slice.stop:
        raise ValueError(f"prev_filtered must be (T, >= {arm_slice.stop}), got {prev.shape}")
    ctx = int(max(0, min(int(context_frames), prev.shape[0])))
    if ctx < 3:
        out, meta = apply_zero_phase_dct(
            actions, cutoff_hz=cutoff_hz, fps=fps, arm_slice=arm_slice
        )
        meta["mode"] = "zero_phase_dct_cross"
        meta["cross_fallback"] = "short_prev"
        meta["context_frames"] = ctx
        return out, meta

    raw_new = actions[:, arm_slice].astype(np.float64)
    raw_ctx = prev[-ctx:, arm_slice].astype(np.float64)
    stitched = np.concatenate([raw_ctx, raw_new], axis=0)
    filtered = lowpass_dct(stitched, cutoff_hz=float(cutoff_hz), fps=float(fps))
    out = actions.copy()
    out[:, arm_slice] = filtered[ctx:].astype(np.float32)
    delta = np.abs(filtered[ctx:] - raw_new)
    handoff_jump = float(np.max(np.abs(out[0, arm_slice] - prev[-1, arm_slice])))
    meta = {
        "mode": "zero_phase_dct_cross",
        "cutoff_hz": float(cutoff_hz),
        "fps": float(fps),
        "arm_slice": [arm_slice.start, arm_slice.stop],
        "context_frames": ctx,
        "max_delta_rad": float(delta.max()) if delta.size else 0.0,
        "mean_delta_rad": float(delta.mean()) if delta.size else 0.0,
        "handoff_jump_rad": handoff_jump,
        "cross_fallback": None,
    }
    return out, meta


def apply_arm_chunk_filter(
    actions: np.ndarray,
    *,
    mode: str = "none",
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    fps: float = 30.0,
    prev_filtered: np.ndarray | None = None,
    context_frames: int = DEFAULT_CONTEXT_FRAMES,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Dispatch filter. ``mode='none'`` returns a copy and ``meta=None``."""
    mode_n = str(mode or "none").lower()
    if mode_n in ("none", "off", ""):
        return np.asarray(actions, dtype=np.float32).copy(), None
    if mode_n == "zero_phase_dct":
        return apply_zero_phase_dct(actions, cutoff_hz=cutoff_hz, fps=fps)
    if mode_n == "zero_phase_dct_cross":
        return apply_zero_phase_dct_cross(
            actions,
            prev_filtered,
            context_frames=context_frames,
            cutoff_hz=cutoff_hz,
            fps=fps,
        )
    raise ValueError(
        f"unknown arm_chunk_filter {mode!r}; "
        "expected none|zero_phase_dct|zero_phase_dct_cross"
    )
