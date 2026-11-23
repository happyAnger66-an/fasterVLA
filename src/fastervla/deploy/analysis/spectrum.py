"""Frequency-band helpers for jitter metrics (hf3 / hf7 / hf8)."""

from __future__ import annotations

import numpy as np

FC_HZ = 3.0


def band_rms(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """RMS of ``x`` restricted to ``[lo, hi)`` Hz; ``x`` is ``(T, D)``."""
    n = int(x.shape[0])
    if n < 2:
        return np.zeros(x.shape[1], dtype=np.float64)
    y = np.concatenate([x, x[::-1]], axis=0)
    spec = np.fft.rfft(y, axis=0)
    freqs = np.fft.rfftfreq(2 * n, d=1.0 / fs)
    spec[(freqs < lo) | (freqs >= hi)] = 0
    out = np.fft.irfft(spec, n=2 * n, axis=0)[:n]
    return np.sqrt(np.mean(out**2, axis=0))
