"""Monotone cubic Hermite (PCHIP) helpers without a scipy hard dependency."""

from __future__ import annotations

import numpy as np


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson slopes. ``x`` (N,), ``y`` (N, D) → slopes (N, D)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    n, d = y.shape
    if x.shape[0] != n:
        raise ValueError(f"x length {x.shape[0]} != y rows {n}")
    if n == 1:
        return np.zeros((1, d), dtype=np.float64)
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing")
    delta = np.diff(y, axis=0) / h[:, None]
    m = np.zeros((n, d), dtype=np.float64)
    if n == 2:
        m[0] = delta[0]
        m[1] = delta[0]
        return m
    # Interior
    for i in range(1, n - 1):
        d0 = delta[i - 1]
        d1 = delta[i]
        for j in range(d):
            if d0[j] * d1[j] <= 0:
                m[i, j] = 0.0
            else:
                w1 = 2.0 * h[i] + h[i - 1]
                w2 = h[i] + 2.0 * h[i - 1]
                m[i, j] = (w1 + w2) / (w1 / d0[j] + w2 / d1[j])
    # Endpoints (one-sided)
    m[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    m[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    for j in range(d):
        if m[0, j] * delta[0, j] <= 0:
            m[0, j] = 0.0
        elif delta[0, j] * delta[1, j] < 0 and abs(m[0, j]) > abs(3.0 * delta[0, j]):
            m[0, j] = 3.0 * delta[0, j]
        if m[-1, j] * delta[-1, j] <= 0:
            m[-1, j] = 0.0
        elif delta[-1, j] * delta[-2, j] < 0 and abs(m[-1, j]) > abs(3.0 * delta[-1, j]):
            m[-1, j] = 3.0 * delta[-1, j]
    return m


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Public alias for cached evaluation via ``pchip_eval(..., slopes=...)``."""
    return _pchip_slopes(x, y)


def pchip_eval(
    x: np.ndarray, y: np.ndarray, xq: float, *, slopes: np.ndarray | None = None
) -> np.ndarray:
    """Evaluate PCHIP at scalar ``xq``. Returns shape ``(D,)``.

    ``slopes`` from :func:`pchip_slopes` on the same ``(x, y)`` skips recomputation.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64)
    squeeze = y.ndim == 1
    if squeeze:
        y = y[:, None]
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("empty PCHIP")
    if n == 1:
        out = y[0].copy()
        return out[0] if squeeze else out
    xq_f = float(xq)
    if xq_f <= x[0]:
        out = y[0].copy()
        return out[0] if squeeze else out
    if xq_f >= x[-1]:
        out = y[-1].copy()
        return out[0] if squeeze else out
    if slopes is None:
        m = _pchip_slopes(x, y)
    else:
        m = np.asarray(slopes, dtype=np.float64)
        if m.ndim == 1:
            m = m[:, None]
    i = int(np.searchsorted(x, xq_f, side="right") - 1)
    i = max(0, min(i, n - 2))
    h = x[i + 1] - x[i]
    t = (xq_f - x[i]) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    out = h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]
    return out[0] if squeeze else out
