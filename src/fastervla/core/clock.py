"""Monotonic clock helpers and sliding-window latency percentiles."""

from __future__ import annotations

import time
from collections import deque

import numpy as np


def monotonic_s() -> float:
    return time.monotonic()


def monotonic_ns() -> int:
    return time.monotonic_ns()


class LatencyWindow:
    """Keep the last ``maxlen`` latency samples (e.g. milliseconds)."""

    def __init__(self, maxlen: int = 64) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self._samples: deque[float] = deque(maxlen=int(maxlen))

    def record(self, value: float) -> None:
        self._samples.append(float(value))

    def percentile(self, q: float) -> float | None:
        if not self._samples:
            return None
        arr = np.asarray(self._samples, dtype=np.float64)
        return float(np.percentile(arr, q))

    def p50(self) -> float | None:
        return self.percentile(50)

    def p95(self) -> float | None:
        return self.percentile(95)

    def p99(self) -> float | None:
        return self.percentile(99)
