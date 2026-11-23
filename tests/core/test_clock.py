"""模块: fastervla.core.clock
功能: 统一 monotonic 时钟；延迟滑窗 P50/P95/P99。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fastervla.core.clock import LatencyWindow, monotonic_ns, monotonic_s


# 用例: monotonic_s / monotonic_ns 单调非降，且与 time.monotonic 同量级
# 参数: 无
def test_monotonic_clock_is_non_decreasing() -> None:
    t0 = monotonic_s()
    n0 = monotonic_ns()
    time.sleep(0.01)
    t1 = monotonic_s()
    n1 = monotonic_ns()
    assert t1 >= t0
    assert n1 >= n0
    assert abs(t1 - time.monotonic()) < 0.05
    assert abs(n1 / 1e9 - time.monotonic()) < 0.05


# 用例: 空窗口分位数行为明确（NaN 或报错，二者择一且稳定）
# 参数: 无
def test_latency_window_empty_quantile() -> None:
    win = LatencyWindow(maxlen=16)
    out = win.percentile(50)
    assert out is None or (isinstance(out, float) and np.isnan(out))


# 用例: 已知样本上 P50/P95/P99 与 numpy 一致（最近 maxlen 条）
# 参数: 无
def test_latency_window_percentiles_match_numpy() -> None:
    samples = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    win = LatencyWindow(maxlen=10)
    for x in samples:
        win.record(x)
    arr = np.asarray(samples, dtype=np.float64)
    for q in (50, 95, 99):
        got = win.percentile(q)
        assert got is not None
        np.testing.assert_allclose(got, np.percentile(arr, q), rtol=1e-6, atol=1e-6)


# 用例: 超过 maxlen 时只保留最近样本
# 参数: 无
def test_latency_window_evicts_old_samples() -> None:
    win = LatencyWindow(maxlen=3)
    for x in (1.0, 2.0, 3.0, 100.0):
        win.record(x)
    # 窗口应为 [2, 3, 100]；P50 落在中位数 3
    got = win.percentile(50)
    assert got is not None
    np.testing.assert_allclose(got, 3.0, atol=1e-6)


# 用例: 便捷别名 p50/p95/p99
# 参数: 无
def test_latency_window_named_helpers() -> None:
    win = LatencyWindow(maxlen=5)
    for x in (10.0, 20.0, 30.0, 40.0, 50.0):
        win.record(x)
    assert win.p50() == win.percentile(50)
    assert win.p95() == win.percentile(95)
    assert win.p99() == win.percentile(99)
