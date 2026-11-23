"""模块: fastervla.deploy.postprocess.arm_chunk_filter
功能: 臂 chunk 零相位 DCT 低通（对齐老栈 soft ZP@3Hz）。
"""

from __future__ import annotations

import numpy as np

from fastervla.deploy.postprocess.arm_chunk_filter import (
    apply_arm_chunk_filter,
    apply_zero_phase_dct,
)


# 用例: mode=none 原样拷贝，meta=None
# 参数: 无
def test_arm_chunk_filter_none_passthrough() -> None:
    actions = np.random.randn(30, 24).astype(np.float32)
    out, meta = apply_arm_chunk_filter(actions, mode="none", fps=30.0)
    assert meta is None
    np.testing.assert_array_equal(out, actions)


# 用例: ZP@3Hz 只改臂 8:22，削弱高频帧间差分；腰/夹爪不变
# 参数: 无
def test_zero_phase_dct_arms_only_attenuates_hf() -> None:
    t = np.arange(30, dtype=np.float64) / 30.0
    actions = np.zeros((30, 24), dtype=np.float32)
    actions[:, 8:22] = (
        0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.05 * np.sin(2 * np.pi * 8.0 * t)
    )[:, None]
    actions[:, 22:] = 0.3
    actions[:, :8] = 0.1
    out, meta = apply_zero_phase_dct(actions, cutoff_hz=3.0, fps=30.0)
    assert meta is not None
    assert meta["mode"] == "zero_phase_dct"
    assert meta["max_delta_rad"] > 0.0
    np.testing.assert_allclose(out[:, :8], actions[:, :8])
    np.testing.assert_allclose(out[:, 22:], actions[:, 22:])
    assert float(np.max(np.abs(np.diff(out[:, 8:22], axis=0)))) < float(
        np.max(np.abs(np.diff(actions[:, 8:22], axis=0)))
    )


# 用例: cross 模式无 prev 时退化为逐 chunk
# 参数: 无
def test_zero_phase_dct_cross_fallback_no_prev() -> None:
    actions = np.random.randn(30, 24).astype(np.float32)
    out, meta = apply_arm_chunk_filter(
        actions, mode="zero_phase_dct_cross", fps=30.0, prev_filtered=None
    )
    assert meta is not None
    assert meta["mode"] == "zero_phase_dct_cross"
    assert meta["cross_fallback"] == "no_prev"
    assert out.shape == actions.shape
