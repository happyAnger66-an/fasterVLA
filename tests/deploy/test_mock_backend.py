"""模块: fastervla.deploy.backends.mock
功能: 可复现假策略（正弦 / 固定种子）；实现 PolicyBackend.sample。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.core.types import Observation, Prefix
from fastervla.deploy.backends.mock import MockBackend


def _obs(seed: int = 0) -> Observation:
    rng = np.random.default_rng(seed)
    return Observation(
        images={
            "base_0_rgb": b"a",
            "left_wrist_0_rgb": b"b",
            "right_wrist_0_rgb": b"c",
        },
        image_ts=0.0,
        state=rng.normal(size=24).astype(np.float32),
        state_ts=0.0,
        prompt="demo",
    )


# 用例: warmup 可重复调用；sample 返回 n 条 Chunk，waypoints [H,D]
# 参数: 无
def test_mock_backend_sample_shape() -> None:
    backend = MockBackend(horizon=30, action_dim=24, seed=7, mode="sine")
    backend.warmup()
    chunks = backend.sample(
        obs=_obs(),
        prefix=None,
        rtc_mode="NONE",
        n=2,
    )
    assert len(chunks) == 2
    for c in chunks:
        assert c.waypoints.shape == (30, 24)
        assert np.isfinite(c.waypoints).all()


# 用例: 相同 seed + 相同观测 → 可复现
# 参数: 无
def test_mock_backend_is_deterministic() -> None:
    a = MockBackend(horizon=10, action_dim=24, seed=42, mode="sine")
    b = MockBackend(horizon=10, action_dim=24, seed=42, mode="sine")
    obs = _obs(1)
    ca = a.sample(obs=obs, prefix=None, rtc_mode="NONE", n=1)[0]
    cb = b.sample(obs=obs, prefix=None, rtc_mode="NONE", n=1)[0]
    np.testing.assert_allclose(ca.waypoints, cb.waypoints)


# 用例: 不支持的 rtc_mode 在 sample / 构造时报错（F0 默认 NONE）
# 参数: 无
def test_mock_backend_rejects_unsupported_rtc_mode() -> None:
    backend = MockBackend(horizon=30, action_dim=24, seed=0, mode="sine")
    with pytest.raises((ValueError, NotImplementedError)):
        backend.sample(obs=_obs(), prefix=None, rtc_mode="PREFILL", n=1)


# 用例: 有 Prefix 时返回的 waypoints 不含前缀段（有效后缀长度 = H - prefix_len）
# 参数: 无；F0 允许 Mock 选择「忽略 prefix」或「裁后缀」——若实现裁后缀则锁此行为
def test_mock_backend_with_prefix_returns_suffix_or_full() -> None:
    backend = MockBackend(horizon=30, action_dim=24, seed=0, mode="sine")
    prefix = Prefix(actions=np.zeros((4, 24), dtype=np.float32), anchor_step=0)
    chunk = backend.sample(
        obs=_obs(),
        prefix=prefix,
        rtc_mode="NONE",
        n=1,
    )[0]
    # F0：至少保证绝对量、有限、维正确；后缀长度 ∈ {26, 30}
    assert chunk.waypoints.shape[1] == 24
    assert chunk.waypoints.shape[0] in (26, 30)
    assert np.isfinite(chunk.waypoints).all()
