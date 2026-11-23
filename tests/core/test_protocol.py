"""模块: fastervla.core.protocol
功能: InferRequest / InferResponse 与 openpi websocket 字段映射；编解码往返无损。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.core.protocol import (
    InferRequest,
    InferResponse,
    RtcPayload,
    pack_request,
    pack_response,
    unpack_request,
    unpack_response,
)
from fastervla.core.types import Observation, Prefix


def _sample_obs() -> Observation:
    return Observation(
        images={
            "base_0_rgb": b"\xff\xd8base",
            "left_wrist_0_rgb": b"\xff\xd8left",
            "right_wrist_0_rgb": b"\xff\xd8right",
        },
        image_ts=1.0,
        state=np.arange(24, dtype=np.float32),
        state_ts=1.001,
        prompt="hit shuttle",
    )


# 用例: 无 RTC 的 InferRequest 打包再解包，字段一致
# 参数: 无
def test_infer_request_roundtrip_without_rtc() -> None:
    req = InferRequest(
        observation=_sample_obs(),
        prefix=None,
        rtc=None,
        kappa=1.0,
        request_ts=12.34,
        rtc_mode="NONE",
    )
    raw = pack_request(req)
    got = unpack_request(raw)
    assert got.kappa == 1.0
    assert got.rtc_mode == "NONE"
    assert got.observation.prompt == "hit shuttle"
    np.testing.assert_array_equal(got.observation.state, req.observation.state)
    assert got.observation.images["base_0_rgb"] == b"\xff\xd8base"
    assert got.prefix is None
    assert got.rtc is None


# 用例: 带 Prefix + RTC INPAINT 载荷往返
# 参数: 无
def test_infer_request_roundtrip_with_rtc_inpaint() -> None:
    prefix = Prefix(actions=np.ones((5, 24), dtype=np.float32), anchor_step=3)
    leftover = np.linspace(0.0, 1.0, 30 * 24, dtype=np.float32).reshape(30, 24)
    rtc = RtcPayload(
        prev_chunk_left_over=leftover,
        inference_delay=4,
        execution_horizon=25,
    )
    req = InferRequest(
        observation=_sample_obs(),
        prefix=prefix,
        rtc=rtc,
        kappa=1.2,
        request_ts=0.5,
        rtc_mode="INPAINT",
    )
    got = unpack_request(pack_request(req))
    assert got.rtc_mode == "INPAINT"
    assert got.prefix is not None
    assert got.prefix.anchor_step == 3
    assert got.rtc is not None
    assert got.rtc.inference_delay == 4
    assert got.rtc.execution_horizon == 25
    np.testing.assert_allclose(got.rtc.prev_chunk_left_over, leftover)


# 用例: InferResponse 含 waypoints / step_dt / timing / retime_status
# 参数: 无
def test_infer_response_roundtrip() -> None:
    h, d = 30, 24
    resp = InferResponse(
        waypoints=np.zeros((h, d), dtype=np.float32),
        step_dt=np.full(h, 1.0 / 30.0, dtype=np.float64),
        speed=np.ones(h, dtype=np.float64),
        raw_waypoints=np.zeros((h, d), dtype=np.float32),
        timing={"encode_ms": 1.0, "sample_ms": 2.0, "retime_ms": 0.5},
        retime_status="ok",
        meta={"rtc_applied": True},
    )
    got = unpack_response(pack_response(resp))
    assert got.retime_status == "ok"
    assert got.meta["rtc_applied"] is True
    np.testing.assert_allclose(got.step_dt, resp.step_dt)
    np.testing.assert_array_equal(got.waypoints, resp.waypoints)


# 用例: 残缺响应（缺 waypoints）解包失败
# 参数: 无
def test_unpack_response_missing_waypoints_raises() -> None:
    with pytest.raises((KeyError, ValueError, TypeError)):
        unpack_response(b"{}")
