"""模块: fastervla.deploy.backends.openpi_ws
功能: OpenPi websocket client backend；启动校验 server metadata；infer 含 RTC 可选载荷。

不改 5090 推理服务；测试用假 transport 注入。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fastervla.core.types import Observation, Prefix
from fastervla.deploy.backends.openpi_ws import (
    OpenPiWebsocketBackend,
    OpenPiWsConfig,
    ServerMetadataError,
)


def _good_metadata() -> dict[str, Any]:
    return {
        "checkpoint_step": 89999,
        "norm_stats_name": "badminton651",
        "prompt": "play badminton",
        "num_steps": 10,
        "rtc": True,
        "action_shape": [30, 24],
        "image_transport": "jpeg",
    }


class FakeWsTransport:
    """Stand-in for openpi websocket client used in unit tests."""

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = metadata if metadata is not None else _good_metadata()
        self.infer_calls: list[dict[str, Any]] = []
        self.closed = False

    def get_server_metadata(self) -> dict[str, Any]:
        return dict(self.metadata)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.infer_calls.append(obs)
        actions = np.zeros((30, 24), dtype=np.float32)
        if "state" in obs:
            actions[0] = np.asarray(obs["state"], dtype=np.float32)
        return {
            "actions": actions,
            "policy_timing": {"infer_ms": 23.0},
            "rtc_applied": bool(obs.get("rtc")),
        }

    def close(self) -> None:
        self.closed = True


def _obs() -> Observation:
    return Observation(
        images={
            "base_0_rgb": b"\xff\xd8b",
            "left_wrist_0_rgb": b"\xff\xd8l",
            "right_wrist_0_rgb": b"\xff\xd8r",
        },
        image_ts=1.0,
        state=np.linspace(0.0, 1.0, 24, dtype=np.float32),
        state_ts=1.0,
        prompt="play badminton",
    )


@pytest.fixture
def ws_cfg() -> OpenPiWsConfig:
    return OpenPiWsConfig(
        host="10.42.1.201",
        port=8013,
        expect_checkpoint_step=89999,
        expect_norm_stats_name="badminton651",
        expect_prompt="play badminton",
        expect_num_steps=10,
        expect_rtc=True,
        expect_action_shape=(30, 24),
        expect_image_transport="jpeg",
        horizon=30,
        action_dim=24,
    )


# 用例: metadata 合法时 warmup 成功
# 参数: 无
def test_openpi_ws_warmup_accepts_good_metadata(ws_cfg: OpenPiWsConfig) -> None:
    transport = FakeWsTransport()
    backend = OpenPiWebsocketBackend(config=ws_cfg, transport=transport)
    backend.warmup()
    backend.close()


# 用例: checkpoint / norm / rtc / num_steps / action_shape 不符 → ServerMetadataError
# 参数: key, bad_value
@pytest.mark.parametrize(
    "key,bad",
    [
        ("checkpoint_step", 1),
        ("norm_stats_name", "other"),
        ("num_steps", 5),
        ("rtc", False),
        ("action_shape", [30, 21]),
        ("image_transport", "rgb"),
    ],
)
def test_openpi_ws_warmup_rejects_bad_metadata(
    ws_cfg: OpenPiWsConfig, key: str, bad: Any
) -> None:
    meta = _good_metadata()
    meta[key] = bad
    backend = OpenPiWebsocketBackend(config=ws_cfg, transport=FakeWsTransport(meta))
    with pytest.raises(ServerMetadataError):
        backend.warmup()


# 用例: infer 把 Observation 编成 state + 三路 JPEG images
# 参数: 无
def test_openpi_ws_infer_payload_images_and_state(ws_cfg: OpenPiWsConfig) -> None:
    transport = FakeWsTransport()
    backend = OpenPiWebsocketBackend(config=ws_cfg, transport=transport)
    backend.warmup()
    chunks = backend.sample(obs=_obs(), prefix=None, rtc_mode="NONE", n=1)
    assert len(chunks) == 1
    assert chunks[0].waypoints.shape == (30, 24)
    assert len(transport.infer_calls) == 1
    payload = transport.infer_calls[0]
    assert "state" in payload and np.asarray(payload["state"]).shape == (24,)
    assert payload.get("image_encoding") == "jpeg"
    assert set(payload["images_compressed"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    for v in payload["images_compressed"].values():
        assert isinstance(v, np.ndarray) and v.dtype == np.uint8
    backend.close()


# 用例: rtc_mode=INPAINT 时附带 rtc 字典（leftover 30x24、inference_delay、execution_horizon）
# 参数: 无
def test_openpi_ws_infer_attaches_rtc_payload(ws_cfg: OpenPiWsConfig) -> None:
    transport = FakeWsTransport()
    backend = OpenPiWebsocketBackend(config=ws_cfg, transport=transport)
    backend.warmup()
    leftover = np.ones((30, 24), dtype=np.float32)
    prefix = Prefix(actions=np.zeros((4, 24), dtype=np.float32), anchor_step=2)
    backend.sample(
        obs=_obs(),
        prefix=prefix,
        rtc_mode="INPAINT",
        n=1,
        rtc={
            "prev_chunk_left_over": leftover,
            "inference_delay": 4,
            "execution_horizon": 25,
        },
    )
    rtc = transport.infer_calls[0]["rtc"]
    assert rtc["inference_delay"] == 4
    assert rtc["execution_horizon"] == 25
    np.testing.assert_array_equal(rtc["prev_chunk_left_over"], leftover)
    backend.close()


# 用例: 不支持的 rtc_mode 硬失败
# 参数: 无
def test_openpi_ws_rejects_prefill(ws_cfg: OpenPiWsConfig) -> None:
    backend = OpenPiWebsocketBackend(config=ws_cfg, transport=FakeWsTransport())
    backend.warmup()
    with pytest.raises((ValueError, NotImplementedError)):
        backend.sample(obs=_obs(), prefix=None, rtc_mode="PREFILL", n=1)
    backend.close()


# 用例: 服务器返回残缺 actions 形状时报错
# 参数: 无
def test_openpi_ws_rejects_bad_action_shape(ws_cfg: OpenPiWsConfig) -> None:
    class BadTransport(FakeWsTransport):
        def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
            return {"actions": np.zeros((10, 24), dtype=np.float32)}

    backend = OpenPiWebsocketBackend(config=ws_cfg, transport=BadTransport())
    backend.warmup()
    with pytest.raises((ValueError, RuntimeError)):
        backend.sample(obs=_obs(), prefix=None, rtc_mode="NONE", n=1)
    backend.close()
