"""OpenPi websocket policy backend (F1).

Unit tests inject a fake ``transport``. Production can wrap openpi-client
websocket I/O behind the same ``get_server_metadata`` / ``infer`` / ``close`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fastervla.core.types import Chunk, Observation, Prefix

_SUPPORTED_RTC = {"NONE", "INPAINT"}


class ServerMetadataError(RuntimeError):
    """Server metadata does not match the expected deploy recipe."""


@dataclass
class OpenPiWsConfig:
    host: str = "10.42.1.201"
    port: int = 8013
    expect_checkpoint_step: int = 89999
    expect_norm_stats_name: str = "badminton651"
    expect_prompt: str = "play badminton"
    expect_num_steps: int = 10
    expect_rtc: bool = True
    expect_action_shape: tuple[int, int] = (30, 24)
    expect_image_transport: str = "jpeg"
    horizon: int = 30
    action_dim: int = 24


class OpenPiWebsocketBackend:
    def __init__(
        self,
        *,
        config: OpenPiWsConfig,
        transport: Any,
    ) -> None:
        self.config = config
        self.transport = transport
        self._warm = False

    def warmup(self) -> None:
        meta = self.transport.get_server_metadata()
        self._validate_metadata(meta)
        self._warm = True

    def sample(
        self,
        *,
        obs: Observation,
        prefix: Prefix | None,
        rtc_mode: str,
        n: int,
        rtc: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not self._warm:
            raise RuntimeError("OpenPiWebsocketBackend.warmup() required before sample()")
        mode = str(rtc_mode).upper()
        if mode not in _SUPPORTED_RTC:
            raise ValueError(f"unsupported rtc_mode={rtc_mode!r}")

        payload = self._encode_obs(obs)
        if mode == "INPAINT":
            if rtc is None:
                raise ValueError("rtc_mode=INPAINT requires rtc payload")
            payload["rtc"] = {
                "prev_chunk_left_over": np.asarray(
                    rtc["prev_chunk_left_over"], dtype=np.float32
                ),
                "inference_delay": int(rtc["inference_delay"]),
                "execution_horizon": int(rtc["execution_horizon"]),
            }
            if prefix is not None:
                payload["prefix"] = {
                    "actions": np.asarray(prefix.actions, dtype=np.float32),
                    "anchor_step": int(prefix.anchor_step),
                }

        out: list[Chunk] = []
        for _ in range(int(n)):
            resp = self.transport.infer(payload)
            waypoints = np.asarray(resp["actions"], dtype=np.float32)
            expected = (self.config.horizon, self.config.action_dim)
            if waypoints.ndim != 2 or tuple(waypoints.shape) != expected:
                raise ValueError(
                    f"actions shape {waypoints.shape} != expected {expected}"
                )
            out.append(
                Chunk(
                    waypoints=waypoints,
                    model_dt=1.0 / 30.0,
                    obs_ts=float(obs.image_ts),
                    meta={
                        "policy_timing": dict(resp.get("policy_timing") or {}),
                        "rtc_applied": bool(resp.get("rtc_applied", False)),
                    },
                )
            )
        return out

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

    def _encode_obs(self, obs: Observation) -> dict[str, Any]:
        state = np.asarray(obs.state, dtype=np.float32).reshape(-1)
        # Match openpi jpeg live contract used by the 5090 badminton651 server.
        if self.config.expect_image_transport == "jpeg":
            compressed = {
                k: np.frombuffer(bytes(v), dtype=np.uint8) for k, v in obs.images.items()
            }
            return {
                "state": state,
                "image_encoding": "jpeg",
                "images_compressed": compressed,
                "prompt": obs.prompt,
            }
        return {
            "state": state,
            "images": {k: bytes(v) for k, v in obs.images.items()},
            "prompt": obs.prompt,
        }

    def _validate_metadata(self, meta: dict[str, Any]) -> None:
        cfg = self.config
        checks: list[tuple[str, Any, Any]] = [
            ("checkpoint_step", meta.get("checkpoint_step"), cfg.expect_checkpoint_step),
            ("norm_stats_name", meta.get("norm_stats_name"), cfg.expect_norm_stats_name),
            ("num_steps", meta.get("num_steps"), cfg.expect_num_steps),
            ("rtc", meta.get("rtc"), cfg.expect_rtc),
            ("image_transport", meta.get("image_transport"), cfg.expect_image_transport),
        ]
        for name, got, expect in checks:
            if got != expect:
                raise ServerMetadataError(f"{name}: got {got!r}, expected {expect!r}")

        shape = meta.get("action_shape")
        expect_shape = list(cfg.expect_action_shape)
        if list(shape) != expect_shape:
            raise ServerMetadataError(
                f"action_shape: got {shape!r}, expected {expect_shape!r}"
            )

        # Prompt is advisory in metadata; mismatch still fails warmup when configured.
        if cfg.expect_prompt and meta.get("prompt") not in (None, cfg.expect_prompt):
            raise ServerMetadataError(
                f"prompt: got {meta.get('prompt')!r}, expected {cfg.expect_prompt!r}"
            )
