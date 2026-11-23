"""Infer request/response pack format (stdlib JSON + base64; F0, no msgpack dep)."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fastervla.core.types import Observation, Prefix


@dataclass
class RtcPayload:
    prev_chunk_left_over: np.ndarray
    inference_delay: int
    execution_horizon: int


@dataclass
class InferRequest:
    observation: Observation
    prefix: Prefix | None
    rtc: RtcPayload | None
    kappa: float
    request_ts: float
    rtc_mode: str = "NONE"


@dataclass
class InferResponse:
    waypoints: np.ndarray
    step_dt: np.ndarray
    speed: np.ndarray
    raw_waypoints: np.ndarray
    timing: dict[str, float]
    retime_status: str
    meta: dict[str, Any] = field(default_factory=dict)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _arr(a: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(a)
    return {"dtype": str(arr.dtype), "shape": list(arr.shape), "data": arr.tolist()}


def _unarr(obj: dict[str, Any]) -> np.ndarray:
    return np.asarray(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])


def _pack_obs(obs: Observation) -> dict[str, Any]:
    return {
        "images": {k: _b64(v) for k, v in obs.images.items()},
        "image_ts": float(obs.image_ts),
        "state": _arr(obs.state),
        "state_ts": float(obs.state_ts),
        "prompt": obs.prompt,
    }


def _unpack_obs(obj: dict[str, Any]) -> Observation:
    return Observation(
        images={k: _unb64(v) for k, v in obj["images"].items()},
        image_ts=float(obj["image_ts"]),
        state=_unarr(obj["state"]),
        state_ts=float(obj["state_ts"]),
        prompt=str(obj.get("prompt") or ""),
    )


def pack_request(req: InferRequest) -> bytes:
    body: dict[str, Any] = {
        "observation": _pack_obs(req.observation),
        "prefix": None,
        "rtc": None,
        "kappa": float(req.kappa),
        "request_ts": float(req.request_ts),
        "rtc_mode": str(req.rtc_mode),
    }
    if req.prefix is not None:
        body["prefix"] = {
            "actions": _arr(req.prefix.actions),
            "anchor_step": int(req.prefix.anchor_step),
        }
    if req.rtc is not None:
        body["rtc"] = {
            "prev_chunk_left_over": _arr(req.rtc.prev_chunk_left_over),
            "inference_delay": int(req.rtc.inference_delay),
            "execution_horizon": int(req.rtc.execution_horizon),
        }
    return json.dumps(body).encode("utf-8")


def unpack_request(raw: bytes) -> InferRequest:
    body = json.loads(raw.decode("utf-8"))
    prefix = None
    if body.get("prefix") is not None:
        p = body["prefix"]
        prefix = Prefix(actions=_unarr(p["actions"]), anchor_step=int(p["anchor_step"]))
    rtc = None
    if body.get("rtc") is not None:
        r = body["rtc"]
        rtc = RtcPayload(
            prev_chunk_left_over=_unarr(r["prev_chunk_left_over"]),
            inference_delay=int(r["inference_delay"]),
            execution_horizon=int(r["execution_horizon"]),
        )
    return InferRequest(
        observation=_unpack_obs(body["observation"]),
        prefix=prefix,
        rtc=rtc,
        kappa=float(body["kappa"]),
        request_ts=float(body["request_ts"]),
        rtc_mode=str(body.get("rtc_mode") or "NONE"),
    )


def pack_response(resp: InferResponse) -> bytes:
    body = {
        "waypoints": _arr(resp.waypoints),
        "step_dt": _arr(resp.step_dt),
        "speed": _arr(resp.speed),
        "raw_waypoints": _arr(resp.raw_waypoints),
        "timing": dict(resp.timing),
        "retime_status": str(resp.retime_status),
        "meta": dict(resp.meta),
    }
    return json.dumps(body).encode("utf-8")


def unpack_response(raw: bytes) -> InferResponse:
    body = json.loads(raw.decode("utf-8"))
    if "waypoints" not in body:
        raise KeyError("missing waypoints")
    return InferResponse(
        waypoints=_unarr(body["waypoints"]),
        step_dt=_unarr(body["step_dt"]),
        speed=_unarr(body["speed"]),
        raw_waypoints=_unarr(body["raw_waypoints"]),
        timing=dict(body.get("timing") or {}),
        retime_status=str(body.get("retime_status") or ""),
        meta=dict(body.get("meta") or {}),
    )
