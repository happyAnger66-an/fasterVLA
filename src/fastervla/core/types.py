"""Core deploy/training value types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# F0 / G2 badminton651 policy action width used by Chunk shape checks.
DEFAULT_ACTION_DIM = 24


@dataclass
class Observation:
    images: dict[str, bytes]
    image_ts: float
    state: np.ndarray
    state_ts: float
    prompt: str = ""


@dataclass
class Prefix:
    actions: np.ndarray
    anchor_step: int


@dataclass
class Chunk:
    """Absolute joint waypoints for one policy chunk, shape [H, D]."""

    waypoints: np.ndarray
    model_dt: float
    obs_ts: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        w = np.asarray(self.waypoints)
        if w.ndim != 2 or w.shape[0] < 1 or w.shape[1] != DEFAULT_ACTION_DIM:
            raise ValueError(
                f"Chunk.waypoints must have shape [H, {DEFAULT_ACTION_DIM}], got {w.shape}"
            )
        self.waypoints = w


@dataclass
class TimedChunk:
    chunk: Chunk
    step_dt: np.ndarray
    speed: np.ndarray

    def __post_init__(self) -> None:
        h = int(self.chunk.waypoints.shape[0])
        step_dt = np.asarray(self.step_dt, dtype=np.float64)
        speed = np.asarray(self.speed, dtype=np.float64)
        if step_dt.shape != (h,):
            raise ValueError(
                f"TimedChunk.step_dt length must equal horizon {h}, got {step_dt.shape}"
            )
        if speed.shape != (h,):
            raise ValueError(
                f"TimedChunk.speed length must equal horizon {h}, got {speed.shape}"
            )
        self.step_dt = step_dt
        self.speed = speed


@dataclass
class RobotState:
    q: np.ndarray
    qd: np.ndarray
    timestamp: float
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotCommand:
    q_des: np.ndarray
    qd_des: np.ndarray | None
    timestamp: float
