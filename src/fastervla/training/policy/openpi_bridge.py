"""Hooks for openpi/LeRobot data loading: long action windows + velocity skip."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Mapping
from typing import Protocol
from typing import Sequence

import numpy as np

from fastervla.training.data.limits import JointVelocityLimits
from fastervla.training.data.limits import effective_v_max
from fastervla.training.resample.velocity_skip import build_skip_chunk


def make_action_delta_timestamps(
    action_keys: Sequence[str],
    fps: float,
    window: int,
) -> dict[str, list[float]]:
    """Build LeRobot-style delta_timestamps for action sequence keys.

    Offsets are nominal ``frame_index / fps`` seconds relative to the current
    sample, matching openpi's dense-horizon construction.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    offsets = [t / float(fps) for t in range(int(window))]
    return {str(key): list(offsets) for key in action_keys}


@dataclass
class VelocitySkipConfig:
    """Config for replacing a long action window with an H-step skipped chunk."""

    skip_frames: bool
    horizon: int
    window: int
    v_lim: np.ndarray
    action_key: str = "actions"
    dt: float | None = None

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.window < self.horizon:
            raise ValueError(
                f"window ({self.window}) must be >= horizon ({self.horizon})"
            )
        self.v_lim = np.asarray(self.v_lim, dtype=np.float64).reshape(-1)

    @classmethod
    def from_limits(
        cls,
        limits: JointVelocityLimits,
        *,
        horizon: int,
        window: int,
        skip_frames: bool,
        action_key: str = "actions",
        dt: float | None = None,
    ) -> VelocitySkipConfig:
        return cls(
            skip_frames=skip_frames,
            horizon=horizon,
            window=window,
            v_lim=effective_v_max(limits),
            action_key=action_key,
            dt=dt,
        )


class VelocitySkipActions:
    """Transform: map actions[window, D] -> actions[horizon, D] via velocity skip."""

    def __init__(self, config: VelocitySkipConfig) -> None:
        self._cfg = config

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        key = self._cfg.action_key
        if key not in data:
            raise KeyError(f"sample missing action key {key!r}")

        actions = np.asarray(data[key], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[0] < self._cfg.window:
            raise ValueError(
                f"actions length {actions.shape[0]} < configured window {self._cfg.window}"
            )

        window = actions[: self._cfg.window]
        if self._cfg.skip_frames:
            chunk = build_skip_chunk(
                window,
                h0=0,
                horizon=self._cfg.horizon,
                v_lim=self._cfg.v_lim,
                dt=self._cfg.dt,
                skip_frames=True,
            )
        else:
            chunk = window[: self._cfg.horizon]

        out = dict(data)
        out[key] = chunk
        return out


@dataclass(frozen=True)
class TrainStepResult:
    """Outcome of a single injected training step."""

    loss: float
    params_updated: bool


class TrainBackend(Protocol):
    """Minimal openpi-facing backend: one forward/backward/update step."""

    def train_step(self, batch: Mapping[str, Any]) -> float:
        """Run one train step on a sample/batch; return scalar loss."""


def run_one_train_step(
    *,
    dataset: Any,
    skip_cfg: VelocitySkipConfig,
    backend: TrainBackend,
    index: int = 0,
) -> TrainStepResult:
    """Take one dataset sample, apply velocity-skip, run one backend train step.

    Real openpi JAX can be injected as ``backend``; unit tests use a fake.
    """
    sample = dataset[int(index)]
    if not isinstance(sample, Mapping):
        raise TypeError(
            f"dataset sample must be a mapping, got {type(sample).__name__}"
        )

    batch = VelocitySkipActions(skip_cfg)(sample)

    has_param = hasattr(backend, "param")
    before = getattr(backend, "param", None) if has_param else None
    loss = float(backend.train_step(batch))
    if has_param:
        params_updated = getattr(backend, "param") != before
    else:
        params_updated = True

    return TrainStepResult(loss=loss, params_updated=params_updated)
