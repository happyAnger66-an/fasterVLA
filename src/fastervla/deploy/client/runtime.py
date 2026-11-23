"""F1 deploy runtime: shadow (default, zero commands) vs execute."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from fastervla.core.types import Observation
from fastervla.deploy.client.io.interfaces import RobotPlatform
from fastervla.deploy.client.io.layout import LayoutMapper


class _PolicyBackend(Protocol):
    def sample(
        self,
        *,
        obs: Observation,
        prefix: Any,
        rtc_mode: str,
        n: int,
    ) -> list[Any]: ...


@dataclass
class DeployRuntimeConfig:
    mode: str = "shadow"
    max_steps: int | None = None
    control_hz: float = 100.0
    policy_fps: float = 30.0
    kappa: float = 1.0
    kappa_max: float = 1.5
    kappa_min: float = 1.0
    horizon: int = 30
    action_dim: int = 24
    command_dim: int = 21

    def __post_init__(self) -> None:
        if self.mode not in ("shadow", "execute"):
            raise ValueError(f"unknown mode={self.mode!r}; expected 'shadow' or 'execute'")
        if not (self.kappa_min <= self.kappa <= self.kappa_max):
            raise ValueError(
                f"kappa={self.kappa} outside [{self.kappa_min}, {self.kappa_max}]"
            )
        if self.control_hz <= 0 or self.policy_fps <= 0:
            raise ValueError("control_hz and policy_fps must be > 0")


@dataclass
class DeployRuntimeSummary:
    steps: int = 0
    infer_calls: int = 0
    commands_sent: int = 0
    limit_violations: int = 0
    elapsed_s: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


def require_execute_tty(*, mode: str) -> None:
    """Physical execute requires an interactive TTY; shadow does not."""
    if mode == "execute" and not sys.stdin.isatty():
        raise RuntimeError("execute mode requires an interactive TTY")


class DeployRuntime:
    """Synchronous F1 loop: always infer; send only when ``mode=execute``."""

    def __init__(
        self,
        *,
        platform: RobotPlatform,
        backend: _PolicyBackend,
        config: DeployRuntimeConfig | None = None,
    ) -> None:
        self.platform = platform
        self.backend = backend
        self.config = config or DeployRuntimeConfig()
        self._mapper = LayoutMapper()

    def run(self) -> DeployRuntimeSummary:
        cfg = self.config
        # TTY gate is caller's responsibility (CLI); see require_execute_tty().

        dt_ctrl = 1.0 / cfg.control_hz
        step_dt = 1.0 / (cfg.kappa * cfg.policy_fps)
        summary = DeployRuntimeSummary()

        # Best-effort align mock plant dt (no-op for real platforms).
        plant = getattr(self.platform, "_plant", None)
        if plant is not None and hasattr(plant, "dt_s"):
            plant.dt_s = dt_ctrl

        state0 = self.platform.state.read()
        waist_baseline = np.asarray(state0.q[:5], dtype=np.float64).copy()

        waypoints: np.ndarray | None = None
        model_progress = 0.0

        q_min = getattr(self.platform, "q_min", None)
        q_max = getattr(self.platform, "q_max", None)

        def _need_chunk() -> bool:
            return waypoints is None or model_progress >= cfg.horizon - 1e-9

        def _fetch_chunk() -> None:
            nonlocal waypoints, model_progress
            st = self.platform.state.read()
            imgs = self.platform.images.read()
            obs = Observation(
                images=imgs,
                image_ts=st.timestamp,
                state=st.q.astype(np.float32),
                state_ts=st.timestamp,
                prompt="",
            )
            chunks = self.backend.sample(obs=obs, prefix=None, rtc_mode="NONE", n=1)
            summary.infer_calls += 1
            waypoints = np.asarray(chunks[0].waypoints, dtype=np.float64)
            model_progress = 0.0

        max_steps = cfg.max_steps
        while True:
            if max_steps is not None and summary.steps >= max_steps:
                break
            if _need_chunk():
                _fetch_chunk()

            assert waypoints is not None
            idx = min(int(model_progress), waypoints.shape[0] - 1)
            action = waypoints[idx]
            cmd = self._mapper.from_policy(
                action, waist_baseline=waist_baseline, timestamp=state0.timestamp
            )

            if cfg.mode == "execute":
                q = cmd.q_des.copy()
                if q_min is not None and q_max is not None:
                    violated = bool(np.any(q < q_min - 1e-9) or np.any(q > q_max + 1e-9))
                    if violated:
                        summary.limit_violations += 1
                    q = np.clip(q, q_min, q_max)
                    cmd.q_des = q
                self.platform.command.send(cmd)
                summary.commands_sent += 1

            summary.steps += 1
            summary.elapsed_s += dt_ctrl
            model_progress += dt_ctrl / step_dt

        return summary
