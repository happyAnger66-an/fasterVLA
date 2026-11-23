"""F0 synchronous 1× control loop (no async inference thread)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fastervla.core.types import Observation
from fastervla.deploy.backends.mock import MockBackend
from fastervla.deploy.client.io.layout import LayoutMapper
from fastervla.deploy.client.io.mock_sim import MockRobotPlatform


@dataclass
class SyncLoopConfig:
    control_hz: float = 100.0
    policy_fps: float = 30.0
    kappa: float = 1.0
    kappa_max: float = 1.5
    kappa_min: float = 1.0
    horizon: int = 30
    max_steps: int | None = None
    action_dim: int = 24
    command_dim: int = 21
    stop_after_one_chunk: bool = False

    def __post_init__(self) -> None:
        if not (self.kappa_min <= self.kappa <= self.kappa_max):
            raise ValueError(
                f"kappa={self.kappa} outside [{self.kappa_min}, {self.kappa_max}]"
            )
        if self.control_hz <= 0 or self.policy_fps <= 0:
            raise ValueError("control_hz and policy_fps must be > 0")


@dataclass
class SyncLoopSummary:
    steps: int = 0
    commands_sent: int = 0
    elapsed_s: float = 0.0
    max_abs_command: float = 0.0
    limit_violations: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class SyncControlLoop:
    """Pull one chunk from backend, play it at κ·policy_fps on a mock/real platform."""

    def __init__(
        self,
        *,
        platform: MockRobotPlatform | None,
        backend: MockBackend | None,
        config: SyncLoopConfig | None = None,
    ) -> None:
        if platform is None or backend is None:
            raise ValueError("SyncControlLoop requires platform and backend")
        self.platform = platform
        self.backend = backend
        self.config = config or SyncLoopConfig()
        self._mapper = LayoutMapper()

    def run(self) -> SyncLoopSummary:
        cfg = self.config
        dt_ctrl = 1.0 / cfg.control_hz
        step_dt = 1.0 / (cfg.kappa * cfg.policy_fps)
        summary = SyncLoopSummary()

        # Align mock plant dt with control tick.
        self.platform._plant.dt_s = dt_ctrl  # noqa: SLF001 — F0 mock coupling

        state0 = self.platform.state.read()
        waist_baseline = state0.q[:5].copy()

        waypoints: np.ndarray | None = None
        model_progress = 0.0  # fractional model step
        chunk_sim_t = 0.0
        chunk_total_t = cfg.horizon * step_dt

        def _need_chunk() -> bool:
            return waypoints is None or model_progress >= cfg.horizon - 1e-9

        def _fetch_chunk() -> None:
            nonlocal waypoints, model_progress, chunk_sim_t, chunk_total_t
            st = self.platform.state.read()
            imgs = self.platform.images.read()
            obs = Observation(
                images=imgs,
                image_ts=st.timestamp,
                state=st.q.astype(np.float32),
                state_ts=st.timestamp,
                prompt="",
            )
            chunks = self.backend.sample(
                obs=obs, prefix=None, rtc_mode="NONE", n=1
            )
            waypoints = np.asarray(chunks[0].waypoints, dtype=np.float64)
            model_progress = 0.0
            chunk_sim_t = 0.0
            chunk_total_t = float(waypoints.shape[0]) * step_dt

        max_steps = cfg.max_steps
        while True:
            if max_steps is not None and summary.steps >= max_steps:
                break
            if _need_chunk():
                if cfg.stop_after_one_chunk and summary.commands_sent > 0:
                    break
                _fetch_chunk()

            assert waypoints is not None
            idx = min(int(model_progress), waypoints.shape[0] - 1)
            action = waypoints[idx]
            cmd = self._mapper.from_policy(
                action, waist_baseline=waist_baseline, timestamp=state0.timestamp
            )

            # Clip + count violations against platform limits.
            q = cmd.q_des.copy()
            lo = self.platform.q_min
            hi = self.platform.q_max
            violated = bool(np.any(q < lo - 1e-9) or np.any(q > hi + 1e-9))
            if violated:
                summary.limit_violations += 1
            q = np.clip(q, lo, hi)
            cmd.q_des = q
            self.platform.command.send(cmd)

            summary.steps += 1
            summary.commands_sent += 1
            summary.max_abs_command = max(
                summary.max_abs_command, float(np.max(np.abs(q)))
            )
            summary.elapsed_s += dt_ctrl

            # Advance plan by one control tick in model-step units.
            model_progress += dt_ctrl / step_dt
            chunk_sim_t += dt_ctrl

            if cfg.stop_after_one_chunk and chunk_sim_t + 1e-12 >= chunk_total_t:
                # Snap elapsed to exact one-chunk duration for κ timing contract.
                summary.elapsed_s = chunk_total_t
                break

        return summary
