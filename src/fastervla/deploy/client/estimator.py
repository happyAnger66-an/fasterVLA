"""Lag-state estimator: replay issued commands through a first-order lag plant."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class LagStateEstimatorConfig:
    dim: int = 14
    tau_s: float = 0.057
    dt_s: float = 0.01
    pure_delay_s: float = 0.0
    t_proprio_s: float = 0.0
    anchor_lookback_s: float = 0.05
    command_history_s: float = 1.0

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        if self.tau_s <= 0:
            raise ValueError("tau_s must be > 0")
        if self.dt_s <= 0:
            raise ValueError("dt_s must be > 0")
        if self.pure_delay_s < 0:
            raise ValueError("pure_delay_s must be >= 0")
        if self.anchor_lookback_s < 0:
            raise ValueError("anchor_lookback_s must be >= 0")
        if self.command_history_s <= 0:
            raise ValueError("command_history_s must be > 0")


class LagStateEstimator:
    """Estimate current joint state by replaying commands from a lagged observation.

    Plant matches ``FirstOrderLagPlant``:
    ``q ← q + (dt/τ)·(y_applied − q)`` with optional pure delay on ``y``.
    """

    def __init__(self, config: LagStateEstimatorConfig | None = None) -> None:
        self.config = config or LagStateEstimatorConfig()
        self._cmds: deque[tuple[float, np.ndarray]] = deque()
        self._obs: deque[tuple[float, np.ndarray]] = deque()
        delay_steps = (
            int(round(self.config.pure_delay_s / self.config.dt_s))
            if self.config.dt_s > 0
            else 0
        )
        self._delay_steps = max(0, delay_steps)

    def push_command(self, t: float, y: np.ndarray) -> None:
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if y_arr.shape[0] != self.config.dim:
            raise ValueError(
                f"command dim {y_arr.shape[0]} != config.dim {self.config.dim}"
            )
        self._cmds.append((float(t), y_arr.copy()))
        self._prune(float(t))

    def push_observation(self, t: float, q: np.ndarray) -> None:
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_arr.shape[0] != self.config.dim:
            raise ValueError(
                f"observation dim {q_arr.shape[0]} != config.dim {self.config.dim}"
            )
        # Callers may pass raw stamps; optional proprio shift for hardware clocks.
        t_eff = float(t) - float(self.config.t_proprio_s)
        self._obs.append((t_eff, q_arr.copy()))
        self._prune(t_eff)

    def estimate(self, t_now: float, contact: bool = False) -> np.ndarray:
        if not self._obs:
            raise RuntimeError("LagStateEstimator requires at least one observation")
        t_now = float(t_now)
        # Latest observation (for contact / empty replay).
        t_last, q_last = self._obs[-1]
        if contact:
            return q_last.copy()

        t_anchor, q_anchor = self._select_anchor(t_now)
        cmds_after = [(t, y) for t, y in self._cmds if t_anchor < t <= t_now + 1e-15]
        if not cmds_after:
            return q_anchor.copy()

        q = q_anchor.copy()
        delay_q = self._delay_queue_at(t_anchor, q_anchor)
        alpha = min(1.0, self.config.dt_s / self.config.tau_s)
        for _t, y in cmds_after:
            if self._delay_steps > 0:
                delay_q.append(y.copy())
                applied = delay_q.popleft()
            else:
                applied = y
            q = q + alpha * (applied - q)
        return q

    def _select_anchor(self, t_now: float) -> tuple[float, np.ndarray]:
        t_target = t_now - float(self.config.anchor_lookback_s)
        # Prefer the newest observation at or before the lookback target.
        candidates = [(t, q) for t, q in self._obs if t <= t_target + 1e-15]
        if candidates:
            return candidates[-1]
        # Not enough history: use earliest observation still <= t_now.
        for t, q in self._obs:
            if t <= t_now + 1e-15:
                return t, q
        return self._obs[0]

    def _delay_queue_at(self, t_obs: float, q_obs: np.ndarray) -> deque[np.ndarray]:
        """Rebuild pure-delay FIFO to match ``FirstOrderLagPlant`` after ``t_obs``."""
        n = self._delay_steps
        if n <= 0:
            return deque()
        # Replay from cold start so queue state matches the plant exactly.
        # Seed with the earliest observation (or q_obs) like plant seeds with q0.
        t0, q0 = self._obs[0]
        dq: deque[np.ndarray] = deque(maxlen=n + 1)
        for _ in range(n):
            dq.append(q0.copy())
        alpha = min(1.0, self.config.dt_s / self.config.tau_s)
        q = q0.copy()
        for t, y in self._cmds:
            if t > t_obs + 1e-15:
                break
            dq.append(y.copy())
            applied = dq.popleft()
            q = q + alpha * (applied - q)
        # ``q`` should ≈ q_obs; keep plant queue (length n after popleft).
        _ = (t0, q_obs)
        return dq

    def _prune(self, t_now: float) -> None:
        cutoff = t_now - float(self.config.command_history_s)
        while self._cmds and self._cmds[0][0] < cutoff - 1e-15:
            self._cmds.popleft()
        while self._obs and self._obs[0][0] < cutoff - 1e-15:
            # Keep at least one observation.
            if len(self._obs) <= 1:
                break
            self._obs.popleft()
