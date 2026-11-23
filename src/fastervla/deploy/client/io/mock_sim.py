"""Mock robot platform with a first-order lag plant (no hardware)."""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from fastervla.core.clock import monotonic_s
from fastervla.core.types import RobotCommand, RobotState
from fastervla.deploy.client.io.interfaces import (
    CommandSink,
    ImageSource,
    RobotPlatform,
    StateSource,
)
from fastervla.deploy.client.io.layout import COMMAND_INDICES


class FirstOrderLagPlant:
    """Discrete first-order lag ``x += (dt/tau)*(u-x)`` with optional pure delay."""

    def __init__(
        self,
        *,
        dim: int,
        tau_s: float,
        pure_delay_s: float,
        dt_s: float,
        q0: np.ndarray,
    ) -> None:
        self.dim = int(dim)
        self.tau_s = float(tau_s)
        self.pure_delay_s = float(pure_delay_s)
        self.dt_s = float(dt_s)
        self.q = np.asarray(q0, dtype=np.float64).reshape(self.dim).copy()
        delay_steps = int(round(self.pure_delay_s / self.dt_s)) if self.dt_s > 0 else 0
        self._delay_steps = max(0, delay_steps)
        self._cmd_queue: deque[np.ndarray] = deque(maxlen=self._delay_steps + 1)
        for _ in range(self._delay_steps):
            self._cmd_queue.append(self.q.copy())

    def step(self, u: np.ndarray) -> np.ndarray:
        cmd = np.asarray(u, dtype=np.float64).reshape(self.dim)
        if self._delay_steps > 0:
            self._cmd_queue.append(cmd.copy())
            applied = self._cmd_queue.popleft()
        else:
            applied = cmd
        if self.tau_s <= 0:
            self.q = applied.copy()
        else:
            alpha = min(1.0, self.dt_s / self.tau_s)
            self.q = self.q + alpha * (applied - self.q)
        return self.q


class MockStateSource(StateSource):
    def __init__(self, owner: "MockRobotPlatform") -> None:
        self._owner = owner

    def read(self) -> RobotState:
        return self._owner._read_state()

    def close(self) -> None:
        return None


class MockCommandSink(CommandSink):
    def __init__(self, owner: "MockRobotPlatform") -> None:
        self._owner = owner

    def send(self, cmd: RobotCommand) -> None:
        self._owner._send_command(cmd)

    def close(self) -> None:
        return None


class MockImageSource(ImageSource):
    def __init__(
        self,
        *,
        camera_names: Sequence[str],
        seed: int = 0,
        owner: "MockRobotPlatform | None" = None,
    ) -> None:
        self._names = tuple(camera_names)
        self._rng = np.random.default_rng(seed)
        self._owner = owner
        self._dropped = 0

    def read(self) -> dict[str, bytes]:
        # Tiny fake JPEG-ish payloads (enough for contract tests).
        out: dict[str, bytes] = {}
        for name in self._names:
            noise = int(self._rng.integers(0, 255))
            out[name] = b"\xff\xd8" + bytes([noise]) + name.encode("ascii")
        return out

    def drop_frame(self) -> None:
        self._dropped += 1

    def close(self) -> None:
        return None


class MockRobotPlatform(RobotPlatform):
    def __init__(
        self,
        *,
        state_dim: int = 24,
        command_dim: int = 21,
        camera_names: Sequence[str] = ("base_0_rgb",),
        plant_tau_s: float = 0.05,
        plant_delay_s: float = 0.0,
        seed: int = 0,
        q_min: np.ndarray | None = None,
        q_max: np.ndarray | None = None,
        control_dt_s: float = 0.01,
    ) -> None:
        self.state_dim = int(state_dim)
        self.command_dim = int(command_dim)
        self.q_min = (
            np.asarray(q_min, dtype=np.float64)
            if q_min is not None
            else np.full(self.command_dim, -10.0)
        )
        self.q_max = (
            np.asarray(q_max, dtype=np.float64)
            if q_max is not None
            else np.full(self.command_dim, 10.0)
        )
        self._state_q = np.zeros(self.state_dim, dtype=np.float64)
        self._plant = FirstOrderLagPlant(
            dim=self.command_dim,
            tau_s=plant_tau_s,
            pure_delay_s=plant_delay_s,
            dt_s=control_dt_s,
            q0=np.zeros(self.command_dim, dtype=np.float64),
        )
        self._state_src = MockStateSource(self)
        self._cmd_sink = MockCommandSink(self)
        self._img_src = MockImageSource(camera_names=camera_names, seed=seed, owner=self)
        self._closed = False

    @property
    def state(self) -> StateSource:
        return self._state_src

    @property
    def command(self) -> CommandSink:
        return self._cmd_sink

    @property
    def images(self) -> ImageSource:
        return self._img_src

    def _read_state(self) -> RobotState:
        # Reflect plant into command indices of the full policy state.
        if self.command_dim == len(COMMAND_INDICES) and self.state_dim > max(COMMAND_INDICES):
            self._state_q[COMMAND_INDICES] = self._plant.q
        elif self.command_dim == self.state_dim:
            self._state_q[:] = self._plant.q
        return RobotState(
            q=self._state_q.copy(),
            qd=np.zeros(self.state_dim, dtype=np.float64),
            timestamp=monotonic_s(),
            extras={},
        )

    def _send_command(self, cmd: RobotCommand) -> None:
        q = np.asarray(cmd.q_des, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.command_dim:
            raise ValueError(f"command dim {q.shape[0]} != {self.command_dim}")
        q = np.clip(q, self.q_min, self.q_max)
        self._plant.step(q)

    def close(self) -> None:
        self._closed = True
        self._state_src.close()
        self._cmd_sink.close()
        self._img_src.close()
