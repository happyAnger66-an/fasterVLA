"""Agibot G2 GDK platform adapter (F1).

Unit tests inject a fake ``robot`` object. A thin wrapper can map the real
``agibot_gdk.Robot`` API onto the same methods later without changing callers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from fastervla.core.clock import monotonic_s
from fastervla.core.types import RobotCommand, RobotState
from fastervla.deploy.client.io.interfaces import (
    CommandSink,
    ImageSource,
    RobotPlatform,
    StateSource,
)
from fastervla.deploy.client.io.layout import G2_ACTION_DIM, G2_COMMAND_DIM

REQUIRED_ROBOT_AID = "G2A0004C205445"

ERROR_FIELDS = (
    "right_arm_error",
    "left_arm_error",
    "right_end_error",
    "left_end_error",
    "waist_error",
    "lift_error",
    "neck_error",
    "chassis_error",
)


@dataclass
class AgibotG2Config:
    robot_aid: str = REQUIRED_ROBOT_AID
    expected_joint_names: tuple[str, ...] = field(default_factory=tuple)
    camera_names: tuple[str, ...] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    control_period_s: float = 0.02
    read_timeout_s: float = 0.05
    servo_timeout_s: float = 0.05
    joint_stall_timeout_s: float = 0.15
    arm_clip_tolerance: float = 0.02
    gripper_clip_tolerance: float = 0.01


class _G2StateSource(StateSource):
    def __init__(self, owner: "AgibotG2Platform") -> None:
        self._owner = owner

    def read(self) -> RobotState:
        return self._owner._read_state()

    def close(self) -> None:
        return None


class _G2CommandSink(CommandSink):
    def __init__(self, owner: "AgibotG2Platform") -> None:
        self._owner = owner

    def send(self, cmd: RobotCommand) -> None:
        self._owner._send_command(cmd)

    def close(self) -> None:
        return None


class _G2ImageSource(ImageSource):
    def __init__(self, owner: "AgibotG2Platform") -> None:
        self._owner = owner

    def read(self) -> dict[str, bytes]:
        return self._owner._read_images()

    def drop_frame(self) -> None:
        return None

    def close(self) -> None:
        return None


class AgibotG2Platform(RobotPlatform):
    def __init__(
        self,
        *,
        robot: Any,
        config: AgibotG2Config,
        mode: str = "shadow",
    ) -> None:
        if mode not in ("shadow", "execute"):
            raise ValueError(f"unknown mode={mode!r}")
        aid = getattr(robot, "aid", None)
        if aid != config.robot_aid:
            raise ValueError(f"robot_aid mismatch: expected {config.robot_aid!r}, got {aid!r}")
        self.robot = robot
        self.config = config
        self.mode = mode
        self._state_src = _G2StateSource(self)
        self._cmd_sink = _G2CommandSink(self)
        self._img_src = _G2ImageSource(self)
        self._last_stamp_ms: int | None = None
        self._stamp_advanced_at = monotonic_s()
        self._locked_control_mode: int | None = None
        self._locked_mode: int | None = None
        self._limits_lo, self._limits_hi = robot.get_joint_limits()
        self._limits_lo = np.asarray(self._limits_lo, dtype=np.float64).reshape(-1)
        self._limits_hi = np.asarray(self._limits_hi, dtype=np.float64).reshape(-1)
        if self._limits_lo.shape[0] != G2_COMMAND_DIM:
            raise ValueError("get_joint_limits must return length-21 vectors")

    @property
    def state(self) -> StateSource:
        return self._state_src

    @property
    def command(self) -> CommandSink:
        return self._cmd_sink

    @property
    def images(self) -> ImageSource:
        return self._img_src

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of GDK soft limits for the 21-D command layout."""
        return self._limits_lo.copy(), self._limits_hi.copy()

    def close(self) -> None:
        return None

    def _read_state(self) -> RobotState:
        started = monotonic_s()
        status = self.robot.get_whole_body_status()
        elapsed = monotonic_s() - started
        if elapsed > self.config.read_timeout_s:
            raise TimeoutError(
                f"get_whole_body_status exceeded {self.config.read_timeout_s}s ({elapsed:.3f}s)"
            )

        for name in ERROR_FIELDS:
            if getattr(status, name, None) != 0:
                raise RuntimeError(f"robot health fault: {name}={getattr(status, name, None)!r}")

        control_mode = int(status.control_mode)
        mode = int(status.mode)
        if control_mode not in (1, 3) or mode not in (0, 1):
            raise RuntimeError(f"unsupported control_mode/mode: {control_mode}/{mode}")
        if self._locked_control_mode is None:
            self._locked_control_mode = control_mode
            self._locked_mode = mode
        elif control_mode != self._locked_control_mode or mode != self._locked_mode:
            raise RuntimeError("control_mode/mode changed during session")

        joints = list(status.joint_states)
        names = [j.name for j in joints]
        expected = set(self.config.expected_joint_names)
        if set(names) != expected:
            raise RuntimeError("joint name set mismatch")

        # Timestamp: use max joint timestamp_ms as source stamp.
        # Brief regressions are common with multi-joint max(); clamp and rely on stall.
        stamps = [int(j.timestamp_ms) for j in joints]
        stamp = max(stamps) if stamps else 0
        now = monotonic_s()
        if self._last_stamp_ms is not None and stamp < self._last_stamp_ms:
            stamp = self._last_stamp_ms
        if self._last_stamp_ms is None or stamp > self._last_stamp_ms:
            self._last_stamp_ms = stamp
            self._stamp_advanced_at = now
        else:
            # Clock went backwards (unit-test monkeypatch) → re-anchor stall timer.
            if now < self._stamp_advanced_at:
                self._stamp_advanced_at = now
            if now - self._stamp_advanced_at > self.config.joint_stall_timeout_s:
                raise RuntimeError("joint timestamp stalled")

        # End-effector / gripper checks.
        for end in status.end_states:
            if not bool(end.enable) or int(end.err) != 0:
                raise RuntimeError(f"gripper unhealthy: {end.name}")

        # Build 24-d policy state: 22 joints + 2 grippers (by end_states order).
        q = np.zeros(G2_ACTION_DIM, dtype=np.float64)
        qd = np.zeros(G2_ACTION_DIM, dtype=np.float64)
        effort = np.full(G2_ACTION_DIM, np.nan, dtype=np.float64)
        by_name = {j.name: j for j in joints}
        for i, name in enumerate(self.config.expected_joint_names):
            j = by_name[name]
            q[i] = float(j.position)
            v = float(getattr(j, "velocity", 0.0) or 0.0)
            qd[i] = v if np.isfinite(v) else 0.0
            effort[i] = float(getattr(j, "effort", np.nan))
        if len(status.end_states) >= 2:
            q[22] = float(status.end_states[0].position)
            q[23] = float(status.end_states[1].position)

        return RobotState(
            q=q,
            qd=qd,
            timestamp=now,
            extras={
                "stamp_ms": stamp,
                "wall_ms": time.time() * 1000.0,
                "read_ms": elapsed * 1000.0,
                "effort": effort,
            },
        )

    def _read_images(self) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for name in self.config.camera_names:
            data = self.robot.get_latest_image(name)
            out[name] = bytes(data)
        return out

    def _send_command(self, cmd: RobotCommand) -> None:
        if self.mode == "shadow":
            return
        q = np.asarray(cmd.q_des, dtype=np.float64).reshape(-1)
        if q.shape[0] != G2_COMMAND_DIM:
            raise ValueError(f"command dim {q.shape[0]} != {G2_COMMAND_DIM}")

        clipped = np.clip(q, self._limits_lo, self._limits_hi)
        excess = np.abs(q - clipped)
        # Last two command dims are grippers (policy 22,23).
        arm_excess = excess[:-2] if excess.shape[0] >= 2 else excess
        grip_excess = excess[-2:] if excess.shape[0] >= 2 else np.zeros(0)
        if (
            np.any(arm_excess > self.config.arm_clip_tolerance)
            or np.any(grip_excess > self.config.gripper_clip_tolerance)
        ):
            i = int(np.argmax(excess))
            raise RuntimeError(
                "command exceeds joint limits beyond clip tolerance: "
                f"dim={i} q={float(q[i]):+.6f} "
                f"lim=[{float(self._limits_lo[i]):+.6f}, {float(self._limits_hi[i]):+.6f}] "
                f"excess={float(excess[i]):.6f}"
            )

        req = SimpleNamespace(
            joint_positions=clipped.tolist(),
            control_period=float(self.config.control_period_s),
        )
        started = monotonic_s()
        rc = int(self.robot.joint_servo_control(req))
        elapsed = monotonic_s() - started
        # Never retry: one call only.
        if elapsed > self.config.servo_timeout_s:
            raise TimeoutError(
                f"joint_servo_control exceeded {self.config.servo_timeout_s}s ({elapsed:.3f}s)"
            )
        if rc != 0:
            raise RuntimeError(f"joint_servo_control failed rc={rc}")
