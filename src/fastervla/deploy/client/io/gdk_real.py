"""Real ``agibot_gdk`` session adapted to the FakeGdkRobot surface used by AgibotG2Platform.

Right gripper health is currently known-bad on this unit; shadow runs can ignore it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from fastervla.deploy.client.io.agibot_g2 import ERROR_FIELDS, REQUIRED_ROBOT_AID
from fastervla.deploy.client.io.layout import COMMAND_INDICES, G2_COMMAND_DIM

# Policy body/head/arm names (22) + native gripper joint names for limits/servo.
G2_JOINT_NAMES: tuple[str, ...] = (
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
)
G2_FINGER_NAMES: tuple[str, ...] = (
    "idx31_gripper_l_inner_joint1",
    "idx71_gripper_r_inner_joint1",
)
COMMAND_JOINT_NAMES: tuple[str, ...] = tuple(
    G2_JOINT_NAMES[i] if i < 22 else G2_FINGER_NAMES[i - 22] for i in COMMAND_INDICES.tolist()
)

CAMERA_ATTRS: tuple[tuple[str, str], ...] = (
    ("base_0_rgb", "kHeadColor"),
    ("left_wrist_0_rgb", "kHandLeftColor"),
    ("right_wrist_0_rgb", "kHandRightColor"),
)


@dataclass
class GdkEnvConfig:
    sdk_root: str = "/home/agi/app/gdk"
    sdk_source_ip: str = "10.42.1.101"
    discovery_uri: str = "http://10.42.1.101:2379"
    robot_aid: str = REQUIRED_ROBOT_AID
    # Legacy g2s: right end-state bus can block; when True, skip get_end_state and
    # leave grip obs at last value (often 0). h7 / healthy G2 must keep False.
    ignore_right_gripper_health: bool = False


def apply_acados_process_env(
    acados_root: str = "/home/agi/deps/acados",
) -> None:
    """If a local acados build exists, export paths needed by ``acados_template``."""
    root = Path(acados_root)
    lib = root / "lib"
    if not (lib / "libacados.so").is_file() and not any(lib.glob("libacados.so*")):
        return
    os.environ.setdefault("ACADOS_SOURCE_DIR", str(root))
    lib_s = str(lib)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_s not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_s}:{cur}" if cur else lib_s
    # pip --user puts ``cython`` here; needed to build the cython OCP wrapper.
    local_bin = str(Path.home() / ".local" / "bin")
    path = os.environ.get("PATH", "")
    if local_bin not in path.split(":"):
        os.environ["PATH"] = f"{local_bin}:{path}" if path else local_bin


def apply_gdk_process_env(cfg: GdkEnvConfig) -> None:
    """Set GDK env vars. Caller should re-exec once so LD_LIBRARY_PATH is visible to dlopen."""
    root = Path(cfg.sdk_root)
    lib = str(root / "lib")
    dep = str(root / "build_dep/cpp/aarch64/lib")
    os.environ["ROBOT_SDK_ROOT"] = str(root)
    os.environ["LOCATOR_IP"] = cfg.sdk_source_ip
    os.environ["AORTA_DISCOVERY_URI"] = cfg.discovery_uri
    os.environ["PYTHONPATH"] = f"{lib}:" + os.environ.get("PYTHONPATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        f"/home/agi/app/lib:/home/agi/app/lib/hal:{lib}:{dep}:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    apply_acados_process_env()
    conf = root / "config/app_conf.yml"
    if conf.is_file():
        os.environ["APP_CONF_PATH"] = str(conf)
    if lib not in sys.path:
        sys.path.insert(0, lib)


def reexec_with_gdk_env(cfg: GdkEnvConfig, *, flag: str = "_FASTERVLA_GDK_READY") -> None:
    if os.environ.get(flag) == "1":
        return
    apply_gdk_process_env(cfg)
    os.environ[flag] = "1"
    os.execv(sys.executable, [sys.executable, "-u", *sys.argv])


def _opt_float(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        v = row.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return float("nan")


class RealGdkRobot:
    """Maps live ``agibot_gdk.Robot`` + ``Camera`` onto the unit-test robot surface."""

    def __init__(self, *, sdk: Any, robot: Any, camera: Any, config: GdkEnvConfig) -> None:
        self._sdk = sdk
        self._robot = robot
        self._camera = camera
        self._cfg = config
        self.aid = str(sdk.get_robot_aid())
        self._camera_types = {
            name: getattr(sdk.CameraType, attr) for name, attr in CAMERA_ATTRS
        }
        self._last_grip = [0.0, 0.0]

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        limits = self._robot.get_joint_limits()
        lo = np.zeros(G2_COMMAND_DIM, dtype=np.float64)
        hi = np.zeros(G2_COMMAND_DIM, dtype=np.float64)
        for i, name in enumerate(COMMAND_JOINT_NAMES):
            entry = limits[name]
            lo[i] = float(entry["min"])
            hi[i] = float(entry["max"])
        return lo, hi

    def get_whole_body_status(self) -> SimpleNamespace:
        health = self._robot.get_whole_body_status()
        if not isinstance(health, dict):
            raise RuntimeError("get_whole_body_status did not return a dict")
        motion = self._robot.get_motion_control_status()
        joints = self._robot.get_joint_states()
        rows = joints.get("states", [])
        stamp_ns = int(joints.get("timestamp") or 0)
        stamp_ms = stamp_ns // 1_000_000 if stamp_ns > 0 else 0

        joint_states = [
            SimpleNamespace(
                name=str(row.get("name")),
                position=float(row.get("motor_position")),
                velocity=_opt_float(row, "motor_velocity", "velocity"),
                effort=_opt_float(row, "motor_current", "motor_effort", "effort"),
                timestamp_ms=stamp_ms,
                error_code=int(row.get("error_code") or 0),
            )
            for row in rows
        ]

        end_states = self._read_end_states()

        # Overlay health fields; optionally force right_end_error clean for shadow.
        fields = {name: int(health.get(name, 0) or 0) for name in ERROR_FIELDS}
        if self._cfg.ignore_right_gripper_health:
            fields["right_end_error"] = 0

        return SimpleNamespace(
            control_mode=int(motion.control_mode),
            mode=int(motion.mode),
            joint_states=joint_states,
            end_states=end_states,
            **fields,
        )

    def get_latest_image(self, camera_name: str) -> bytes:
        cam_type = self._camera_types[camera_name]
        image = self._camera.get_latest_image(cam_type, 500.0)
        if image is None:
            raise RuntimeError(f"no image for {camera_name}")
        raw = np.asarray(image.data, dtype=np.uint8).reshape(-1)
        encoding = str(getattr(image, "encoding", ""))
        if "JPEG" not in encoding.upper() and not (
            raw.size >= 2 and int(raw[0]) == 0xFF and int(raw[1]) == 0xD8
        ):
            raise RuntimeError(
                f"expected JPEG for {camera_name}, encoding={encoding!r} size={raw.size}"
            )
        return bytes(raw.tobytes())

    def joint_servo_control(self, req: Any) -> int:
        request = self._sdk.JointServoControlReq()
        request.joint_names = list(COMMAND_JOINT_NAMES)
        request.joint_positions = list(req.joint_positions)
        request.joint_velocities = []
        request.control_period = float(getattr(req, "control_period", 0.02))
        return int(self._robot.joint_servo_control(request, False))

    def close(self) -> None:
        return None

    def _read_end_states(self) -> list[SimpleNamespace]:
        # Right end-state bus is currently broken on this unit and can block ~1s
        # before raising; skip the GDK call entirely when ignoring gripper health.
        if self._cfg.ignore_right_gripper_health:
            return [
                SimpleNamespace(
                    name=G2_FINGER_NAMES[0],
                    enable=True,
                    err=0,
                    position=float(self._last_grip[0]),
                ),
                SimpleNamespace(
                    name=G2_FINGER_NAMES[1],
                    enable=True,
                    err=0,
                    position=float(self._last_grip[1]),
                ),
            ]

        ends = self._robot.get_end_state()
        out: list[SimpleNamespace] = []
        for i, side in enumerate(("left", "right")):
            group = ends.get(side + "_end_state", {}) or {}
            values = group.get("end_states", []) or []
            native = G2_FINGER_NAMES[i]
            if not values:
                pos = float(self._last_grip[i])
                enable, err = True, 0
            else:
                pos = float(values[0].get("position") or 0.0)
                enable = bool(values[0].get("enable", True))
                err = int(values[0].get("err_code") or 0)
                self._last_grip[i] = pos
            out.append(SimpleNamespace(name=native, enable=enable, err=err, position=pos))
        return out


def open_real_gdk(config: GdkEnvConfig | None = None) -> tuple[Any, RealGdkRobot]:
    """gdk_init + Robot/Camera; returns (sdk_module, RealGdkRobot)."""
    cfg = config or GdkEnvConfig()
    apply_gdk_process_env(cfg)
    import agibot_gdk as sdk  # noqa: WPS433 — must import after env/path setup

    if sdk.gdk_init() != sdk.GDKRes.kSuccess:
        raise RuntimeError("gdk_init failed")
    try:
        aid = str(sdk.get_robot_aid())
        if aid != cfg.robot_aid:
            raise ValueError(f"robot_aid mismatch: expected {cfg.robot_aid!r}, got {aid!r}")
        robot = sdk.Robot()
        camera = sdk.Camera()
        return sdk, RealGdkRobot(sdk=sdk, robot=robot, camera=camera, config=cfg)
    except BaseException:
        sdk.gdk_release()
        raise


def close_real_gdk(sdk: Any) -> None:
    if sdk is not None:
        sdk.gdk_release()
