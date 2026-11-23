"""模块: fastervla.deploy.client.io.agibot_g2
功能: 智元 G2 GDK 适配；用假 SDK 覆盖 §2 拒绝路径；shadow 下可读、execute 才可发。

其它本体另建 ``test_<embodiment>_*.py``。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fastervla.core.types import RobotCommand
from fastervla.deploy.client.io.agibot_g2 import (
    AgibotG2Config,
    AgibotG2Platform,
    REQUIRED_ROBOT_AID,
)
from fastervla.deploy.client.io.interfaces import (
    CommandSink,
    ImageSource,
    RobotPlatform,
    StateSource,
)
from fastervla.deploy.client.io.layout import COMMAND_INDICES, G2_ACTION_DIM


def _ok_joint_names() -> list[str]:
    # 22 body joints + 2 grippers mapped via COMMAND path; names must match schema set.
    return [f"j{i}" for i in range(22)]


def _healthy_status(*, ts_ms: int = 1_000, control_mode: int = 1, mode: int = 0) -> SimpleNamespace:
    errors = {
        "right_arm_error": 0,
        "left_arm_error": 0,
        "right_end_error": 0,
        "left_end_error": 0,
        "waist_error": 0,
        "lift_error": 0,
        "neck_error": 0,
        "chassis_error": 0,
    }
    joints = [
        SimpleNamespace(name=n, position=0.01 * i, timestamp_ms=ts_ms)
        for i, n in enumerate(_ok_joint_names())
    ]
    end_states = [
        SimpleNamespace(name="idx31_gripper_l_inner_joint1", enable=True, err=0, position=0.0),
        SimpleNamespace(name="idx71_gripper_r_inner_joint1", enable=True, err=0, position=0.0),
    ]
    return SimpleNamespace(
        control_mode=control_mode,
        mode=mode,
        joint_states=joints,
        end_states=end_states,
        **errors,
    )


class FakeGdkRobot:
    """Minimal stand-in for agibot_gdk robot used by AgibotG2Platform tests."""

    def __init__(self) -> None:
        self.aid = REQUIRED_ROBOT_AID
        self._status = _healthy_status()
        self._limits_lo = np.full(len(COMMAND_INDICES), -2.0)
        self._limits_hi = np.full(len(COMMAND_INDICES), 2.0)
        self.servo_calls: list[Any] = []
        self.servo_delay_s = 0.0
        self.servo_rc = 0
        self.read_delay_s = 0.0
        self._cameras = {
            "base_0_rgb": b"\xff\xd8base",
            "left_wrist_0_rgb": b"\xff\xd8left",
            "right_wrist_0_rgb": b"\xff\xd8right",
        }

    def get_whole_body_status(self) -> SimpleNamespace:
        if self.read_delay_s:
            import time

            time.sleep(self.read_delay_s)
        return self._status

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._limits_lo.copy(), self._limits_hi.copy()

    def joint_servo_control(self, req: Any) -> int:
        import time

        if self.servo_delay_s:
            time.sleep(self.servo_delay_s)
        self.servo_calls.append(req)
        return int(self.servo_rc)

    def get_latest_image(self, camera_name: str) -> bytes:
        return self._cameras[camera_name]


@pytest.fixture
def fake_robot() -> FakeGdkRobot:
    return FakeGdkRobot()


@pytest.fixture
def g2_cfg() -> AgibotG2Config:
    return AgibotG2Config(
        robot_aid=REQUIRED_ROBOT_AID,
        expected_joint_names=tuple(_ok_joint_names()),
        camera_names=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        control_period_s=0.02,
        read_timeout_s=0.05,
        servo_timeout_s=0.05,
        joint_stall_timeout_s=0.15,
        arm_clip_tolerance=0.02,
        gripper_clip_tolerance=0.01,
    )


# 用例: AgibotG2Platform 实现 IO 抽象
# 参数: 无
def test_g2_platform_is_robot_platform(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="shadow")
    assert isinstance(plat, RobotPlatform)
    assert isinstance(plat.state, StateSource)
    assert isinstance(plat.command, CommandSink)
    assert isinstance(plat.images, ImageSource)
    plat.close()


# 用例: robot_aid 不符拒绝启动
# 参数: 无
def test_g2_rejects_wrong_robot_aid(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config) -> None:
    fake_robot.aid = "WRONG_AID"
    with pytest.raises((ValueError, RuntimeError)):
        AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="shadow")


# 用例: 健康字段非 0 拒绝读状态
# 参数: field 是 ERROR_FIELDS 之一
@pytest.mark.parametrize(
    "field",
    [
        "right_arm_error",
        "left_arm_error",
        "waist_error",
        "chassis_error",
    ],
)
def test_g2_rejects_nonzero_health_field(
    fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config, field: str
) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="shadow")
    setattr(fake_robot._status, field, 1)
    with pytest.raises((ValueError, RuntimeError)):
        plat.state.read()
    plat.close()


# 用例: control_mode / mode 运行中变化拒绝
# 参数: 无
def test_g2_rejects_mode_change(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="shadow")
    plat.state.read()  # lock initial modes
    fake_robot._status.control_mode = 3
    with pytest.raises((ValueError, RuntimeError)):
        plat.state.read()
    plat.close()


# 用例: 关节名集合不匹配拒绝
# 参数: 无
def test_g2_rejects_joint_name_mismatch(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="shadow")
    fake_robot._status.joint_states[0].name = "not_in_schema"
    with pytest.raises((ValueError, RuntimeError)):
        plat.state.read()
    plat.close()


# 用例: 关节时间戳短暂回退时钳位并通过（stall 另测）
# 参数: 无
def test_g2_clamps_timestamp_regression(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="shadow")
    plat.state.read()
    for j in fake_robot._status.joint_states:
        j.timestamp_ms = 900  # earlier than first read
    st = plat.state.read()
    assert st is not None
    plat.close()


# 用例: 关节时间戳停滞超过 150 ms 拒绝
# 参数: 无
def test_g2_rejects_timestamp_stall(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config, monkeypatch: pytest.MonkeyPatch) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="shadow")
    plat.state.read()
    # Same timestamp_ms on next read, but wall clock advanced past stall budget.
    t0 = {"v": 0.0}

    def fake_mono() -> float:
        return t0["v"]

    monkeypatch.setattr("fastervla.deploy.client.io.agibot_g2.monotonic_s", fake_mono)
    t0["v"] = 0.0
    plat.state.read()
    t0["v"] = 0.20  # > 150 ms, timestamps unchanged
    with pytest.raises((ValueError, RuntimeError)):
        plat.state.read()
    plat.close()


# 用例: 单次读状态超过 50 ms 拒绝（不重试）
# 参数: 无
def test_g2_rejects_slow_status_read(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="shadow")
    fake_robot.read_delay_s = 0.06
    with pytest.raises((TimeoutError, ValueError, RuntimeError)):
        plat.state.read()
    plat.close()


# 用例: shadow 模式 send 不得调用 joint_servo_control
# 参数: 无
def test_g2_shadow_send_does_not_servo(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="shadow")
    cmd = RobotCommand(
        q_des=np.zeros(len(COMMAND_INDICES), dtype=np.float64),
        qd_des=None,
        timestamp=0.0,
    )
    plat.command.send(cmd)
    assert fake_robot.servo_calls == []
    plat.close()


# 用例: execute 模式 servo 超 50 ms 立即失败且不重试
# 参数: 无
def test_g2_execute_servo_timeout_no_retry(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="execute")
    fake_robot.servo_delay_s = 0.06
    cmd = RobotCommand(
        q_des=np.zeros(len(COMMAND_INDICES), dtype=np.float64),
        qd_des=None,
        timestamp=0.0,
    )
    with pytest.raises((TimeoutError, ValueError, RuntimeError)):
        plat.command.send(cmd)
    assert len(fake_robot.servo_calls) == 1  # attempted once, no retry
    plat.close()


# 用例: execute 模式 servo 返回非 0 立即失败且不重试
# 参数: 无
def test_g2_execute_servo_nonzero_rc_no_retry(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="execute")
    fake_robot.servo_rc = 7
    cmd = RobotCommand(
        q_des=np.zeros(len(COMMAND_INDICES), dtype=np.float64),
        qd_des=None,
        timestamp=0.0,
    )
    with pytest.raises((ValueError, RuntimeError)):
        plat.command.send(cmd)
    assert len(fake_robot.servo_calls) == 1
    plat.close()


# 用例: 超限位先 clip，再按 tolerance 报违规（臂 0.02 / 夹爪 0.01）
# 参数: 无
def test_g2_clips_and_reports_limit_violation(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="execute")
    q = np.zeros(len(COMMAND_INDICES), dtype=np.float64)
    q[5] = 3.0  # far beyond +2.0 limit (> arm_clip_tolerance)
    cmd = RobotCommand(q_des=q, qd_des=None, timestamp=0.0)
    with pytest.raises((ValueError, RuntimeError)):
        plat.command.send(cmd)
    plat.close()


# 用例: 健康路径读状态得到 24 维；三路 JPEG 图像键齐全
# 参数: 无
def test_g2_healthy_read_state_and_images(fake_robot: FakeGdkRobot, g2_cfg: AgibotG2Config) -> None:
    plat = AgibotG2Platform(robot=fake_robot, config=g2_cfg, mode="shadow")
    st = plat.state.read()
    assert st.q.shape == (G2_ACTION_DIM,)
    imgs = plat.images.read()
    assert set(imgs) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    for v in imgs.values():
        assert isinstance(v, (bytes, bytearray)) and v.startswith(b"\xff\xd8")
    plat.close()
