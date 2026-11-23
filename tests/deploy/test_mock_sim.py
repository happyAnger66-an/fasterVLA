"""模块: fastervla.deploy.client.io.mock_sim
功能: Mock 平台 + 一阶滞后被控对象；无硬件跑通 IO 契约。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.core.types import RobotCommand
from fastervla.deploy.client.io.interfaces import (
    CommandSink,
    ImageSource,
    RobotPlatform,
    StateSource,
)
from fastervla.deploy.client.io.mock_sim import (
    MockCommandSink,
    MockImageSource,
    MockRobotPlatform,
    MockStateSource,
    FirstOrderLagPlant,
)


# 用例: Mock* 实现对应抽象接口
# 参数: 无
def test_mock_classes_are_io_interfaces() -> None:
    assert issubclass(MockStateSource, StateSource)
    assert issubclass(MockCommandSink, CommandSink)
    assert issubclass(MockImageSource, ImageSource)
    assert issubclass(MockRobotPlatform, RobotPlatform)


# 用例: FirstOrderLagPlant 在常值命令下指数逼近，纯延迟可配
# 参数: 无
def test_first_order_lag_plant_tracks_constant_command() -> None:
    plant = FirstOrderLagPlant(
        dim=21,
        tau_s=0.05,
        pure_delay_s=0.0,
        dt_s=0.01,
        q0=np.zeros(21, dtype=np.float64),
    )
    target = np.full(21, 0.5, dtype=np.float64)
    for _ in range(200):
        plant.step(target)
    np.testing.assert_allclose(plant.q, target, atol=1e-3)


# 用例: 纯延迟使输出在 delay 内保持初值
# 参数: 无
def test_first_order_lag_plant_respects_pure_delay() -> None:
    plant = FirstOrderLagPlant(
        dim=3,
        tau_s=0.01,
        pure_delay_s=0.05,
        dt_s=0.01,
        q0=np.zeros(3, dtype=np.float64),
    )
    target = np.ones(3, dtype=np.float64)
    for _ in range(4):  # 40 ms < 50 ms delay
        plant.step(target)
    np.testing.assert_allclose(plant.q, np.zeros(3), atol=1e-9)


# 用例: MockRobotPlatform 读状态 / 发命令 / 取图像键齐全
# 参数: 无
def test_mock_platform_read_send_images() -> None:
    plat = MockRobotPlatform(
        state_dim=24,
        command_dim=21,
        camera_names=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        plant_tau_s=0.05,
        plant_delay_s=0.0,
        seed=0,
    )
    st = plat.state.read()
    assert st.q.shape == (24,)
    assert st.timestamp > 0.0

    imgs = plat.images.read()
    assert set(imgs.keys()) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    for v in imgs.values():
        assert isinstance(v, (bytes, bytearray))
        assert len(v) > 0

    cmd = RobotCommand(q_des=np.zeros(21, dtype=np.float64), qd_des=None, timestamp=st.timestamp)
    plat.command.send(cmd)
    plat.close()


# 用例: drop_frame 可调用且不抛错（清积压语义）
# 参数: 无
def test_mock_image_source_drop_frame() -> None:
    src = MockImageSource(
        camera_names=("base_0_rgb",),
        seed=1,
    )
    src.read()
    src.drop_frame()
    src.close()
