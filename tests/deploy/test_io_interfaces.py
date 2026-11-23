"""模块: fastervla.deploy.client.io.interfaces
功能: StateSource / CommandSink / ImageSource / RobotPlatform 抽象契约。
"""

from __future__ import annotations

import pytest

from fastervla.deploy.client.io.interfaces import (
    CommandSink,
    ImageSource,
    RobotPlatform,
    StateSource,
)


# 用例: 抽象基类不可直接实例化
# 参数: cls 是接口类型
@pytest.mark.parametrize(
    "cls",
    [StateSource, CommandSink, ImageSource, RobotPlatform],
)
def test_io_interfaces_are_abstract(cls: type) -> None:
    with pytest.raises(TypeError):
        cls()  # type: ignore[call-arg]
