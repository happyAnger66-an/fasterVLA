"""模块: fastervla.deploy.client.runtime（F1 shadow）
功能: 默认 shadow；execute 需显式开启；shadow 零下发但仍读状态/图像/推理。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fastervla.core.types import RobotCommand
from fastervla.deploy.backends.mock import MockBackend
from fastervla.deploy.client.io.layout import COMMAND_INDICES
from fastervla.deploy.client.io.mock_sim import MockRobotPlatform
from fastervla.deploy.client.runtime import (
    DeployRuntime,
    DeployRuntimeConfig,
    require_execute_tty,
)


class _CountingSinkPlatform(MockRobotPlatform):
    """Mock platform that counts command.send calls."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.send_count = 0
        self._orig_send = self._send_command

        def counted(cmd: RobotCommand) -> None:
            self.send_count += 1
            self._orig_send(cmd)

        self._send_command = counted  # type: ignore[method-assign]


# 用例: 默认 mode 为 shadow
# 参数: 无
def test_runtime_default_mode_is_shadow() -> None:
    cfg = DeployRuntimeConfig()
    assert cfg.mode == "shadow"


# 用例: 非法 mode 拒绝
# 参数: 无
def test_runtime_rejects_unknown_mode() -> None:
    with pytest.raises((ValueError, TypeError)):
        DeployRuntimeConfig(mode="dry-run")


# 用例: shadow 跑若干拍：有推理调用，但 platform 下发计数为 0
# 参数: 无
def test_shadow_runtime_zero_commands() -> None:
    plat = _CountingSinkPlatform(
        state_dim=24,
        command_dim=21,
        camera_names=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        seed=0,
    )
    backend = MockBackend(horizon=30, action_dim=24, seed=0, mode="sine")
    rt = DeployRuntime(
        platform=plat,
        backend=backend,
        config=DeployRuntimeConfig(mode="shadow", max_steps=50, control_hz=100.0),
    )
    summary = rt.run()
    assert summary.steps == 50
    assert summary.infer_calls >= 1
    assert summary.commands_sent == 0
    assert plat.send_count == 0
    plat.close()


# 用例: execute 模式会真正 send（mock 平台）
# 参数: 无
def test_execute_runtime_sends_commands() -> None:
    plat = _CountingSinkPlatform(
        state_dim=24,
        command_dim=21,
        camera_names=("base_0_rgb",),
        seed=1,
    )
    backend = MockBackend(horizon=30, action_dim=24, seed=1, mode="sine")
    rt = DeployRuntime(
        platform=plat,
        backend=backend,
        config=DeployRuntimeConfig(mode="execute", max_steps=30, control_hz=100.0),
    )
    summary = rt.run()
    assert summary.commands_sent == 30
    assert plat.send_count == 30
    plat.close()


# 用例: execute 启动前必须通过 TTY 检查（非 TTY 拒绝）
# 参数: 无
def test_execute_requires_interactive_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises((RuntimeError, ValueError, SystemExit)):
        require_execute_tty(mode="execute")


# 用例: shadow 不要求 TTY
# 参数: 无
def test_shadow_skips_tty_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    require_execute_tty(mode="shadow")  # must not raise


# 用例: shadow 下即使调用方构造了命令对象，runtime 也不得下发越限位命令
# 参数: 无（契约：shadow 路径根本不 send）
def test_shadow_ignores_would_be_commands() -> None:
    plat = _CountingSinkPlatform(
        state_dim=24,
        command_dim=len(COMMAND_INDICES),
        camera_names=("base_0_rgb",),
        seed=2,
        q_min=np.full(21, -0.01),
        q_max=np.full(21, 0.01),
    )
    backend = MockBackend(horizon=30, action_dim=24, seed=2, mode="sine")
    summary = DeployRuntime(
        platform=plat,
        backend=backend,
        config=DeployRuntimeConfig(mode="shadow", max_steps=20, control_hz=100.0),
    ).run()
    assert summary.commands_sent == 0
    assert summary.limit_violations == 0
    plat.close()
