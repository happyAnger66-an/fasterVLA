"""模块: fastervla.deploy.client.io.layout（智元 G2）
功能: G2 策略 24 维 ↔ 命令 21 维映射；腰 baseline 保持；头不下发。

其它本体另建 ``test_<embodiment>_layout.py``，勿把机型常量塞进通用 IO 测试。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.deploy.client.io.layout import (
    ARM_POLICY_SLICE,
    COMMAND_INDICES,
    G2_ACTION_DIM,
    G2_COMMAND_DIM,
    LayoutMapper,
)


# 用例: COMMAND_INDICES = [0..4, 8..23]，长度 21
# 参数: 无
def test_command_indices_match_g2_schema() -> None:
    expected = np.asarray([*range(5), *range(8, 24)], dtype=np.int64)
    np.testing.assert_array_equal(COMMAND_INDICES, expected)
    assert G2_ACTION_DIM == 24
    assert G2_COMMAND_DIM == 21
    assert len(COMMAND_INDICES) == G2_COMMAND_DIM


# 用例: 进 L2/L3 的臂维是策略下标 8..21（14 维）
# 参数: 无
def test_arm_policy_slice_is_14_dims() -> None:
    start, stop = ARM_POLICY_SLICE
    assert stop - start == 14
    assert (start, stop) == (8, 22)


# 用例: from_policy 按 COMMAND_INDICES 抽取，腰用 baseline 覆盖 [0:5]
# 参数: 无
def test_from_policy_holds_waist_baseline() -> None:
    mapper = LayoutMapper()
    baseline = np.arange(24, dtype=np.float64) * 0.01
    action = np.linspace(1.0, 2.0, 24, dtype=np.float64)
    cmd = mapper.from_policy(action, waist_baseline=baseline[:5])
    assert cmd.q_des.shape == (21,)
    # 命令前 5 维对应策略 0..4，但腰必须是 baseline
    np.testing.assert_allclose(cmd.q_des[:5], baseline[:5])
    # 其余 16 维对应策略 8..23
    np.testing.assert_allclose(cmd.q_des[5:], action[8:24])


# 用例: 未提供 waist_baseline 时，腰维取 action[0:5]（仍按下发索引）
# 参数: 无
def test_from_policy_without_baseline_uses_action_waist() -> None:
    mapper = LayoutMapper()
    action = np.arange(24, dtype=np.float64)
    cmd = mapper.from_policy(action, waist_baseline=None)
    np.testing.assert_allclose(cmd.q_des, action[COMMAND_INDICES])


# 用例: to_policy 对 24 维状态保形（策略布局）
# 参数: 无
def test_to_policy_identity_on_policy_layout_state() -> None:
    mapper = LayoutMapper()
    q = np.linspace(-1.0, 1.0, 24, dtype=np.float64)
    out = mapper.to_policy(q)
    np.testing.assert_allclose(out, q)
    assert out.dtype == np.float32 or out.dtype == np.float64


# 用例: from_policy 拒绝错误维数
# 参数: 无
def test_from_policy_rejects_bad_dim() -> None:
    mapper = LayoutMapper()
    with pytest.raises((ValueError, TypeError)):
        mapper.from_policy(np.zeros(21), waist_baseline=None)
