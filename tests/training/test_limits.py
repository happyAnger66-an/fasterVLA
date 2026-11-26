"""模块: fastervla.training.data.limits
功能: 从 YAML 加载关节速度上限，并按 scale / ignore 维得到有效 v_max。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from fastervla.training.data.limits import (
    effective_v_max,
    load_joint_velocity_limits,
)


# 用例: 合法 YAML 能读出名称、v_max、scale
# 参数: tmp_limits_yaml 是临时合法配置文件路径
def test_load_joint_velocity_limits_reads_fields(tmp_limits_yaml: Path) -> None:
    limits = load_joint_velocity_limits(tmp_limits_yaml)
    assert limits.joint_names == ["j0", "j1", "j2"]
    np.testing.assert_allclose(limits.v_max, [1.0, 2.0, 0.5])
    assert limits.velocity_scale == 1.0
    assert limits.ignore_velocity_dims == []
    # Defaults when YAML omits gripper_no_skip.
    assert limits.gripper_no_skip.enabled is True
    assert limits.gripper_no_skip.window_s == 1.5
    assert limits.gripper_no_skip.dims == [22, 23]


# 用例: YAML 可覆盖夹爪 no-skip 窗口
# 参数: tmp_path 写带 gripper_no_skip 的配置
def test_load_gripper_no_skip_override(tmp_path: Path) -> None:
    path = tmp_path / "grip.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "joint_names": ["j0"],
                "v_max": [1.0],
                "velocity_scale": 1.0,
                "gripper_no_skip": {
                    "enabled": False,
                    "dims": [0],
                    "delta": 0.05,
                    "window_s": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )
    limits = load_joint_velocity_limits(path)
    assert limits.gripper_no_skip.enabled is False
    assert limits.gripper_no_skip.dims == [0]
    assert limits.gripper_no_skip.delta == 0.05
    assert limits.gripper_no_skip.window_s == 0.75


# 用例: 缺少必填键时报错
# 参数: tmp_path 用于写残缺 YAML；missing 是被删掉的键名
@pytest.mark.parametrize("missing", ["joint_names", "v_max", "velocity_scale"])
def test_load_joint_velocity_limits_missing_key_raises(
    tmp_path: Path, missing: str
) -> None:
    data = {
        "joint_names": ["j0"],
        "v_max": [1.0],
        "velocity_scale": 0.9,
        "ignore_velocity_dims": [],
    }
    del data[missing]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises((KeyError, ValueError)):
        load_joint_velocity_limits(path)


# 用例: joint_names 与 v_max 长度不一致时报错
# 参数: tmp_path 写长度不匹配的 YAML
def test_load_joint_velocity_limits_length_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "joint_names": ["j0", "j1"],
                "v_max": [1.0],
                "velocity_scale": 1.0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises((KeyError, ValueError)):
        load_joint_velocity_limits(path)


# 用例: effective_v_max = velocity_scale * v_max
# 参数: tmp_limits_yaml；scale 覆盖配置中的系数
@pytest.mark.parametrize("scale", [0.8, 0.9, 1.0])
def test_effective_v_max_applies_scale(tmp_limits_yaml: Path, scale: float) -> None:
    limits = load_joint_velocity_limits(tmp_limits_yaml)
    limits.velocity_scale = scale
    got = effective_v_max(limits)
    np.testing.assert_allclose(got, scale * np.asarray([1.0, 2.0, 0.5]))


# 用例: ignore_velocity_dims 对应维在有效上限里视为无穷大
# 参数: tmp_limits_yaml；忽略第 1 维（j1）
def test_effective_v_max_ignores_configured_dims(tmp_limits_yaml: Path) -> None:
    limits = load_joint_velocity_limits(tmp_limits_yaml)
    limits.ignore_velocity_dims = [1]
    got = effective_v_max(limits)
    assert np.isinf(got[1])
    np.testing.assert_allclose(got[0], 1.0)
    np.testing.assert_allclose(got[2], 0.5)
