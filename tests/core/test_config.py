"""模块: fastervla.core.config
功能: dataclass 配置加载；未知键 / 缺键必须硬失败。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from fastervla.core.config import from_mapping, load_dataclass_yaml


@dataclass(frozen=True)
class _ToyCfg:
    name: str
    horizon: int
    kappa: float = 1.0


# 用例: 合法映射能构造 dataclass，默认字段可省略
# 参数: 无
def test_from_mapping_accepts_known_keys() -> None:
    cfg = from_mapping(_ToyCfg, {"name": "mock", "horizon": 30})
    assert cfg.name == "mock"
    assert cfg.horizon == 30
    assert cfg.kappa == 1.0


# 用例: 出现未知键时报错（不允许静默忽略）
# 参数: 无
def test_from_mapping_rejects_unknown_key() -> None:
    with pytest.raises((KeyError, TypeError, ValueError)):
        from_mapping(_ToyCfg, {"name": "mock", "horizon": 30, "typo": 1})


# 用例: 缺少无默认值的必填键时报错
# 参数: 无
def test_from_mapping_rejects_missing_required_key() -> None:
    with pytest.raises((KeyError, TypeError, ValueError)):
        from_mapping(_ToyCfg, {"name": "mock"})


# 用例: YAML 文件经 load_dataclass_yaml 与 from_mapping 行为一致
# 参数: tmp_path 写合法 / 非法 YAML
def test_load_dataclass_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "toy.yaml"
    path.write_text(
        yaml.safe_dump({"name": "g2", "horizon": 30, "kappa": 1.2}),
        encoding="utf-8",
    )
    cfg = load_dataclass_yaml(_ToyCfg, path)
    assert cfg.kappa == 1.2


def test_load_dataclass_yaml_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({"name": "g2", "horizon": 30, "extra": True}),
        encoding="utf-8",
    )
    with pytest.raises((KeyError, TypeError, ValueError)):
        load_dataclass_yaml(_ToyCfg, path)
