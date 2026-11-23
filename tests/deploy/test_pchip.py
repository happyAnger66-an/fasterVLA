"""模块: fastervla.deploy.postprocess.pchip
功能: 无 scipy 依赖的 PCHIP 求值。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.deploy.postprocess.pchip import pchip_eval


# 用例: 线性数据上 PCHIP 在采样点与中点贴合直线
# 参数: 无
def test_pchip_eval_linear_data() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([[0.0], [1.0], [2.0], [3.0]])
    for t in (0.0, 0.5, 1.5, 3.0):
        v = pchip_eval(x, y, t)
        assert v.shape == (1,)
        assert float(v[0]) == pytest.approx(t, abs=1e-9)


# 用例: 单调台阶不产生过冲
# 参数: 无
def test_pchip_eval_no_overshoot_on_step() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([[0.0], [0.0], [1.0], [1.0], [1.0]])
    for t in np.linspace(0.0, 4.0, 41):
        v = float(pchip_eval(x, y, float(t))[0])
        assert -1e-9 <= v <= 1.0 + 1e-9
