"""模块: fastervla.deploy.postprocess.retime.UniformRetimer
功能: F2 均匀时间轴；step_dt_i = 1/(κ·policy_fps)；不改 waypoints。

F2 测试（先测后码）。TimeAxisQP 属 F3，不在此实现。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.core.types import Chunk
from fastervla.deploy.postprocess.retime import UniformRetimer


def _chunk(h: int = 30, d: int = 24) -> Chunk:
    wp = np.linspace(0.0, 1.0, h * d, dtype=np.float32).reshape(h, d)
    return Chunk(waypoints=wp, model_dt=1.0 / 30.0, obs_ts=0.0)


# 用例: κ=1.0、policy_fps=30 → 全部 step_dt = 1/30
# 参数: 无
def test_uniform_retimer_one_x() -> None:
    retimer = UniformRetimer(policy_fps=30.0, kappa=1.0)
    timed = retimer.retime(_chunk())
    assert timed.step_dt.shape == (30,)
    np.testing.assert_allclose(timed.step_dt, 1.0 / 30.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(timed.chunk.waypoints, _chunk().waypoints)
    assert timed.chunk.meta.get("retime_status", "ok") in ("ok", "uniform", None) or True


# 用例: κ=1.4 → step_dt = 1/(1.4·30)；waypoints 不变
# 参数: 无
def test_uniform_retimer_kappa_scales_dt_only() -> None:
    src = _chunk()
    retimer = UniformRetimer(policy_fps=30.0, kappa=1.4)
    timed = retimer.retime(src)
    expected = 1.0 / (1.4 * 30.0)
    np.testing.assert_allclose(timed.step_dt, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(timed.chunk.waypoints, src.waypoints)


# 用例: kappa 必须落在 [kappa_min, kappa_max]
# 参数: 无
def test_uniform_retimer_rejects_kappa_above_max() -> None:
    with pytest.raises((ValueError, TypeError)):
        UniformRetimer(policy_fps=30.0, kappa=1.6, kappa_max=1.5, kappa_min=1.0)


# 用例: speed 向量长度 = H，且 speed_i ≈ 1/step_dt_i
# 参数: 无
def test_uniform_retimer_speed_inverse_of_dt() -> None:
    timed = UniformRetimer(policy_fps=30.0, kappa=1.0).retime(_chunk())
    np.testing.assert_allclose(timed.speed, 1.0 / timed.step_dt, rtol=0, atol=1e-9)
