"""模块: fastervla.deploy.client.estimator.LagStateEstimator
功能: 从滞后观测锚点起，用一阶滤波回放已下发命令，得到当前 q̂。

F4 测试（先测后码）。对齐 design §5.2 / deploy-g2-plan §4.5。
被控对象模型与 mock_sim.FirstOrderLagPlant 一致：
  q ← q + (dt/τ)·(y − q)  ⇔  q' = a·q + (1−a)·y，a = 1 − dt/τ。
"""

from __future__ import annotations

import numpy as np
import pytest

from fastervla.deploy.client.estimator import LagStateEstimator, LagStateEstimatorConfig
from fastervla.deploy.client.io.mock_sim import FirstOrderLagPlant


def _cfg(**kwargs: object) -> LagStateEstimatorConfig:
    base = dict(
        dim=14,
        tau_s=0.057,
        dt_s=0.01,
        pure_delay_s=0.0,
        t_proprio_s=0.0,
        anchor_lookback_s=0.05,
        command_history_s=1.0,
    )
    base.update(kwargs)
    return LagStateEstimatorConfig(**base)  # type: ignore[arg-type]


# 用例: 默认参数与 G2 滞后辨识同量级（τ≈57ms，控制拍 10ms，锚点约 50ms）
# 参数: 无
def test_lag_estimator_default_g2_scale() -> None:
    cfg = LagStateEstimatorConfig()
    assert cfg.tau_s == pytest.approx(0.057)
    assert cfg.dt_s == pytest.approx(0.01)
    assert cfg.anchor_lookback_s == pytest.approx(0.05)
    assert cfg.dim == 14


# 用例: τ≤0 或 dim≤0 拒绝
# 参数: 无
def test_lag_estimator_rejects_bad_config() -> None:
    with pytest.raises((ValueError, TypeError)):
        LagStateEstimatorConfig(tau_s=0.0, dt_s=0.01, dim=14)
    with pytest.raises((ValueError, TypeError)):
        LagStateEstimatorConfig(tau_s=0.057, dt_s=0.01, dim=0)


# 用例: 无命令历史时 estimate = 最近观测（已减 t_proprio 语义由调用方保证时间戳）
# 参数: 无
def test_lag_estimator_returns_last_obs_without_commands() -> None:
    est = LagStateEstimator(_cfg())
    q = np.linspace(0.0, 0.13, 14)
    est.push_observation(t=1.0, q=q)
    out = est.estimate(t_now=1.05)
    np.testing.assert_allclose(out, q, atol=1e-12)


# 用例: contact=True 时不外推，直接返回最近观测
# 参数: 无
def test_lag_estimator_contact_holds_observation() -> None:
    est = LagStateEstimator(_cfg(tau_s=0.057))
    q0 = np.zeros(14)
    q1 = np.full(14, 0.2)
    est.push_observation(t=0.0, q=q0)
    for k in range(20):
        t = (k + 1) * 0.01
        est.push_command(t=t, y=np.full(14, 1.0))
    est.push_observation(t=0.2, q=q1)
    out = est.estimate(t_now=0.25, contact=True)
    np.testing.assert_allclose(out, q1, atol=1e-12)


# 用例: 与 FirstOrderLagPlant 同 τ/dt 开环回放，估计误差小于阈值
# 参数: 无
def test_lag_estimator_matches_first_order_plant() -> None:
    tau, dt, dim = 0.057, 0.01, 14
    q0 = np.zeros(dim)
    plant = FirstOrderLagPlant(
        dim=dim, tau_s=tau, pure_delay_s=0.0, dt_s=dt, q0=q0
    )
    est = LagStateEstimator(_cfg(tau_s=tau, dt_s=dt, dim=dim, pure_delay_s=0.0))

    est.push_observation(t=0.0, q=q0.copy())
    rng = np.random.default_rng(0)
    t = 0.0
    for _ in range(80):
        t += dt
        y = rng.normal(scale=0.05, size=dim)
        plant.step(y)
        est.push_command(t=t, y=y)
        # Slow proprio: observe every 5 ticks at plant state.
        if int(round(t / dt)) % 5 == 0:
            est.push_observation(t=t, q=plant.q.copy())

    q_hat = est.estimate(t_now=t)
    err = float(np.max(np.abs(q_hat - plant.q)))
    # Replay from ~50ms-old obs through known plant; should stay tight.
    assert err < 5e-3, f"max |q_hat - q_plant| = {err}"


# 用例: 纯延迟 >0 时，回放须错开 delay_steps 个命令（与 plant 一致）
# 参数: 无
def test_lag_estimator_respects_pure_delay() -> None:
    tau, dt, delay = 0.05, 0.01, 0.03  # 3 steps
    dim = 14
    q0 = np.zeros(dim)
    plant = FirstOrderLagPlant(
        dim=dim, tau_s=tau, pure_delay_s=delay, dt_s=dt, q0=q0
    )
    est = LagStateEstimator(
        _cfg(tau_s=tau, dt_s=dt, dim=dim, pure_delay_s=delay, anchor_lookback_s=0.05)
    )
    est.push_observation(t=0.0, q=q0.copy())
    t = 0.0
    cmd = np.full(dim, 0.5)
    for _ in range(40):
        t += dt
        plant.step(cmd)
        est.push_command(t=t, y=cmd)
        if int(round(t / dt)) % 5 == 0:
            est.push_observation(t=t, q=plant.q.copy())
    q_hat = est.estimate(t_now=t)
    assert float(np.max(np.abs(q_hat - plant.q))) < 8e-3


# 用例: 维数不符的 push 抛错
# 参数: 无
def test_lag_estimator_rejects_wrong_dim() -> None:
    est = LagStateEstimator(_cfg(dim=14))
    with pytest.raises(ValueError):
        est.push_observation(t=0.0, q=np.zeros(10))
    with pytest.raises(ValueError):
        est.push_command(t=0.0, y=np.zeros(21))


# 用例: 尚未观测就 estimate 抛错
# 参数: 无
def test_lag_estimator_requires_observation() -> None:
    est = LagStateEstimator(_cfg())
    with pytest.raises(RuntimeError):
        est.estimate(t_now=0.0)
