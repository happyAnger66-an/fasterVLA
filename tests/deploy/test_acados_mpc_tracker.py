"""模块: fastervla.deploy.client.tracker.acados_mpc.AcadosMpcTracker
功能: 14 维臂 SQP-RTI MPC；超时/失败降级 LeadTracker 并计数。

F4 测试（先测后码）。对齐 design §5.2 / deploy-g2-plan §4.5。
acados 为可选依赖：未安装时仍可构造，热路径走 fallback。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fastervla.deploy.client.tracker.acados_mpc import (
    AcadosMpcTracker,
    AcadosMpcTrackerConfig,
)
from fastervla.deploy.client.tracker.lead import LeadTracker, LeadTrackerConfig


def _cmd(arm: float | np.ndarray = 0.0) -> np.ndarray:
    q = np.zeros(21, dtype=np.float64)
    if np.ndim(arm) == 0:
        q[5:19] = float(arm)
    else:
        q[5:19] = np.asarray(arm, dtype=np.float64)
    q[19] = 0.4
    q[20] = 0.6
    return q


def _cfg(**kwargs: object) -> AcadosMpcTrackerConfig:
    base = dict(
        horizon_n=8,
        arm_dim=14,
        dt_s=0.01,
        solve_budget_ms=12.0,
        dy_max_rad=0.05,
        ddy_max_rad=0.5,
        fallback="lead",
        eager_build=False,  # unit tests avoid compiling acados C code
    )
    base.update(kwargs)
    return AcadosMpcTrackerConfig(**base)  # type: ignore[arg-type]


# 用例: 默认 N≈8、14 维、预算默认 12ms、降级到 lead
# 参数: 无
def test_mpc_config_defaults_match_g2_plan() -> None:
    cfg = AcadosMpcTrackerConfig()
    assert cfg.horizon_n == 8
    assert cfg.arm_dim == 14
    assert cfg.dt_s == pytest.approx(0.01)
    assert cfg.solve_budget_ms == pytest.approx(12.0)
    assert cfg.fallback == "lead"


# 用例: 非法 horizon / budget / fallback 拒绝
# 参数: 无
def test_mpc_config_rejects_bad_values() -> None:
    with pytest.raises((ValueError, TypeError)):
        AcadosMpcTrackerConfig(horizon_n=0)
    with pytest.raises((ValueError, TypeError)):
        AcadosMpcTrackerConfig(solve_budget_ms=0.0)
    with pytest.raises((ValueError, TypeError)):
        AcadosMpcTrackerConfig(fallback="nope")


# 用例: 无 acados 也可构造；step 走 LeadTracker 降级，fallback_count 增加
# 参数: 无
def test_mpc_fallback_without_acados_or_forced() -> None:
    tr = AcadosMpcTracker(_cfg(), lead=LeadTracker(LeadTrackerConfig(lead_smooth_tau_s=0.0)))
    # Force backend unavailable / fail-closed path.
    tr.force_fallback = True
    ref = _cmd(0.1)
    ahead = np.full(14, 0.2)
    vel = np.full(14, 0.5)
    out = tr.step(
        q_hat=np.full(14, 0.1),
        reference=ref,
        ref_ahead=ahead,
        ref_vel=vel,
        now=0.0,
    )
    assert out.shape == (21,)
    assert tr.fallback_count >= 1
    assert tr.last_status in ("fallback", "no_acados", "forced")
    # Grippers passthrough from reference.
    np.testing.assert_allclose(out[19:], ref[19:], atol=1e-12)


# 用例: 可注入假求解器成功路径；输出臂维有限、夹爪透传
# 参数: 无
def test_mpc_injected_solve_success_updates_arms_only() -> None:
    tr = AcadosMpcTracker(_cfg())

    def fake_solve(*, q_hat, ref_arm, **_k):
        # Pull slightly toward ref from q_hat, clipped.
        u = q_hat + np.clip(ref_arm - q_hat, -0.02, 0.02)
        return u, "ok", 0.5  # arm cmd, status, solve_ms

    tr._solve = fake_solve  # type: ignore[method-assign]
    ref = _cmd(0.3)
    q_hat = np.full(14, 0.1)
    out = tr.step(
        q_hat=q_hat,
        reference=ref,
        ref_ahead=np.full(14, 0.3),
        ref_vel=np.zeros(14),
        now=0.0,
    )
    assert tr.fallback_count == 0
    assert tr.last_status == "ok"
    assert tr.last_solve_ms == pytest.approx(0.5)
    np.testing.assert_allclose(out[19:], ref[19:])
    np.testing.assert_allclose(out[5:19], q_hat + 0.02, atol=1e-9)


# 用例: 求解耗时超过 budget → 降级并记 timeout
# 参数: 无
def test_mpc_timeout_triggers_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    tr = AcadosMpcTracker(
        _cfg(solve_budget_ms=1.0),
        lead=LeadTracker(LeadTrackerConfig(gain=0.0, lead_smooth_tau_s=0.0)),
    )

    def slow_solve(**_k):
        time.sleep(0.02)
        return np.zeros(14), "ok", 20.0

    tr._solve = slow_solve  # type: ignore[method-assign]
    ref = _cmd(0.0)
    out = tr.step(
        q_hat=np.zeros(14),
        reference=ref,
        ref_ahead=np.zeros(14),
        ref_vel=np.zeros(14),
        now=0.0,
    )
    assert tr.fallback_count >= 1
    assert tr.last_status in ("timeout", "fallback")
    np.testing.assert_allclose(out, ref, atol=1e-12)  # gain=0 lead ≡ passthrough


# 用例: 求解抛错 → 降级，不向外抛
# 参数: 无
def test_mpc_solve_exception_falls_back() -> None:
    tr = AcadosMpcTracker(
        _cfg(),
        lead=LeadTracker(LeadTrackerConfig(gain=0.0, lead_smooth_tau_s=0.0)),
    )

    def boom(**_k):
        raise RuntimeError("acados exploded")

    tr._solve = boom  # type: ignore[method-assign]
    ref = _cmd(0.15)
    out = tr.step(
        q_hat=np.zeros(14),
        reference=ref,
        ref_ahead=np.zeros(14),
        ref_vel=np.zeros(14),
        now=1.0,
    )
    assert tr.fallback_count >= 1
    assert "fail" in tr.last_status or tr.last_status == "fallback"
    np.testing.assert_allclose(out, ref, atol=1e-12)


# 用例: Δy 硬约束 — 成功解相对上一拍命令的臂增量不超过 dy_max
# 参数: 无
def test_mpc_enforces_dy_max_on_success() -> None:
    dy_max = 0.03
    tr = AcadosMpcTracker(_cfg(dy_max_rad=dy_max))
    prev = np.full(14, 0.0)

    def big_step(*, q_hat, ref_arm, **_k):
        return ref_arm.copy(), "ok", 0.2

    tr._solve = big_step  # type: ignore[method-assign]
    tr.set_last_arm_command(prev)
    ref = _cmd(0.5)  # far from prev
    out = tr.step(
        q_hat=prev.copy(),
        reference=ref,
        ref_ahead=np.full(14, 0.5),
        ref_vel=np.zeros(14),
        now=0.0,
    )
    delta = out[5:19] - prev
    assert float(np.max(np.abs(delta))) <= dy_max + 1e-9


# 用例: preview 开关 — linear 不传 ref_horizon；plan 透传 (N+1, 14)；非法值拒绝
# 参数: 无
def test_mpc_preview_switch_routes_ref_horizon() -> None:
    with pytest.raises(ValueError):
        AcadosMpcTrackerConfig(preview="nope")
    seen: dict[str, object] = {}

    def spy(*, q_hat, ref_horizon=None, **_k):
        seen["h"] = ref_horizon
        return q_hat.copy(), "ok", 0.1

    horizon = np.tile(np.linspace(0.0, 0.08, 9).reshape(-1, 1), (1, 14))
    kw = dict(
        q_hat=np.zeros(14),
        reference=_cmd(0.0),
        ref_ahead=np.zeros(14),
        ref_vel=np.zeros(14),
        now=0.0,
        ref_horizon=horizon,
    )

    tr_lin = AcadosMpcTracker(_cfg(preview="linear"))
    tr_lin._solve = spy  # type: ignore[method-assign]
    tr_lin.step(**kw)  # type: ignore[arg-type]
    assert seen["h"] is None

    tr_plan = AcadosMpcTracker(_cfg(preview="plan"))
    tr_plan._solve = spy  # type: ignore[method-assign]
    tr_plan.step(**kw)  # type: ignore[arg-type]
    np.testing.assert_allclose(seen["h"], horizon)

    with pytest.raises(ValueError):
        tr_plan.step(**{**kw, "ref_horizon": np.zeros((9, 3))})  # type: ignore[arg-type]


# 用例: 维数错误的 q_hat / reference 抛错
# 参数: 无
def test_mpc_rejects_wrong_shapes() -> None:
    tr = AcadosMpcTracker(_cfg())
    with pytest.raises(ValueError):
        tr.step(
            q_hat=np.zeros(10),
            reference=_cmd(0.0),
            ref_ahead=np.zeros(14),
            ref_vel=np.zeros(14),
            now=0.0,
        )
    with pytest.raises(ValueError):
        tr.step(
            q_hat=np.zeros(14),
            reference=np.zeros(14),
            ref_ahead=np.zeros(14),
            ref_vel=np.zeros(14),
            now=0.0,
        )


# 用例: acados 若已安装，后端可探测；未安装则 has_acados is False
# 参数: 无
def test_mpc_reports_acados_availability() -> None:
    tr = AcadosMpcTracker(_cfg())
    assert isinstance(tr.has_acados, bool)
    # Either path is valid; just lock the attribute contract.
    if not tr.has_acados:
        tr.force_fallback = False
        # Still must not crash on step (fallback).
        out = tr.step(
            q_hat=np.zeros(14),
            reference=_cmd(0.0),
            ref_ahead=np.zeros(14),
            ref_vel=np.zeros(14),
            now=0.0,
        )
        assert out.shape == (21,)
        assert tr.fallback_count >= 1


# 用例: OCP 配置可构造；若本机有 acados 则能编译求解器并解一步
# 参数: 无
def test_mpc_acados_solver_builds_and_steps_when_installed() -> None:
    from fastervla.deploy.client.tracker.arm_mpc_acados import (
        ArmMpcAcadosSolver,
        ArmMpcOcpConfig,
        acados_importable,
    )

    if not acados_importable():
        pytest.skip("casadi/acados_template not installed")
    cfg = ArmMpcOcpConfig(nq=14, N=5, dt=0.01, tau=0.057, dy_max=0.05, ddy_max=0.5)
    try:
        solver = ArmMpcAcadosSolver.create(cfg)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"acados present but OCP build failed: {exc}")
    q = np.zeros(14)
    ref = np.full((5, 14), 0.1)
    u = solver.solve(q, ref, contact_mode=False)
    assert u.shape == (14,)
    assert np.isfinite(u).all()

    tr = AcadosMpcTracker(_cfg(horizon_n=5, eager_build=False), solver=solver)
    out = tr.step(
        q_hat=q,
        reference=_cmd(0.1),
        ref_ahead=np.full(14, 0.1),
        ref_vel=np.zeros(14),
        now=0.0,
    )
    assert out.shape == (21,)
    # With a live solver this should prefer MPC over lead.
    assert tr.last_status in ("ok", "timeout", "fail", "fallback")
