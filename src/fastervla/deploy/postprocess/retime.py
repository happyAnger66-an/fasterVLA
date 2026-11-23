"""Time-axis retimers: UniformRetimer (F2) + TimeAxisQP (F3).

``TimeAxisQP`` ports the box-constrained rate QP from the old-stack
``arm_timeline.py``: waypoints stay fixed; only ``step_dt`` (via rates
``s_i = 1/dt_i``) is solved under dt-box, per-joint ``v_max``, optional
prefix lock, and optional inference-buffer wall-clock floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from fastervla.core.types import Chunk, TimedChunk

_ARM_SLICE = slice(8, 22)
_ARM_DIM = 14


@dataclass
class UniformRetimer:
    policy_fps: float = 30.0
    kappa: float = 1.0
    kappa_max: float = 1.5
    kappa_min: float = 1.0

    def __post_init__(self) -> None:
        if not (self.kappa_min <= self.kappa <= self.kappa_max):
            raise ValueError(
                f"kappa={self.kappa} outside [{self.kappa_min}, {self.kappa_max}]"
            )
        if self.policy_fps <= 0:
            raise ValueError("policy_fps must be > 0")

    def retime(self, chunk: Chunk, speed: np.ndarray | None = None, limits=None) -> TimedChunk:
        h = int(chunk.waypoints.shape[0])
        dt = 1.0 / (float(self.kappa) * float(self.policy_fps))
        step_dt = np.full(h, dt, dtype=np.float64)
        if speed is None:
            speed_arr = 1.0 / step_dt
        else:
            speed_arr = np.asarray(speed, dtype=np.float64).reshape(h)
        meta = dict(chunk.meta)
        meta["retime_status"] = "uniform"
        out_chunk = Chunk(
            waypoints=np.asarray(chunk.waypoints, dtype=np.float32).copy(),
            model_dt=float(dt),
            obs_ts=float(chunk.obs_ts),
            meta=meta,
        )
        return TimedChunk(chunk=out_chunk, step_dt=step_dt, speed=speed_arr)


@dataclass
class TimeAxisQPConfig:
    policy_fps: float = 30.0
    kappa: float = 1.0
    kappa_max: float = 1.5
    kappa_min: float = 1.0
    dt_min_s: float = 0.015
    dt_max_s: float = 0.08
    lambda_time: float = 0.1
    lambda_acc: float = 1.0
    v_max_rad_s: float | np.ndarray | None = None
    v_max_from_lag: bool = False
    lag_tau_s: float = 0.057
    lag_gain: float = 0.25
    lag_max_lead_rad: float = 0.06
    buffer_window_enabled: bool = False
    prefix_frames: int = 4
    buffer_frames: int = 4
    r2r_default_ms: float = 100.0
    buffer_margin_ms: float = 20.0
    r2r_window: int = 30
    solver: str = "active_set"
    calibrate_iters: int = 8

    def __post_init__(self) -> None:
        if not (self.kappa_min <= self.kappa <= self.kappa_max):
            raise ValueError(
                f"kappa={self.kappa} outside [{self.kappa_min}, {self.kappa_max}]"
            )
        if self.policy_fps <= 0:
            raise ValueError("policy_fps must be > 0")
        if self.dt_min_s >= self.dt_max_s:
            raise ValueError("dt_min_s must be < dt_max_s")
        if self.lambda_time <= 0:
            raise ValueError("lambda_time must be > 0")
        if self.solver not in ("active_set", "osqp", "lbfgsb"):
            raise ValueError(f"unknown solver {self.solver!r}")
        dt_ref = 1.0 / (float(self.policy_fps) * float(self.kappa))
        if not (self.dt_min_s <= dt_ref <= self.dt_max_s):
            raise ValueError(
                "1/(policy_fps·κ) must lie in [dt_min_s, dt_max_s]"
            )


def rate_hessian(
    dp: np.ndarray,
    s_ref: float,
    lambda_time: float,
    lambda_acc: float,
) -> tuple[np.ndarray, float]:
    """Quadratic part of the rate objective; same normalization as old stack."""
    dp = np.asarray(dp, dtype=np.float64)
    n = int(dp.shape[0])
    sq = np.einsum("ij,ij->i", dp, dp)
    cross = np.einsum("ij,ij->i", dp[:-1], dp[1:])
    wt = float(lambda_time) / (float(s_ref) ** 2 + 1e-6)
    wa = float(lambda_acc) / (float(np.mean(sq * float(s_ref) ** 2)) + 1e-6)
    hessian = 2.0 * wt * np.eye(n)
    for i in range(n - 1):
        hessian[i, i] += 2.0 * wa * sq[i]
        hessian[i + 1, i + 1] += 2.0 * wa * sq[i + 1]
        hessian[i, i + 1] -= 2.0 * wa * cross[i]
        hessian[i + 1, i] -= 2.0 * wa * cross[i]
    return hessian, wt


def box_qp(
    hessian: np.ndarray,
    linear: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    iterations: int = 60,
) -> tuple[np.ndarray, bool]:
    """min ½ x'Hx + q'x on a box (H PD): active-set Newton. Returns (x, kkt_ok)."""
    hessian = np.asarray(hessian, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    fixed = lower >= upper
    x = np.clip(np.linalg.solve(hessian, -linear), lower, upper)
    for _ in range(int(iterations)):
        gradient = hessian @ x + linear
        at_lower = (x <= lower) & (gradient >= 0)
        at_upper = (x >= upper) & (gradient <= 0)
        free = ~(at_lower | at_upper | fixed)
        candidate = np.where(
            at_upper, upper, np.where(at_lower | fixed, np.maximum(lower, x), x)
        )
        if np.any(free):
            bound = ~free
            rhs = -linear[free] - hessian[np.ix_(free, bound)] @ candidate[bound]
            candidate[free] = np.linalg.solve(hessian[np.ix_(free, free)], rhs)
        candidate = np.clip(candidate, lower, upper)
        if np.allclose(candidate, x, rtol=0, atol=1e-10):
            x = candidate
            break
        x = candidate
    gradient = hessian @ x + linear
    scale = 1e-7 * (1.0 + float(np.max(np.abs(linear))))
    interior = (x > lower) & (x < upper)
    kkt = (
        np.all(np.abs(gradient[interior]) <= scale)
        and np.all(gradient[(x <= lower) & ~(x >= upper)] >= -scale)
        and np.all(gradient[(x >= upper) & ~(x <= lower)] <= scale)
    )
    return x, bool(kkt)


def _solve_rates_lbfgsb(
    hessian: np.ndarray,
    linear: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    x0: np.ndarray,
) -> tuple[np.ndarray, str]:
    from scipy.optimize import minimize

    def objective(s: np.ndarray):
        hs = hessian @ s
        return 0.5 * float(s @ hs) + float(linear @ s), hs + linear

    result = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
    )
    status = "lbfgsb" if result.success else "lbfgsb_unconverged"
    return np.clip(result.x, lower, upper), status


def _solve_rates_osqp(
    hessian: np.ndarray,
    linear: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    import osqp
    from scipy import sparse

    n = int(linear.shape[0])
    p = sparse.csc_matrix(hessian)
    a = sparse.eye(n, format="csc")
    prob = osqp.OSQP()
    prob.setup(
        P=p,
        q=linear,
        A=a,
        l=lower,
        u=upper,
        verbose=False,
        eps_abs=1e-8,
        eps_rel=1e-8,
        polish=True,
    )
    result = prob.solve()
    if result.x is None:
        raise RuntimeError(f"osqp failed: {result.info.status}")
    return np.clip(np.asarray(result.x, dtype=np.float64), lower, upper)


def _r2r_budget_s(
    samples_ms: Sequence[float],
    *,
    r2r_default_ms: float,
    buffer_margin_ms: float,
    r2r_window: int,
) -> float:
    recent = list(samples_ms)[-int(r2r_window) :]
    base = (
        float(np.percentile(recent, 95))
        if len(recent) >= 5
        else float(r2r_default_ms)
    )
    return (base + float(buffer_margin_ms)) / 1000.0


@dataclass
class TimeAxisQP:
    config: TimeAxisQPConfig
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Dataclass already validated via TimeAxisQPConfig.__post_init__.
        pass

    def solve_box(
        self,
        hessian: np.ndarray,
        linear: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        """Solve the boxed QP with the configured backend (active_set / osqp / lbfgsb)."""
        solver = self.config.solver
        if solver == "osqp":
            return _solve_rates_osqp(hessian, linear, lower, upper)
        x, kkt = box_qp(hessian, linear, lower, upper)
        if solver == "active_set" and kkt:
            return x
        x2, _ = _solve_rates_lbfgsb(hessian, linear, lower, upper, x)
        return x2

    def _solve_rates(
        self,
        hessian: np.ndarray,
        wt: float,
        target: float,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        linear = -2.0 * float(wt) * float(target) * np.ones(len(lower))
        solver = self.config.solver
        if solver == "osqp":
            return self.solve_box(hessian, linear, lower, upper), "osqp"
        rates, kkt = box_qp(hessian, linear, lower, upper)
        if kkt and solver == "active_set":
            return rates, "active_set"
        rates2, status = _solve_rates_lbfgsb(hessian, linear, lower, upper, rates)
        return rates2, status

    def _calibrated_rates(
        self,
        hessian: np.ndarray,
        wt: float,
        s_ref: float,
        lower: np.ndarray,
        upper: np.ndarray,
        window: np.ndarray,
    ) -> tuple[np.ndarray, str, float]:
        target = float(s_ref)
        calibrate = window[upper[window] >= s_ref * (1.0 - 1e-9)]
        rates = np.clip(np.full(len(lower), s_ref), lower, upper)
        solver = "active_set"
        for _ in range(int(self.config.calibrate_iters)):
            rates, solver = self._solve_rates(hessian, wt, target, lower, upper)
            if len(calibrate) == 0:
                break
            achieved = len(calibrate) / float(np.sum(1.0 / rates[calibrate]))
            ratio = s_ref / achieved
            if abs(ratio - 1.0) < 2e-3:
                break
            target = float(np.clip(target * ratio, s_ref * 0.5, s_ref * 4.0))
        return rates, solver, target

    def _velocity_limit(self) -> np.ndarray:
        limit = np.full(_ARM_DIM, np.inf, dtype=np.float64)
        cfg = self.config
        if cfg.v_max_rad_s is not None:
            values = np.asarray(cfg.v_max_rad_s, dtype=np.float64).reshape(-1)
            if values.shape not in ((1,), (_ARM_DIM,)):
                raise ValueError("v_max_rad_s must be scalar or length 14")
            limit = np.minimum(limit, np.broadcast_to(values, (_ARM_DIM,)))
        if cfg.v_max_from_lag:
            tau = float(cfg.lag_tau_s)
            gain = float(cfg.lag_gain)
            lead = float(cfg.lag_max_lead_rad)
            if tau > 0 and gain > 0:
                lag_cap = lead / (gain * tau)
                limit = np.minimum(limit, lag_cap)
        return limit

    def retime(
        self,
        chunk: Chunk,
        *,
        state: np.ndarray | None = None,
        lock_durations: np.ndarray | Sequence[float] | None = None,
        r2r_samples_ms: Sequence[float] | None = None,
        speed: np.ndarray | None = None,
        limits=None,
    ) -> TimedChunk:
        cfg = self.config
        try:
            return self._retime_qp(
                chunk,
                state=state,
                lock_durations=lock_durations,
                r2r_samples_ms=r2r_samples_ms,
            )
        except Exception as exc:  # noqa: BLE001 — intentional soft fallback
            uni = UniformRetimer(
                policy_fps=cfg.policy_fps,
                kappa=cfg.kappa,
                kappa_max=cfg.kappa_max,
                kappa_min=cfg.kappa_min,
            ).retime(chunk, speed=speed, limits=limits)
            meta = dict(uni.chunk.meta)
            meta["retime_status"] = "uniform_fallback"
            meta["retime_error"] = f"{type(exc).__name__}: {exc}"
            out = Chunk(
                waypoints=uni.chunk.waypoints,
                model_dt=uni.chunk.model_dt,
                obs_ts=uni.chunk.obs_ts,
                meta=meta,
            )
            return TimedChunk(chunk=out, step_dt=uni.step_dt, speed=uni.speed)

    def _retime_qp(
        self,
        chunk: Chunk,
        *,
        state: np.ndarray | None,
        lock_durations: np.ndarray | Sequence[float] | None,
        r2r_samples_ms: Sequence[float] | None,
    ) -> TimedChunk:
        cfg = self.config
        actions = np.asarray(chunk.waypoints, dtype=np.float64)
        h = int(actions.shape[0])
        if state is None:
            state_arm = actions[0, _ARM_SLICE].copy()
        else:
            st = np.asarray(state, dtype=np.float64).reshape(-1)
            if st.shape[0] < 22:
                raise ValueError("state must include arm joints [8:22]")
            state_arm = st[_ARM_SLICE]

        points = np.vstack([state_arm, actions[:, _ARM_SLICE]])
        dp = np.diff(points, axis=0)
        policy_fps = float(cfg.policy_fps)
        s_ref = policy_fps * float(cfg.kappa)
        s_min = 1.0 / float(cfg.dt_max_s)
        s_max = 1.0 / float(cfg.dt_min_s)
        lower = np.full(h, s_min, dtype=np.float64)
        upper = np.full(h, s_max, dtype=np.float64)

        v_max = self._velocity_limit()
        with np.errstate(divide="ignore"):
            per_joint = np.where(np.abs(dp) > 1e-9, v_max / np.abs(dp), np.inf)
        velocity_cap = per_joint.min(axis=1)
        upper = np.minimum(upper, np.maximum(velocity_cap, s_min))

        prefix = int(cfg.prefix_frames)
        buffer = int(cfg.buffer_frames)
        budget = 0.0
        budget_cap = np.inf
        if cfg.buffer_window_enabled:
            budget = _r2r_budget_s(
                list(r2r_samples_ms or ()),
                r2r_default_ms=cfg.r2r_default_ms,
                buffer_margin_ms=cfg.buffer_margin_ms,
                r2r_window=cfg.r2r_window,
            )
            budget_cap = max(buffer / max(budget, 1e-9), s_min)
            upper[prefix : prefix + buffer] = np.minimum(
                upper[prefix : prefix + buffer], budget_cap
            )

        locked = 0 if lock_durations is None else len(lock_durations)
        if locked:
            rates_lock = 1.0 / np.asarray(lock_durations, dtype=np.float64)
            lower[:locked] = rates_lock
            upper[:locked] = rates_lock

        if cfg.buffer_window_enabled:
            end = prefix + buffer
            if locked < end:
                free = np.arange(locked, end)
            else:
                free = np.array([end - 1], dtype=np.int64)
        else:
            end = h
            free = np.arange(locked, h) if locked < h else np.array([h - 1], dtype=np.int64)

        hessian, wt = rate_hessian(dp, s_ref, cfg.lambda_time, cfg.lambda_acc)
        rates, solver, target = self._calibrated_rates(
            hessian, wt, s_ref, lower, upper, free
        )
        durations = 1.0 / rates
        if locked:
            durations[:locked] = np.asarray(lock_durations, dtype=np.float64)

        times = np.concatenate([[0.0], np.cumsum(durations)])
        if not np.all(np.diff(times) > 0):
            raise RuntimeError("non-positive frame durations after QP")

        velocity_hits = [
            int(i) for i in range(locked, end) if velocity_cap[i] < s_ref
        ]
        executed_span = float(times[end] - times[min(locked, max(end - 1, 0))])
        n_exec = end - min(locked, max(end - 1, 0))
        executed_speed = (n_exec / policy_fps) / max(executed_span, 1e-12)

        meta = dict(chunk.meta)
        meta.update(
            {
                "retime_status": solver if solver in ("active_set", "lbfgsb", "osqp") else "qp",
                "mode": "qp",
                "solver": solver,
                "locked_frames": locked,
                "executed_speed": float(executed_speed),
                "time_target_speed": float(target / policy_fps),
                "velocity_limited_frames": velocity_hits,
                "kappa": float(cfg.kappa),
                "s_ref": float(cfg.kappa),
            }
        )
        if cfg.buffer_window_enabled:
            meta["buffer_window_ms"] = float(times[end] - times[prefix]) * 1000.0
            meta["buffer_budget_ms"] = float(budget) * 1000.0

        out_chunk = Chunk(
            waypoints=np.asarray(chunk.waypoints, dtype=np.float32).copy(),
            model_dt=float(np.mean(durations)),
            obs_ts=float(chunk.obs_ts),
            meta=meta,
        )
        speed_arr = 1.0 / durations
        return TimedChunk(chunk=out_chunk, step_dt=durations.astype(np.float64), speed=speed_arr)
