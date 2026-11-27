"""Acados MPC arm tracker with LeadTracker fallback (F4)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from fastervla.deploy.client.tracker.arm_mpc_acados import (
    ArmMpcAcadosSolver,
    ArmMpcOcpConfig,
    acados_importable,
)
from fastervla.deploy.client.tracker.lead import LeadTracker, LeadTrackerConfig


@dataclass
class AcadosMpcTrackerConfig:
    # N=8 (~80 ms) fits the 10 ms / 100 Hz budget on g2s (cython + A76 BLASFEO).
    horizon_n: int = 8
    arm_dim: int = 14
    dt_s: float = 0.01
    plant_tau_s: float = 0.057
    solve_budget_ms: float = 12.0
    # Per-tick command step vs last issued arm cmd (post-solve rate limit).
    # 0.05@100Hz was loose: plan-preview runs saw |cmd−ref| p95≈0.18 / max≈0.33
    # (Lead clips to 0.10) → sway / desk hits. Prefer ≤0.03 until retuned.
    dy_max_rad: float = 0.03
    ddy_max_rad: float = 0.3
    v_max_rad_s: float = 50.0
    q_min_rad: float = -np.pi
    q_max_rad: float = np.pi
    # Optional per-joint boxes (length arm_dim); override scalar q_min_rad/q_max_rad.
    q_min: np.ndarray | None = None
    q_max: np.ndarray | None = None
    w_track0: float = 12.0
    # Smoothness vs tracking; raise when plan preview overshoots (was 10).
    w_ddy: float = 30.0
    # Hard clip of issued arm vs current plan ref (same role as Lead max_lead).
    # None disables; default matches Lead2 guardrail.
    max_lead_rad: float | None = 0.10
    closed_grip_max_lead_rad: float | None = 0.05
    grip_closed_threshold: float = -0.39
    # EMA on (cmd−ref) after clip; kills ±lead_cap chatter that knocks balls.
    lead_smooth_tau_s: float = 0.04
    # Horizon reference:
    #   linear — r → r_{+1 model frame} lerp over N stages (legacy).
    #   plan   — PlanBuffer samples at t + k·dt_s (true preview; caller passes
    #            ``ref_horizon``; falls back to linear if missing).
    preview: str = "linear"
    fallback: str = "lead"
    # If True, build the acados OCP at construction time (compiles C code once).
    eager_build: bool = True

    def __post_init__(self) -> None:
        if int(self.horizon_n) <= 0:
            raise ValueError("horizon_n must be > 0")
        if int(self.arm_dim) <= 0:
            raise ValueError("arm_dim must be > 0")
        if float(self.dt_s) <= 0:
            raise ValueError("dt_s must be > 0")
        if float(self.plant_tau_s) <= 0:
            raise ValueError("plant_tau_s must be > 0")
        if float(self.solve_budget_ms) <= 0:
            raise ValueError("solve_budget_ms must be > 0")
        if self.fallback != "lead":
            raise ValueError(f"unsupported fallback {self.fallback!r}")
        preview = str(self.preview or "linear").lower()
        if preview not in ("linear", "plan"):
            raise ValueError(f"unknown preview {self.preview!r}")
        self.preview = preview
        if self.max_lead_rad is not None and float(self.max_lead_rad) <= 0:
            raise ValueError("max_lead_rad must be > 0 when set")
        if (
            self.closed_grip_max_lead_rad is not None
            and float(self.closed_grip_max_lead_rad) <= 0
        ):
            raise ValueError("closed_grip_max_lead_rad must be > 0 when set")
        if float(self.dy_max_rad) <= 0 or float(self.ddy_max_rad) <= 0:
            raise ValueError("dy_max_rad and ddy_max_rad must be > 0")
        if float(self.w_track0) <= 0 or float(self.w_ddy) <= 0:
            raise ValueError("w_track0 and w_ddy must be > 0")
        if float(self.lead_smooth_tau_s) < 0:
            raise ValueError("lead_smooth_tau_s must be >= 0")
        nq = int(self.arm_dim)
        if self.q_min is not None:
            self.q_min = np.asarray(self.q_min, dtype=np.float64).reshape(-1)
            if self.q_min.shape[0] != nq:
                raise ValueError(f"q_min length {self.q_min.shape[0]} != arm_dim {nq}")
        if self.q_max is not None:
            self.q_max = np.asarray(self.q_max, dtype=np.float64).reshape(-1)
            if self.q_max.shape[0] != nq:
                raise ValueError(f"q_max length {self.q_max.shape[0]} != arm_dim {nq}")


class AcadosMpcTracker:
    """14-DoF arm MPC (acados SQP-RTI); timeout/failure degrades to ``LeadTracker``."""

    ARM_SLICE = slice(5, 19)

    def __init__(
        self,
        config: AcadosMpcTrackerConfig | None = None,
        *,
        lead: LeadTracker | None = None,
        solver: ArmMpcAcadosSolver | None = None,
    ) -> None:
        self.config = config or AcadosMpcTrackerConfig()
        self._lead = lead or LeadTracker(LeadTrackerConfig())
        self.has_acados: bool = acados_importable()
        self.force_fallback: bool = False
        self.fallback_count: int = 0
        self.last_status: str = "init"
        self.last_solve_ms: float = 0.0
        self._last_arm: np.ndarray | None = None
        self._lead_filt = np.zeros(int(self.config.arm_dim), dtype=np.float64)
        self._prev_now: float | None = None
        self._solver: ArmMpcAcadosSolver | None = solver
        self._build_error: str | None = None
        nq = int(self.config.arm_dim)
        self._q_min = (
            np.asarray(self.config.q_min, dtype=np.float64)
            if self.config.q_min is not None
            else np.full(nq, float(self.config.q_min_rad))
        )
        self._q_max = (
            np.asarray(self.config.q_max, dtype=np.float64)
            if self.config.q_max is not None
            else np.full(nq, float(self.config.q_max_rad))
        )
        if solver is None and self.has_acados and self.config.eager_build:
            self._ensure_solver()

    def set_last_arm_command(self, arm: np.ndarray) -> None:
        self._last_arm = (
            np.asarray(arm, dtype=np.float64).reshape(self.config.arm_dim).copy()
        )

    def _ocp_config(self) -> ArmMpcOcpConfig:
        cfg = self.config
        nq = int(cfg.arm_dim)
        q_min = (
            np.asarray(cfg.q_min, dtype=np.float64)
            if cfg.q_min is not None
            else np.full(nq, float(cfg.q_min_rad))
        )
        q_max = (
            np.asarray(cfg.q_max, dtype=np.float64)
            if cfg.q_max is not None
            else np.full(nq, float(cfg.q_max_rad))
        )
        return ArmMpcOcpConfig(
            nq=nq,
            dt=float(cfg.dt_s),
            tau=float(cfg.plant_tau_s),
            N=int(cfg.horizon_n),
            q_min=q_min,
            q_max=q_max,
            v_max=np.full(nq, float(cfg.v_max_rad_s)),
            dy_max=np.full(nq, float(cfg.dy_max_rad)),
            ddy_max=np.full(nq, float(cfg.ddy_max_rad)),
            w_track0=float(cfg.w_track0),
            w_ddy=float(cfg.w_ddy),
        )

    def _ensure_solver(self) -> ArmMpcAcadosSolver | None:
        if self._solver is not None:
            return self._solver
        if not self.has_acados:
            self._build_error = "acados_template/casadi not importable"
            return None
        try:
            self._solver = ArmMpcAcadosSolver.create(self._ocp_config())
            self._build_error = None
        except Exception as exc:  # noqa: BLE001 — surface via fallback path
            self._solver = None
            self._build_error = f"{type(exc).__name__}: {exc}"
        return self._solver

    def step(
        self,
        q_hat: np.ndarray,
        reference: np.ndarray,
        *,
        ref_ahead: np.ndarray,
        ref_vel: np.ndarray,
        now: float,
        blending: bool = False,
        contact: bool = False,
        ref_horizon: np.ndarray | None = None,
    ) -> np.ndarray:
        """``ref_horizon``: arm refs ``(N+1, arm_dim)`` at ``now + k·dt_s``; used when
        ``preview='plan'``."""
        q_hat_arr = np.asarray(q_hat, dtype=np.float64).reshape(-1)
        ref = np.asarray(reference, dtype=np.float64).reshape(-1)
        if q_hat_arr.shape[0] != self.config.arm_dim:
            raise ValueError(
                f"q_hat dim {q_hat_arr.shape[0]} != arm_dim {self.config.arm_dim}"
            )
        if ref.shape[0] != 21:
            raise ValueError(f"reference must be length 21, got {ref.shape}")

        if self.force_fallback:
            return self._fallback(
                reference=ref,
                ref_ahead=ref_ahead,
                ref_vel=ref_vel,
                now=now,
                blending=blending,
                status="forced",
            )

        # Injected ``_solve`` (unit tests) always wins over native acados.
        using_default_solve = getattr(self._solve, "__func__", None) is AcadosMpcTracker._solve
        if using_default_solve:
            solver = self._ensure_solver()
            if solver is None:
                return self._fallback(
                    reference=ref,
                    ref_ahead=ref_ahead,
                    ref_vel=ref_vel,
                    now=now,
                    blending=blending,
                    status="no_acados",
                )

        ref_arm = ref[self.ARM_SLICE]
        horizon = None
        if self.config.preview == "plan" and ref_horizon is not None:
            horizon = np.asarray(ref_horizon, dtype=np.float64)
            if horizon.ndim != 2 or horizon.shape[1] != self.config.arm_dim:
                raise ValueError(
                    f"ref_horizon must be (K, {self.config.arm_dim}), got {horizon.shape}"
                )
        t0 = time.perf_counter()
        try:
            u_arm, status, solve_ms = self._solve(
                q_hat=q_hat_arr,
                ref_arm=ref_arm,
                ref_ahead=np.asarray(ref_ahead, dtype=np.float64).reshape(
                    self.config.arm_dim
                ),
                ref_vel=np.asarray(ref_vel, dtype=np.float64).reshape(
                    self.config.arm_dim
                ),
                now=float(now),
                contact=bool(contact),
                ref_horizon=horizon,
            )
        except Exception:
            return self._fallback(
                reference=ref,
                ref_ahead=ref_ahead,
                ref_vel=ref_vel,
                now=now,
                blending=blending,
                status="fail",
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        reported = float(solve_ms)
        self.last_solve_ms = reported
        budget = float(self.config.solve_budget_ms)
        if reported > budget + 1e-9 or elapsed_ms > budget + 1e-9:
            return self._fallback(
                reference=ref,
                ref_ahead=ref_ahead,
                ref_vel=ref_vel,
                now=now,
                blending=blending,
                status="timeout",
            )
        if status != "ok":
            return self._fallback(
                reference=ref,
                ref_ahead=ref_ahead,
                ref_vel=ref_vel,
                now=now,
                blending=blending,
                status="fallback" if status != "fail" else "fail",
            )

        u = np.asarray(u_arm, dtype=np.float64).reshape(self.config.arm_dim)
        # 1) Bound departure from instantaneous plan ref (Lead-equivalent).
        # 2) EMA the lead to avoid ±cap chatter (plan_safe still rode the clip).
        # 3) Rate-limit vs last issued cmd.
        max_lead = self.config.max_lead_rad
        lead = u - ref_arm
        if max_lead is not None:
            lead_cap = float(max_lead)
            closed = self.config.closed_grip_max_lead_rad
            if closed is not None and float(ref[20]) > float(
                self.config.grip_closed_threshold
            ):
                lead_cap = min(lead_cap, float(closed))
            lead = np.clip(lead, -lead_cap, lead_cap)
        tau = float(self.config.lead_smooth_tau_s)
        if tau > 0.0 and self._prev_now is not None and float(now) > self._prev_now:
            dt = float(now) - self._prev_now
            alpha = 1.0 - math.exp(-dt / tau)
            self._lead_filt = self._lead_filt + alpha * (lead - self._lead_filt)
            lead = self._lead_filt
        else:
            self._lead_filt = np.asarray(lead, dtype=np.float64).copy()
        self._prev_now = float(now)
        u = ref_arm + lead
        if self._last_arm is not None:
            dy = float(self.config.dy_max_rad)
            delta = np.clip(u - self._last_arm, -dy, dy)
            u = self._last_arm + delta
        # Saturate to configured joint box (GDK soft limits when wired).
        u = np.clip(u, self._q_min, self._q_max)
        self._last_arm = u.copy()

        out = ref.copy()
        out[self.ARM_SLICE] = u
        self.last_status = "ok"
        return out

    def _fallback(
        self,
        *,
        reference: np.ndarray,
        ref_ahead: np.ndarray,
        ref_vel: np.ndarray,
        now: float,
        blending: bool,
        status: str,
    ) -> np.ndarray:
        self.fallback_count += 1
        self.last_status = status
        out = self._lead.step(
            reference,
            ref_ahead=ref_ahead,
            ref_vel=ref_vel,
            now=now,
            blending=blending,
        )
        self._last_arm = np.asarray(out[self.ARM_SLICE], dtype=np.float64).copy()
        return out

    def _solve(
        self,
        *,
        q_hat: np.ndarray,
        ref_arm: np.ndarray,
        ref_ahead: np.ndarray,
        ref_vel: np.ndarray,
        now: float,
        contact: bool = False,
        ref_horizon: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str, float]:
        """Native acados SQP-RTI step. Returns ``(u_arm, status, solve_ms)``."""
        _ = (ref_vel, now)
        solver = self._ensure_solver()
        if solver is None:
            raise RuntimeError(self._build_error or "acados solver not available")

        n = int(self.config.horizon_n)
        if ref_horizon is not None:
            ai = np.asarray(ref_horizon, dtype=np.float64)
        elif n <= 1:
            ai = ref_arm.reshape(1, -1).copy()
        else:
            # Legacy: hold current ref, then blend toward ~1 model frame ahead.
            alpha = np.linspace(0.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
            ai = (1.0 - alpha) * ref_arm.reshape(1, -1) + alpha * ref_ahead.reshape(
                1, -1
            )

        t0 = time.perf_counter()
        u = solver.solve(q_hat, ai, contact_mode=contact)
        solve_ms = (time.perf_counter() - t0) * 1000.0
        return np.asarray(u, dtype=np.float64), "ok", float(solve_ms)
