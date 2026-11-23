"""Acados MPC arm tracker with LeadTracker fallback (F4)."""

from __future__ import annotations

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
    dy_max_rad: float = 0.05
    ddy_max_rad: float = 0.5
    v_max_rad_s: float = 50.0
    q_min_rad: float = -np.pi
    q_max_rad: float = np.pi
    # Optional per-joint boxes (length arm_dim); override scalar q_min_rad/q_max_rad.
    q_min: np.ndarray | None = None
    q_max: np.ndarray | None = None
    w_track0: float = 10.0
    w_ddy: float = 10.0
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
    ) -> np.ndarray:
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
    ) -> tuple[np.ndarray, str, float]:
        """Native acados SQP-RTI step. Returns ``(u_arm, status, solve_ms)``."""
        _ = (ref_vel, now)
        solver = self._ensure_solver()
        if solver is None:
            raise RuntimeError(self._build_error or "acados solver not available")

        n = int(self.config.horizon_n)
        # Horizon reference: hold current ref, then blend toward ahead.
        if n <= 1:
            ai = ref_arm.reshape(1, -1).copy()
        else:
            alpha = np.linspace(0.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
            ai = (1.0 - alpha) * ref_arm.reshape(1, -1) + alpha * ref_ahead.reshape(
                1, -1
            )

        t0 = time.perf_counter()
        u = solver.solve(q_hat, ai, contact_mode=contact)
        solve_ms = (time.perf_counter() - t0) * 1000.0
        return np.asarray(u, dtype=np.float64), "ok", float(solve_ms)
