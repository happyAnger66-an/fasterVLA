"""Acados SQP-RTI arm MPC (ported from realtime-vla-v2 ``AcadosPlanner``).

MIT-style algorithm port: discrete first-order lag plant, NONLINEAR_LS cost,
h-constraints on lead / Δy / Δ²y. Requires ``casadi`` + ``acados_template`` and
a working acados install (``ACADOS_SOURCE_DIR`` / shared libs).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import casadi as ca
except Exception:  # pragma: no cover
    ca = None

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except Exception:  # pragma: no cover
    AcadosModel = None  # type: ignore[misc, assignment]
    AcadosOcp = None  # type: ignore[misc, assignment]
    AcadosOcpSolver = None  # type: ignore[misc, assignment]


def acados_importable() -> bool:
    return ca is not None and AcadosOcp is not None and AcadosOcpSolver is not None


@dataclass
class ArmMpcOcpConfig:
    """OCP knobs aligned with V2 ``MPCConfig`` (14-DoF arms by default)."""

    nq: int = 14
    dt: float = 0.01
    tau: float = 0.057
    N: int = 8
    q_min: np.ndarray | None = None
    q_max: np.ndarray | None = None
    v_max: np.ndarray | None = None
    dy_max: np.ndarray | None = None
    ddy_max: np.ndarray | None = None
    w_track0: float = 10.0
    track_decay: float = 1.0
    w_cmd: float = 0.0
    w_yx: float = 0.0
    w_dy: float = 0.0
    w_ddy: float = 10.0
    contact_v_scale: float = 0.4
    export_dir: str | None = None
    json_file: str | None = None
    # Cython wrapper keeps set/get in compiled code (much less GIL thrash vs ctypes).
    use_cython: bool = True

    def __post_init__(self) -> None:
        nq = int(self.nq)
        if nq <= 0:
            raise ValueError("nq must be > 0")
        if self.dt <= 0 or self.tau <= 0 or self.N <= 0:
            raise ValueError("dt, tau, N must be > 0")
        self.q_min = _vec(self.q_min, nq, -np.pi)
        self.q_max = _vec(self.q_max, nq, np.pi)
        self.v_max = _vec(self.v_max, nq, 50.0)
        self.dy_max = _vec(self.dy_max, nq, 100.0)
        self.ddy_max = _vec(self.ddy_max, nq, 100.0)

    def stage_sqrt_w(self, k: int) -> float:
        if k <= 0:
            return 0.0
        w = self.w_track0 * (self.track_decay ** (k - 1))
        return float(np.sqrt(max(w, 1e-8)))


def _vec(value: np.ndarray | float | None, n: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(n, float(default), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    if arr.size != n:
        raise ValueError(f"expected length {n}, got {arr.size}")
    return arr.copy()


def build_acados_ocp_solver(cfg: ArmMpcOcpConfig) -> Any:
    """Compile / load an ``AcadosOcpSolver`` for the arm tracking OCP."""
    if not acados_importable():
        raise RuntimeError(
            "acados planner dependencies missing (casadi + acados_template)"
        )
    assert ca is not None and AcadosModel is not None and AcadosOcp is not None
    assert AcadosOcpSolver is not None

    nq = int(cfg.nq)
    nx = 3 * nq
    nu = nq
    np_p = nq + 1

    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    p = ca.SX.sym("p", np_p)
    x_q = x[0:nq]
    y_prev = x[nq : 2 * nq]
    y_prev2 = x[2 * nq : 3 * nq]
    y_cmd = u[0:nq]
    r_ref = p[0:nq]
    sqrt_w_track = p[-1]

    a = float(np.exp(-cfg.dt / cfg.tau))
    x_next = a * x_q + (1.0 - a) * y_cmd
    x_next_full = ca.vertcat(x_next, y_cmd, y_prev)

    model = AcadosModel()
    model.name = "fastervla_arm_mpc_fo_discrete"
    model.x = x
    model.u = u
    model.p = p
    model.disc_dyn_expr = x_next_full

    e_track = sqrt_w_track * (x_q - r_ref)
    e_cmd = ca.sqrt(cfg.w_cmd) * (y_cmd - r_ref)
    e_yx = ca.sqrt(cfg.w_yx) * (y_cmd - x_q)
    dy = y_cmd - y_prev
    ddy = y_cmd - 2.0 * y_prev + y_prev2
    e_dy = ca.sqrt(cfg.w_dy) * dy
    e_ddy = ca.sqrt(cfg.w_ddy) * ddy
    y_expr = ca.vertcat(e_track, e_cmd, e_yx, e_dy, e_ddy)
    ny = int(y_expr.shape[0])
    y_expr_e = sqrt_w_track * (x_q - r_ref)
    ny_e = int(y_expr_e.shape[0])

    ocp = AcadosOcp()
    ocp.model = model
    # Compat: older acados uses dims.N; newer prefers solver_options.N_horizon.
    try:
        ocp.dims.N = int(cfg.N)
    except Exception:
        pass
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.tf = float(cfg.N * cfg.dt)
    try:
        ocp.solver_options.N_horizon = int(cfg.N)
    except Exception:
        pass
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver_cond_N = min(5, int(cfg.N))
    ocp.solver_options.print_level = 0

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr = y_expr
    ocp.model.cost_y_expr_e = y_expr_e
    ocp.cost.W = np.eye(ny)
    ocp.cost.W_e = np.eye(ny_e)
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    ocp.constraints.idxbu = np.arange(nu, dtype=np.int64)
    ocp.constraints.lbu = np.asarray(cfg.q_min, dtype=np.float64).copy()
    ocp.constraints.ubu = np.asarray(cfg.q_max, dtype=np.float64).copy()
    ocp.constraints.idxbx_0 = np.arange(nx, dtype=np.int64)
    ocp.constraints.lbx_0 = np.zeros(nx)
    ocp.constraints.ubx_0 = np.zeros(nx)

    h_expr = ca.vertcat((y_cmd - x_q), (y_cmd - y_prev), (y_cmd - 2.0 * y_prev + y_prev2))
    ocp.model.con_h_expr = h_expr
    try:
        ocp.model.con_h_expr_0 = h_expr
    except Exception:
        pass

    e_max = np.asarray(cfg.tau * np.asarray(cfg.v_max), dtype=np.float64)
    lh = np.concatenate([-e_max, -np.asarray(cfg.dy_max), -np.asarray(cfg.ddy_max)])
    uh = np.concatenate([+e_max, +np.asarray(cfg.dy_max), +np.asarray(cfg.ddy_max)])
    ocp.constraints.lh = lh
    ocp.constraints.uh = uh
    try:
        ocp.constraints.lh_0 = lh.copy()
        ocp.constraints.uh_0 = uh.copy()
    except Exception:
        pass

    ocp.parameter_values = np.zeros(np_p)
    export_dir = cfg.export_dir or str(
        Path(tempfile.gettempdir()) / f"fastervla_acados_arm{nq}"
    )
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    ocp.code_export_directory = export_dir
    json_file = cfg.json_file or str(Path(export_dir) / "acados_ocp.json")
    if cfg.use_cython:
        try:
            return AcadosOcpSolver.create_cython_solver(ocp, verbose=False)
        except Exception as exc:
            # Fall back to ctypes wrapper if cython build/import fails.
            import warnings

            warnings.warn(f"acados cython solver unavailable, using ctypes: {exc}")
    return AcadosOcpSolver(ocp, json_file=json_file, verbose=False)


@dataclass
class ArmMpcAcadosSolver:
    """Stateful RTI wrapper: set x0 from q̂, shoot reference horizon, return u0."""

    cfg: ArmMpcOcpConfig
    solver: Any = field(repr=False)
    x_aug: np.ndarray = field(init=False)
    initialized: bool = False
    _p_buf: np.ndarray = field(init=False, repr=False)
    _sqrt_w: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nq = int(self.cfg.nq)
        self.x_aug = np.zeros(3 * nq, dtype=np.float64)
        self._p_buf = np.zeros(nq + 1, dtype=np.float64)
        self._sqrt_w = np.asarray(
            [self.cfg.stage_sqrt_w(k) for k in range(self.cfg.N + 1)],
            dtype=np.float64,
        )

    @classmethod
    def create(cls, cfg: ArmMpcOcpConfig | None = None) -> "ArmMpcAcadosSolver":
        cfg = cfg or ArmMpcOcpConfig()
        solver = build_acados_ocp_solver(cfg)
        return cls(cfg=cfg, solver=solver)

    def reset(self, q0: np.ndarray) -> None:
        q0 = np.asarray(q0, dtype=np.float64).reshape(self.cfg.nq)
        nq = self.cfg.nq
        self.x_aug[:] = 0.0
        self.x_aug[0:nq] = q0
        self.x_aug[nq : 2 * nq] = q0
        self.x_aug[2 * nq : 3 * nq] = q0
        self.initialized = True

    def solve(
        self,
        q_hat: np.ndarray,
        ai_future: np.ndarray,
        *,
        contact_mode: bool = False,
    ) -> np.ndarray:
        q_hat = np.asarray(q_hat, dtype=np.float64).reshape(self.cfg.nq)
        if not self.initialized:
            self.reset(q_hat)

        cfg = self.cfg
        nq = cfg.nq
        ai = np.asarray(ai_future, dtype=np.float64)
        if ai.ndim != 2:
            raise ValueError("ai_future must be 2D (N, nq)")
        if ai.shape[1] != nq and ai.shape[0] == nq:
            ai = ai.T
        if ai.shape[1] != nq:
            raise ValueError(f"ai_future last dim must be {nq}, got {ai.shape}")
        # Rows 0..N-1 → stages; optional row N → terminal (else reuse row N-1).
        if ai.shape[0] < cfg.N:
            pad = np.repeat(ai[-1:, :], cfg.N - ai.shape[0], axis=0)
            ai = np.vstack([ai, pad])
        else:
            ai = ai[: cfg.N + 1]

        ai_use = (
            np.repeat(q_hat.reshape(1, -1), ai.shape[0], axis=0) if contact_mode else ai
        )

        self.x_aug[0:nq] = q_hat
        self.solver.set(0, "lbx", self.x_aug)
        self.solver.set(0, "ubx", self.x_aug)
        p = self._p_buf
        sw = self._sqrt_w
        for k in range(cfg.N):
            p[:nq] = ai_use[k]
            p[nq] = sw[k]
            self.solver.set(k, "p", p)
        p[:nq] = ai_use[-1]
        p[nq] = sw[cfg.N]
        self.solver.set(cfg.N, "p", p)

        status = int(self.solver.solve())
        if status != 0:
            raise RuntimeError(f"acados solve status={status}")

        u0 = np.asarray(self.solver.get(0, "u"), dtype=np.float64).reshape(nq)
        x1 = np.asarray(self.solver.get(1, "x"), dtype=np.float64).reshape(-1)
        self.x_aug = x1.copy()
        return u0
