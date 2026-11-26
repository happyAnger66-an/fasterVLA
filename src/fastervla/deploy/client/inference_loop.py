"""L1 InferenceLoop + control tick runner (F2) with enriched TraceRecorder fields."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from fastervla.core.types import Chunk, Observation, Prefix, TimedChunk
from fastervla.deploy.client.io.interfaces import RobotPlatform
from fastervla.deploy.client.estimator import LagStateEstimator, LagStateEstimatorConfig
from fastervla.deploy.client.io.layout import ARM_POLICY_SLICE, COMMAND_INDICES, LayoutMapper
from fastervla.deploy.client.plan_buffer import PlanBuffer
from fastervla.deploy.client.recorder import (
    STAGE_BUFFER_EMPTY_HOLD,
    STAGE_STEP,
    TRACKER_FALLBACK,
    TRACKER_LEAD,
    TRACKER_MPC,
    TRACKER_PASSTHROUGH,
    TraceRecorder,
)
from fastervla.deploy.client.tracker.acados_mpc import AcadosMpcTracker
from fastervla.deploy.client.tracker.lead import LeadTracker, LeadTrackerConfig
from fastervla.deploy.postprocess.arm_chunk_filter import apply_arm_chunk_filter
from fastervla.deploy.postprocess.retime import TimeAxisQP, UniformRetimer


class _Backend(Protocol):
    def sample(
        self,
        *,
        obs: Observation,
        prefix: Prefix | None,
        rtc_mode: str,
        n: int,
        rtc: dict[str, Any] | None = None,
    ) -> list[Any]: ...


@dataclass
class InferenceLoopConfig:
    control_hz: float = 100.0
    policy_fps: float = 30.0
    kappa: float = 1.0
    kappa_max: float = 2.0
    kappa_min: float = 1.0
    t_infer_min_s: float = 0.0
    t_switch_min_s: float = 0.0
    latency_margin_ms: float = 10.0
    cold_start_latency_ms: float = 100.0
    duration_s: float = 1.0
    rtc_mode: str = "INPAINT"
    horizon: int = 30
    action_dim: int = 24
    latency_window: int = 32
    # Old-stack 4+4 schedule: wait this many model frames after an accept
    # before the next replan/merge (0 disables). Buffer is the early-replan
    # escape when remaining_steps drops to this level.
    execution_prefix_frames: int = 4
    execution_buffer_frames: int = 4
    use_lead_tracker: bool = True
    tracker: str = "lead"  # none | lead | mpc
    # LeadTrackerConfig defaults (soft-lead A/B via recipe `lead_tracker:`).
    lead_tau_s: float = 0.057
    lead_gain: float = 0.25
    lead_max_lead_rad: float = 0.06
    lead_smooth_tau_s: float = 0.040
    lead_hold_during_blend: bool = True
    # None → LeadTrackerConfig default (disabled). Tighten lead while grip closed.
    lead_closed_grip_max_lead_rad: float | None = None
    # If True, latch grippers to first measured and ignore policy gripper (chatter A/B).
    hold_gripper: bool = False
    track_warn_threshold_rad: float = 0.10
    # Path smooth (old soft): ZP DCT on arms before retime; PlanBuffer interp.
    arm_chunk_filter: str = "none"  # none | zero_phase_dct | zero_phase_dct_cross
    arm_chunk_filter_cutoff_hz: float = 3.0
    arm_chunk_filter_context_frames: int = 10
    arm_interpolation: str = "linear"  # linear | pchip
    # TimeAxisQP only: lock the new chunk's first d_hat + extra frame durations
    # to the old plan's (same waypoints via RTC prefix) so merges keep rate.
    qp_prefix_lock: bool = False
    qp_lock_extra_frames: int = 2

    def __post_init__(self) -> None:
        if not (self.kappa_min <= self.kappa <= self.kappa_max):
            raise ValueError(
                f"kappa={self.kappa} outside [{self.kappa_min}, {self.kappa_max}]"
            )
        if self.control_hz <= 0 or self.policy_fps <= 0:
            raise ValueError("control_hz and policy_fps must be > 0")
        if self.tracker not in ("none", "lead", "mpc"):
            raise ValueError(f"unknown tracker {self.tracker!r}")
        if int(self.execution_prefix_frames) < 0:
            raise ValueError("execution_prefix_frames must be >= 0")
        if int(self.execution_buffer_frames) < 0:
            raise ValueError("execution_buffer_frames must be >= 0")
        if float(self.lead_tau_s) < 0:
            raise ValueError("lead_tau_s must be >= 0")
        if float(self.lead_gain) < 0:
            raise ValueError("lead_gain must be >= 0")
        if float(self.lead_max_lead_rad) <= 0:
            raise ValueError("lead_max_lead_rad must be > 0")
        if float(self.lead_smooth_tau_s) < 0:
            raise ValueError("lead_smooth_tau_s must be >= 0")
        filt = str(self.arm_chunk_filter or "none").lower()
        if filt not in ("none", "zero_phase_dct", "zero_phase_dct_cross"):
            raise ValueError(f"unknown arm_chunk_filter {self.arm_chunk_filter!r}")
        self.arm_chunk_filter = filt
        if float(self.arm_chunk_filter_cutoff_hz) <= 0:
            raise ValueError("arm_chunk_filter_cutoff_hz must be > 0")
        if int(self.arm_chunk_filter_context_frames) < 0:
            raise ValueError("arm_chunk_filter_context_frames must be >= 0")
        interp = str(self.arm_interpolation or "linear").lower()
        if interp not in ("linear", "pchip"):
            raise ValueError(f"unknown arm_interpolation {self.arm_interpolation!r}")
        self.arm_interpolation = interp

    def lead_tracker_config(self) -> LeadTrackerConfig:
        return LeadTrackerConfig(
            tau_s=float(self.lead_tau_s),
            gain=float(self.lead_gain),
            max_lead_rad=float(self.lead_max_lead_rad),
            lead_smooth_tau_s=float(self.lead_smooth_tau_s),
            hold_during_blend=bool(self.lead_hold_during_blend),
            closed_grip_max_lead_rad=(
                None
                if self.lead_closed_grip_max_lead_rad is None
                else float(self.lead_closed_grip_max_lead_rad)
            ),
        )


@dataclass
class InferenceLoopSummary:
    control_steps: int = 0
    infer_calls: int = 0
    empty_buffer_ticks: int = 0
    early_idle_s: float = 0.0
    starved_events: int = 0
    infer_intervals_s: list[float] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def _arm_ref_vel(
    waypoints: np.ndarray, step_dt: np.ndarray, progress: float
) -> tuple[np.ndarray, np.ndarray]:
    """14-DoF arm position at progress and finite-diff velocity (policy layout 8:22)."""
    wp = np.asarray(waypoints, dtype=np.float64)
    h = int(wp.shape[0])
    p = float(np.clip(progress, 0.0, max(0.0, h - 1e-9)))
    i0 = int(np.floor(p))
    i1 = min(i0 + 1, h - 1)
    frac = p - i0
    arm0 = wp[i0, 8:22]
    arm1 = wp[i1, 8:22]
    arm_now = (1.0 - frac) * arm0 + frac * arm1
    seg = float(step_dt[min(i0, int(step_dt.shape[0]) - 1)])
    if seg <= 1e-9:
        vel = np.zeros(14, dtype=np.float64)
    else:
        vel = (arm1 - arm0) / seg
    # Look-ahead ~1 model step for soft lead.
    p_ahead = min(float(h - 1), p + 1.0)
    j0 = int(np.floor(p_ahead))
    j1 = min(j0 + 1, h - 1)
    f2 = p_ahead - j0
    arm_ahead = (1.0 - f2) * wp[j0, 8:22] + f2 * wp[j1, 8:22]
    return arm_ahead.astype(np.float64), vel.astype(np.float64)


def _frame_times_s(origin: float, step_dt: np.ndarray) -> np.ndarray:
    dt = np.asarray(step_dt, dtype=np.float64).reshape(-1)
    edges = np.concatenate([[0.0], np.cumsum(dt)])
    return float(origin) + edges


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = min(a.shape[0], b.shape[0])
    return float(np.max(np.abs(a[:n] - b[:n])))


class InferenceLoop:
    """Runs a control tick loop and a serial inference thread for ``duration_s``."""

    def __init__(
        self,
        *,
        platform: RobotPlatform,
        backend: _Backend,
        plan_buffer: PlanBuffer,
        retimer: UniformRetimer | TimeAxisQP | None = None,
        config: InferenceLoopConfig | None = None,
        recorder: TraceRecorder | None = None,
        lead_tracker: LeadTracker | None = None,
        mpc_tracker: AcadosMpcTracker | None = None,
        lag_estimator: LagStateEstimator | None = None,
    ) -> None:
        self.platform = platform
        self.backend = backend
        self.plan_buffer = plan_buffer
        self.retimer = retimer or UniformRetimer()
        self.config = config or InferenceLoopConfig()
        self.recorder = recorder
        self._mapper = LayoutMapper()
        self._latency_s: deque[float] = deque(maxlen=self.config.latency_window)
        self._r2r_ms: deque[float] = deque(maxlen=self.config.latency_window)
        self._mpc = mpc_tracker
        self._estimator = lag_estimator
        tracker = self.config.tracker
        if self._mpc is not None:
            tracker = "mpc"
        lead_cfg = self.config.lead_tracker_config()
        if lead_tracker is not None:
            self._lead = lead_tracker
        elif tracker == "lead" or (tracker == "mpc" and self._mpc is None):
            self._lead = LeadTracker(lead_cfg)
        elif self.config.use_lead_tracker and tracker != "none":
            self._lead = LeadTracker(lead_cfg)
        else:
            self._lead = None
        if tracker == "mpc" and self._mpc is None:
            # Construct default MPC (may fall back to Lead if acados missing).
            from fastervla.deploy.client.tracker.acados_mpc import (
                AcadosMpcTrackerConfig,
            )

            # Prefer GDK soft limits over ±π so lead cannot trip servo clip abort.
            mpc_kw: dict[str, Any] = {"eager_build": True}
            jl = getattr(self.platform, "joint_limits", None)
            if callable(jl):
                lo21, hi21 = jl()
                lo21 = np.asarray(lo21, dtype=np.float64).reshape(-1)
                hi21 = np.asarray(hi21, dtype=np.float64).reshape(-1)
                if lo21.shape[0] >= 19 and hi21.shape[0] >= 19:
                    mpc_kw["q_min"] = lo21[5:19].copy()
                    mpc_kw["q_max"] = hi21[5:19].copy()
            self._mpc = AcadosMpcTracker(
                AcadosMpcTrackerConfig(**mpc_kw),
                lead=self._lead or LeadTracker(lead_cfg),
            )
        if tracker == "mpc" and self._estimator is None:
            self._estimator = LagStateEstimator(
                LagStateEstimatorConfig(dim=14, dt_s=1.0 / self.config.control_hz)
            )
        self._grip_hold: np.ndarray | None = None
        self._prev_filtered: np.ndarray | None = None
        self.plan_buffer.interpolation = self.config.arm_interpolation

    def _filter_chunk(self, chunk: Chunk) -> tuple[Chunk, dict[str, Any] | None]:
        """ZP path smooth before retimer (design: ZP → Retimer)."""
        cfg = self.config
        filtered, meta = apply_arm_chunk_filter(
            np.asarray(chunk.waypoints, dtype=np.float32),
            mode=cfg.arm_chunk_filter,
            cutoff_hz=float(cfg.arm_chunk_filter_cutoff_hz),
            fps=float(cfg.policy_fps),
            prev_filtered=self._prev_filtered,
            context_frames=int(cfg.arm_chunk_filter_context_frames),
        )
        if meta is not None:
            self._prev_filtered = np.asarray(filtered, dtype=np.float32).copy()
        out = Chunk(
            waypoints=filtered,
            model_dt=float(chunk.model_dt),
            obs_ts=float(chunk.obs_ts),
            meta=dict(chunk.meta),
        )
        if meta is not None:
            out.meta["arm_chunk_filter"] = meta
        return out, meta

    def _retime_chunk(
        self,
        chunk: Chunk,
        *,
        state: np.ndarray,
        lock_durations: np.ndarray | None = None,
    ) -> TimedChunk:
        retimer = self.retimer
        if isinstance(retimer, TimeAxisQP):
            lock = (
                lock_durations
                if self.config.qp_prefix_lock and lock_durations is not None and len(lock_durations)
                else None
            )
            return retimer.retime(
                chunk,
                state=state,
                lock_durations=lock,
                r2r_samples_ms=list(self._r2r_ms),
            )
        return retimer.retime(chunk)

    def run(self) -> InferenceLoopSummary:
        cfg = self.config
        summary = InferenceLoopSummary()
        stop = threading.Event()
        last_infer_mono = {"t": 0.0}
        last_switch_mono = {"t": 0.0}
        # Progress (model frames) at last accepted merge / seed load — 4+4 gate.
        last_switch_progress = {"p": float(self.plan_buffer.progress)}
        shared: dict[str, Any] = {
            "last_command21": None,
            "packet_id": 0,
        }
        shared_lock = threading.Lock()

        if self.recorder is not None:
            self.recorder.set_experiment(
                control_hz=cfg.control_hz,
                policy_fps=cfg.policy_fps,
                kappa=cfg.kappa,
                rtc_mode=cfg.rtc_mode,
                use_lead_tracker=bool(self._lead is not None),
                tracker=cfg.tracker,
                lead_tau_s=float(cfg.lead_tau_s),
                lead_gain=float(cfg.lead_gain),
                lead_max_lead_rad=float(cfg.lead_max_lead_rad),
                lead_smooth_tau_s=float(cfg.lead_smooth_tau_s),
                lead_hold_during_blend=bool(cfg.lead_hold_during_blend),
                hold_gripper=bool(cfg.hold_gripper),
                execution_prefix_frames=int(cfg.execution_prefix_frames),
                execution_buffer_frames=int(cfg.execution_buffer_frames),
                arm_chunk_filter=str(cfg.arm_chunk_filter),
                arm_chunk_filter_cutoff_hz=float(cfg.arm_chunk_filter_cutoff_hz),
                arm_interpolation=str(cfg.arm_interpolation),
                mpc_has_acados=bool(self._mpc.has_acados) if self._mpc else False,
                horizon=cfg.horizon,
                retimer="qp" if isinstance(self.retimer, TimeAxisQP) else "uniform",
                qp_prefix_lock=bool(cfg.qp_prefix_lock),
                qp_lock_extra_frames=int(cfg.qp_lock_extra_frames),
            )
            if isinstance(self.retimer, TimeAxisQP):
                qc = self.retimer.config
                self.recorder.set_experiment(
                    qp_contact_kappa=qc.contact_kappa,
                    qp_contact_window_frames=int(qc.contact_window_frames),
                    qp_transport_kappa_max=qc.transport_kappa_max,
                    qp_dwell_kappa_max=qc.dwell_kappa_max,
                    qp_dwell_step_rad=float(qc.dwell_step_rad),
                )
            self.recorder.track_warn_threshold_rad = float(cfg.track_warn_threshold_rad)

        def _latency_hat() -> float:
            margin = cfg.latency_margin_ms / 1000.0
            if not self._latency_s:
                return cfg.cold_start_latency_ms / 1000.0 + margin
            arr = np.asarray(self._latency_s, dtype=np.float64)
            return float(np.percentile(arr, 95)) + margin

        def _prefix_replan_due() -> bool:
            """Old-stack prefetch_due: prefix frames consumed, or buffer nearly empty."""
            prefix_f = int(cfg.execution_prefix_frames)
            if prefix_f <= 0:
                return True
            adv = float(self.plan_buffer.progress) - float(last_switch_progress["p"])
            if adv >= float(prefix_f) - 1e-9:
                return True
            # Early escape aligned with execution_buffer_frames (avoid empty buffer).
            buf_f = int(cfg.execution_buffer_frames)
            rem = float(self.plan_buffer.remaining_steps())
            return rem <= float(max(buf_f, 1)) + 1e-9

        def _wait_prefix_gate() -> None:
            while not stop.is_set() and not _prefix_replan_due():
                time.sleep(0.005)

        def _infer_once() -> None:
            _wait_prefix_gate()
            if stop.is_set():
                return
            t_wall0 = time.monotonic()
            t_req = t_wall0
            if last_infer_mono["t"] > 0:
                gap = t_req - last_infer_mono["t"]
                wait = cfg.t_infer_min_s - gap
                if wait > 0:
                    time.sleep(wait)
                    t_req = time.monotonic()

            if last_switch_mono["t"] > 0 and cfg.t_switch_min_s > 0:
                since = t_req - last_switch_mono["t"]
                if since < cfg.t_switch_min_s:
                    time.sleep(cfg.t_switch_min_s - since)
                    t_req = time.monotonic()
            # Re-check after sleeps (progress may still be short if control stalled).
            _wait_prefix_gate()
            if stop.is_set():
                return
            if not _prefix_replan_due():
                return
            L_hat = _latency_hat()
            d_hat = self.plan_buffer.estimate_delay_steps(L_hat)
            remaining = self.plan_buffer.remaining_steps()
            starved = d_hat > remaining + 1e-9
            if starved:
                summary.starved_events += 1
                if self.recorder is not None:
                    self.recorder.bump_counter("starved_events", 1)

            prefix = self.plan_buffer.prefix(max(1, d_hat))
            # Snapshot old plan for overlap / guided gap before merge.
            old_wp = self.plan_buffer.waypoints.copy()
            old_dt = self.plan_buffer.step_dt.copy()
            old_progress = float(self.plan_buffer.progress)

            st = self.platform.state.read()
            t_after_state = time.monotonic()
            imgs = self.platform.images.read()
            t_after_img = time.monotonic()
            obs_age_ms = max(0.0, (t_req - float(st.timestamp)) * 1000.0)
            image_age_ms = max(0.0, (t_req - float(st.timestamp)) * 1000.0)
            obs = Observation(
                images=imgs,
                image_ts=st.timestamp,
                state=np.asarray(st.q, dtype=np.float32),
                state_ts=st.timestamp,
                prompt="",
            )

            h = cfg.horizon
            leftover = np.zeros((h, cfg.action_dim), dtype=np.float32)
            start = int(np.floor(old_progress))
            n_left = min(h, max(0, old_wp.shape[0] - start))
            if n_left > 0:
                leftover[:n_left] = old_wp[start : start + n_left].astype(np.float32)
            exec_horizon = max(0, h - start)

            rtc = None
            rtc_mode = str(cfg.rtc_mode).upper()
            if rtc_mode == "INPAINT":
                rtc = {
                    "prev_chunk_left_over": leftover,
                    "inference_delay": int(max(0, d_hat)),
                    "execution_horizon": int(exec_horizon),
                }

            if last_infer_mono["t"] > 0:
                summary.infer_intervals_s.append(t_req - last_infer_mono["t"])
            last_infer_mono["t"] = t_req

            t_infer0 = time.monotonic()
            chunks = self.backend.sample(
                obs=obs,
                prefix=prefix,
                rtc_mode=rtc_mode,
                n=1,
                rtc=rtc,
            )
            t_done = time.monotonic()
            self._latency_s.append(t_done - t_req)
            summary.infer_calls += 1
            client_prep_ms = (t_infer0 - t_req) * 1000.0
            infer_call_ms = (t_done - t_infer0) * 1000.0
            r2r_ms = (t_done - t_req) * 1000.0

            self._r2r_ms.append(r2r_ms)
            raw_chunk = chunks[0]
            filtered_chunk, _filt_meta = self._filter_chunk(raw_chunk)
            anchor = int(prefix.anchor_step)
            n_lock = max(0, d_hat) + int(cfg.qp_lock_extra_frames)
            timed = self._retime_chunk(
                filtered_chunk,
                state=np.asarray(obs.state, dtype=np.float64),
                lock_durations=old_dt[anchor : anchor + n_lock],
            )
            ok = self.plan_buffer.merge(timed, anchor_step=prefix.anchor_step)
            t_accept = time.monotonic()
            with shared_lock:
                shared["packet_id"] = int(shared["packet_id"]) + 1
                packet_id = int(shared["packet_id"])
                last_cmd = shared["last_command21"]

            new_wp = np.asarray(timed.chunk.waypoints, dtype=np.float64)
            # Overlap: old leftover vs new chunk head (guided / RTC region).
            d_ov = int(max(0, d_hat))
            overlap_gap = _max_abs_diff(leftover[:d_ov], new_wp[:d_ov])
            # First executed frame vs last issued command (handoff jump).
            if last_cmd is None:
                prev_diff = float("nan")
            else:
                first21 = self._mapper.from_policy(
                    new_wp[0],
                    waist_baseline=np.asarray(last_cmd)[:5],
                    timestamp=t_accept,
                ).q_des
                # Arm + gripper jump vs last issued command (waist held).
                prev_diff = float(
                    np.max(np.abs(first21[5:] - np.asarray(last_cmd)[5:]))
                )

            origin = float(self.plan_buffer.origin_monotonic)
            step_dt = np.asarray(timed.step_dt, dtype=np.float64)
            frame_times = _frame_times_s(origin, step_dt)
            # Approximate handoff instant: now + remaining time in current model step.
            handoff_at = t_accept

            meta = timed.chunk.meta if timed.chunk.meta else {}
            merge = self.plan_buffer.last_merge if ok else {}
            record = {
                "packet_id": packet_id,
                "accepted": bool(ok),
                "starved": bool(starved),
                "d_hat": int(max(0, d_hat)),
                "d_actual": float(merge.get("d_actual", float("nan"))),
                "request_to_ready_ms": r2r_ms,
                "client_prep_ms": client_prep_ms,
                "infer_call_ms": infer_call_ms,
                "state_read_ms": (t_after_state - t_req) * 1000.0,
                "image_read_ms": (t_after_img - t_after_state) * 1000.0,
                "observation_age_ms": obs_age_ms,
                "overlap_gap": overlap_gap,
                "previous_command_difference": prev_diff,
                "rtc_d": int(max(0, d_hat)),
            }
            if self.recorder is not None:
                self.recorder.note_infer_record(record)
                if not ok:
                    self.recorder.bump_counter("rejected_merges", 1)
                else:
                    last_switch_mono["t"] = t_accept
                    last_switch_progress["p"] = float(self.plan_buffer.progress)
                    self.recorder.record_chunk(
                        actions=new_wp.astype(np.float32),
                        actions_raw=np.asarray(raw_chunk.waypoints, dtype=np.float32),
                        requested_monotonic_s=t_req,
                        ready_monotonic_s=t_done,
                        observation_monotonic_s=float(st.timestamp),
                        accepted_monotonic_s=t_accept,
                        action_origin_monotonic_s=origin,
                        handoff_at_monotonic_s=handoff_at,
                        d_hat=int(max(0, d_hat)),
                        d_actual=float(merge.get("d_actual", 0.0)),
                        anchor_step=int(
                            merge.get("anchor_step", prefix.anchor_step)
                        ),
                        plan_id=int(merge.get("plan_id", self.plan_buffer.plan_id)),
                        rtc_applied=bool(meta.get("rtc_applied", rtc is not None)),
                        rtc_d=int(max(0, d_hat)),
                        rtc_s=int(rtc.get("execution_horizon", -1)) if rtc else -1,
                        rtc_guided_gap=overlap_gap,
                        overlap_s=start,
                        overlap_d=d_ov,
                        overlap_gap=overlap_gap,
                        observation_age_ms=obs_age_ms,
                        image_age_ms=image_age_ms,
                        observation_state=np.asarray(st.q, dtype=np.float64),
                        retime_status=str(meta.get("retime_status", "uniform")),
                        step_dt=step_dt,
                        frame_times_s=frame_times,
                        previous_command_difference=prev_diff,
                        request_to_ready_ms=r2r_ms,
                        client_prep_ms=client_prep_ms,
                        infer_call_ms=infer_call_ms,
                        merge_accepted=True,
                        packet_id=packet_id,
                        start_frame=start,
                    )
            elif ok:
                last_switch_mono["t"] = t_accept
                last_switch_progress["p"] = float(self.plan_buffer.progress)

        # Floor between infer attempts; 4+4 prefix gate is the main replan throttle.
        infer_period_s = max(1.0 / float(cfg.policy_fps), float(cfg.t_infer_min_s))

        def infer_thread() -> None:
            next_infer = time.monotonic()
            while not stop.is_set():
                delay = next_infer - time.monotonic()
                if delay > 0.0 and stop.wait(delay):
                    break
                try:
                    _infer_once()
                except Exception:
                    if stop.is_set():
                        break
                    time.sleep(0.01)
                next_infer = time.monotonic() + infer_period_s

        th = threading.Thread(target=infer_thread, name="fastervla-infer", daemon=True)
        th.start()

        state0 = self.platform.state.read()
        waist = np.asarray(state0.q[:5], dtype=np.float64).copy()
        plant = getattr(self.platform, "_plant", None)
        dt = 1.0 / cfg.control_hz
        if plant is not None and hasattr(plant, "dt_s"):
            plant.dt_s = dt

        t_end = time.monotonic() + float(cfg.duration_s)
        next_tick = time.monotonic()
        try:
            while time.monotonic() < t_end:
                t_tick0 = time.monotonic()
                self.plan_buffer.advance(dt, alpha=1.0)
                empty = self.plan_buffer.remaining_steps() <= 1e-9
                if empty:
                    summary.empty_buffer_ticks += 1
                    if self.recorder is not None:
                        self.recorder.bump_counter("buffer_empty_cycles", 1)

                q24 = self.plan_buffer.reference(
                    t_now=time.monotonic(), horizon_s=0.0
                )
                ref_cmd = self._mapper.from_policy(
                    q24, waist_baseline=waist, timestamp=time.monotonic()
                )
                reference21 = np.asarray(ref_cmd.q_des, dtype=np.float64)
                tracker_status = TRACKER_PASSTHROUGH
                t_read0 = time.monotonic()
                st_now = self.platform.state.read()
                state_read_ms = (time.monotonic() - t_read0) * 1000.0
                measured = np.asarray(st_now.q, dtype=np.float64)
                t_trk0 = time.monotonic()
                if self._mpc is not None and self._estimator is not None and not empty:
                    lo, hi = ARM_POLICY_SLICE
                    arm_meas = measured[lo:hi]
                    self._estimator.push_observation(t=t_tick0, q=arm_meas)
                    q_hat = self._estimator.estimate(t_now=t_tick0)
                    wp = self.plan_buffer.waypoints
                    step_dt_arr = self.plan_buffer.step_dt
                    arm_ahead, arm_vel = _arm_ref_vel(
                        wp, step_dt_arr, float(self.plan_buffer.progress)
                    )
                    command21 = self._mpc.step(
                        q_hat=q_hat,
                        reference=reference21,
                        ref_ahead=arm_ahead,
                        ref_vel=arm_vel,
                        now=t_tick0,
                        blending=False,
                    )
                    self._estimator.push_command(t=t_tick0, y=command21[5:19])
                    if self._mpc.last_status == "ok":
                        tracker_status = TRACKER_MPC
                    else:
                        tracker_status = TRACKER_FALLBACK
                        summary.extras["mpc_fallback_count"] = int(
                            self._mpc.fallback_count
                        )
                        summary.extras["mpc_last_status"] = self._mpc.last_status
                elif self._lead is not None and not empty:
                    wp = self.plan_buffer.waypoints
                    step_dt_arr = self.plan_buffer.step_dt
                    arm_ahead, arm_vel = _arm_ref_vel(
                        wp, step_dt_arr, float(self.plan_buffer.progress)
                    )
                    command21 = self._lead.step(
                        reference21,
                        ref_ahead=arm_ahead,
                        ref_vel=arm_vel,
                        now=t_tick0,
                        blending=False,
                    )
                    tracker_status = TRACKER_LEAD
                else:
                    command21 = reference21.copy()
                tracker_solve_ms = (time.monotonic() - t_trk0) * 1000.0

                if self.config.hold_gripper:
                    # Command dims [-2:] are left/right grippers (policy 22,23).
                    if self._grip_hold is None:
                        self._grip_hold = np.asarray(
                            measured[COMMAND_INDICES][-2:], dtype=np.float64
                        ).copy()
                    command21[-2:] = self._grip_hold

                # Saturate to platform soft limits before send (MPC/Lead may lead past).
                sat_excess = 0.0
                jl = getattr(self.platform, "joint_limits", None)
                if callable(jl):
                    lo21, hi21 = jl()
                    clipped = np.clip(
                        command21,
                        np.asarray(lo21, dtype=np.float64),
                        np.asarray(hi21, dtype=np.float64),
                    )
                    sat_excess = float(np.max(np.abs(command21 - clipped)))
                    command21 = clipped

                send_cmd = ref_cmd
                send_cmd.q_des = command21.copy()
                t_send0 = time.monotonic()
                self.platform.command.send(send_cmd)
                send_ms = (time.monotonic() - t_send0) * 1000.0
                summary.control_steps += 1
                with shared_lock:
                    shared["last_command21"] = command21.copy()

                if self.recorder is not None:
                    servo_ms = (time.monotonic() - t_tick0) * 1000.0
                    # |cmd − meas| on shared command dims.
                    meas21 = measured[COMMAND_INDICES]
                    err = float(np.max(np.abs(command21 - meas21)))
                    warn = self.recorder.note_tracking_error(err, now=t_tick0)
                    if warn:
                        summary.extras["last_tracking_warning_rad"] = err
                    self.recorder.record_tick(
                        monotonic_s=t_tick0,
                        measured24=measured,
                        reference21=reference21,
                        command21=command21,
                        progress=float(self.plan_buffer.progress),
                        plan_id=int(self.plan_buffer.plan_id),
                        stage=STAGE_BUFFER_EMPTY_HOLD if empty else STAGE_STEP,
                        tracker_status=tracker_status,
                        action_frame=int(np.floor(self.plan_buffer.progress)),
                        accepted_prediction=int(self.plan_buffer.plan_id),
                        servo_tick_ms=servo_ms,
                        aux={
                            "measured_velocity24": st_now.qd,
                            "measured_effort24": st_now.extras.get("effort"),
                            "state_stamp_ms": st_now.extras.get("stamp_ms"),
                            "state_wall_ms": st_now.extras.get("wall_ms"),
                            "state_read_ms": state_read_ms,
                            "send_ms": send_ms,
                            "tracker_solve_ms": tracker_solve_ms,
                            "sat_excess_rad": sat_excess,
                        },
                    )

                next_tick += dt
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()
        finally:
            stop.set()
            th.join(timeout=max(1.0, cfg.cold_start_latency_ms / 1000.0 + 1.0))
            if self.recorder is not None:
                out = self.recorder.close()
                summary.extras["trace_dir"] = str(out)
                summary.extras["trace_drops"] = int(self.recorder.drops)
                summary.extras["trace_ticks"] = int(self.recorder.total_ticks)

        summary.early_idle_s = 0.0
        return summary
