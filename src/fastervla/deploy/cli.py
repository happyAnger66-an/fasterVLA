"""Deploy CLI: shadow / run entrypoints for G2 live runs with TraceRecorder."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from fastervla.core.clock import monotonic_s
from fastervla.core.types import Chunk, Observation, RobotCommand
from fastervla.deploy.backends.openpi_ws import OpenPiWebsocketBackend, OpenPiWsConfig
from fastervla.deploy.client.io.agibot_g2 import AgibotG2Config, AgibotG2Platform
from fastervla.deploy.client.io.gdk_real import (
    G2_JOINT_NAMES,
    GdkEnvConfig,
    close_real_gdk,
    open_real_gdk,
    reexec_with_gdk_env,
)
from fastervla.deploy.client.io.layout import LayoutMapper
from fastervla.deploy.client.inference_loop import InferenceLoop, InferenceLoopConfig
from fastervla.deploy.client.plan_buffer import PlanBuffer
from fastervla.deploy.client.recorder import (
    STAGE_STEP,
    PeriodicFlusher,
    TraceRecorder,
    make_run_dir,
    sanitize_run_label,
)
from fastervla.deploy.client.runtime import require_execute_tty
from fastervla.deploy.postprocess.arm_chunk_filter import apply_arm_chunk_filter
from fastervla.deploy.postprocess.retime import (
    TimeAxisQP,
    TimeAxisQPConfig,
    UniformRetimer,
)
from fastervla.deploy.recipe import extract_config_path, load_run_recipe


def _openpi_transport(host: str, port: int):
    import inspect

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    kwargs = {"host": host, "port": port}
    if "ping_timeout" in inspect.signature(WebsocketClientPolicy.__init__).parameters:
        kwargs["ping_timeout"] = None
    return WebsocketClientPolicy(**kwargs)


def _add_common_robot_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="10.42.1.201")
    p.add_argument("--port", type=int, default=8013)
    p.add_argument("--duration-s", type=float, default=60.0)
    p.add_argument("--sdk-root", default="/home/agi/app/gdk")
    p.add_argument(
        "--robot-aid",
        default=None,
        help="Expected GDK robot AID (default: GdkEnvConfig / g2s AID)",
    )
    p.add_argument(
        "--sdk-source-ip",
        default=None,
        help="GDK LOCATOR_IP (h7: 192.168.112.212; g2s default 10.42.1.101)",
    )
    p.add_argument(
        "--discovery-uri",
        default=None,
        help="AORTA_DISCOVERY_URI (h7: http://192.168.112.212:2379)",
    )
    p.add_argument("--prompt", default="Put the shuttlecock into the shuttlecock tube.")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--flush-every-s",
        type=float,
        default=30.0,
        help="Periodic mid-run flush interval; 0 disables (only close at end)",
    )
    p.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable robot-live TraceRecorder (not recommended)",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Label baked into run dir: robot-live-<stamp>-<hex>-<tag>/; also in experiment + run.log",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Exact run id: robot-live-<run-id>/ (must be unique); overrides auto stamp-hex",
    )
    p.add_argument(
        "--ignore-right-gripper-health",
        action="store_true",
        help="Skip GDK end_state read (legacy broken right bus); leaves grip obs at 0",
    )


def _build_parser(
    *, recipe_defaults: dict | None = None
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fastervla-deploy", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sh = sub.add_parser(
        "shadow",
        help="Live shadow: cameras + state + 5090 infer; zero servo; write robot-live trace",
    )
    _add_common_robot_args(sh)
    sh.add_argument("--infer-period-s", type=float, default=1.0)
    sh.add_argument("--max-infers", type=int, default=0, help="0 = until duration")

    run = sub.add_parser(
        "run",
        help="L1 InferenceLoop (100 Hz control + async infer) with TraceRecorder",
    )
    run.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Deploy recipe YAML (configs/deploy/*.yaml); CLI flags override",
    )
    _add_common_robot_args(run)
    run.add_argument(
        "--mode",
        choices=("shadow", "execute"),
        default="shadow",
        help="execute requires TTY and sends servo; default shadow",
    )
    run.add_argument("--kappa", type=float, default=1.0)
    run.add_argument(
        "--kappa-max",
        type=float,
        default=2.0,
        help="Hard ceiling for --kappa (default 2.0)",
    )
    run.add_argument("--kappa-min", type=float, default=1.0)
    run.add_argument(
        "--retimer",
        choices=("uniform", "qp"),
        default="uniform",
        help="L2 time axis: uniform (F2) or TimeAxisQP (F3)",
    )
    run.add_argument(
        "--qp-buffer-window",
        action="store_true",
        help="Enable optional inference-buffer wall-clock floor in TimeAxisQP",
    )
    run.add_argument(
        "--qp-contact-kappa",
        type=float,
        default=None,
        help="TimeAxisQP: cap rate at policy_fps·x within --qp-contact-window frames of a gripper change",
    )
    run.add_argument("--qp-contact-window", type=int, default=10)
    run.add_argument(
        "--qp-contact-window-scale-kappa",
        action="store_true",
        help="TimeAxisQP: multiply contact window by κ so wall-clock soft duration stays ≈constant",
    )
    run.add_argument(
        "--qp-contact-grip-delta",
        type=float,
        default=0.02,
        help="TimeAxisQP: |Δgrip| threshold to mark a contact event (raise to ignore micro-noise)",
    )
    run.add_argument(
        "--qp-transport-kappa-max",
        type=float,
        default=None,
        help="TimeAxisQP: cap every frame's rate at policy_fps·x",
    )
    run.add_argument(
        "--qp-dwell-kappa-max",
        type=float,
        default=None,
        help="TimeAxisQP: pin near-stationary non-contact frames up to policy_fps·x",
    )
    run.add_argument(
        "--qp-dwell-step-rad",
        type=float,
        default=0.005,
        help="TimeAxisQP: max |Δq|_∞ (rad) to count a frame as dwell",
    )
    run.add_argument(
        "--qp-prefix-lock",
        action="store_true",
        help="TimeAxisQP: lock first d_hat+extra frame durations to the old plan at merge",
    )
    run.add_argument("--qp-lock-extra", type=int, default=2)
    run.add_argument(
        "--tracker",
        choices=("none", "lead", "mpc"),
        default="lead",
        help="L3 tracker: passthrough / LeadTracker / AcadosMpcTracker",
    )
    run.add_argument(
        "--lead-tau-s",
        type=float,
        default=0.057,
        help="LeadTracker tau_s (soft-lead time constant)",
    )
    run.add_argument(
        "--lead-gain",
        type=float,
        default=0.25,
        help="LeadTracker gain",
    )
    run.add_argument(
        "--lead-max-lead-rad",
        type=float,
        default=0.06,
        help="LeadTracker per-joint lead clip (rad)",
    )
    run.add_argument(
        "--lead-smooth-tau-s",
        type=float,
        default=0.040,
        help="LeadTracker EMA smooth tau",
    )
    run.add_argument(
        "--lead-hold-during-blend",
        type=lambda s: str(s).strip().lower() in ("1", "true", "yes", "on"),
        default=True,
        help="Hold lead at zero during plan-blend (true/false)",
    )
    run.add_argument(
        "--lead-closed-grip-max-lead-rad",
        type=float,
        default=None,
        help="LeadTracker: tighter max_lead while right grip closed (carry sway A/B)",
    )
    run.add_argument(
        "--hold-gripper",
        action="store_true",
        help="Latch grippers to first measured (ignore policy gripper); chatter A/B",
    )
    run.add_argument(
        "--prefix-frames",
        type=int,
        default=4,
        help=(
            "Replan gate after accept: wait this many model frames before next "
            "infer (old soft 4+4). 0=RTC-native: fire next infer immediately"
        ),
    )
    run.add_argument(
        "--buffer-frames",
        type=int,
        default=4,
        help=(
            "When prefix>0: early-replan escape if remaining_steps <= this "
            "(starve guard). Unused for gating when --prefix-frames 0"
        ),
    )
    run.add_argument(
        "--arm-chunk-filter",
        choices=("none", "zero_phase_dct", "zero_phase_dct_cross"),
        default="none",
        help="Path smooth: ZP DCT on policy arms before retime (soft uses zero_phase_dct)",
    )
    run.add_argument(
        "--arm-filter-cutoff-hz",
        type=float,
        default=3.0,
        help="Arm chunk ZP cutoff (Hz); soft default 3",
    )
    run.add_argument(
        "--arm-interp",
        choices=("linear", "pchip"),
        default="linear",
        help="PlanBuffer reference interpolation (soft uses pchip)",
    )
    run.add_argument(
        "--mpc-preview",
        choices=("linear", "plan"),
        default="linear",
        help="MPC horizon ref: linear (r→r_{+1} lerp, legacy) | plan "
        "(PlanBuffer sampled at t+k·10ms, true preview)",
    )
    run.add_argument("--t-infer-min-s", type=float, default=0.0)
    run.add_argument("--t-switch-min-s", type=float, default=0.0)
    # After all add_argument: otherwise action default= clobbers recipe set_defaults.
    if recipe_defaults:
        run.set_defaults(**recipe_defaults)

    an = sub.add_parser(
        "analyze",
        help="Offline analysis: speed/sway/stability/accuracy (+ legacy hf3/r2r)",
    )
    an.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Single robot-live-<id> directory",
    )
    an.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Scan robot-live-* under this directory (when --run omitted)",
    )
    an.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON summary (default: stdout)",
    )
    an.add_argument(
        "--wide",
        action="store_true",
        help="Emit flattened wide-table rows (one run per row) instead of full summary",
    )
    an.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Optional URDF for EE speed FK (default: g2_tools resolve)",
    )
    an.add_argument(
        "--ee-stride",
        type=int,
        default=5,
        help="Subsample stride for EE FK speed (default 5 ≈20Hz at 100Hz ctrl)",
    )
    an.add_argument(
        "--min-chunks",
        type=int,
        default=1,
        help="Skip runs with fewer prediction chunks",
    )

    rep = sub.add_parser(
        "report",
        help="One-shot post-run line: ball cycles + seek dwell + control lag/EE",
    )
    rep.add_argument("--run", type=Path, required=True, help="robot-live-<id> directory")
    rep.add_argument("--out", type=Path, default=None, help="Optional JSON path")
    rep.add_argument("--urdf", type=Path, default=None)
    rep.add_argument(
        "--no-control",
        action="store_true",
        help="Skip control_quality (faster; no lag/EE)",
    )
    return p


def _gdk_env_from_args(args: argparse.Namespace) -> GdkEnvConfig:
    # Default False: read real end_states into policy obs. Legacy g2s units with a
    # broken right-end bus can pass --ignore-right-gripper-health.
    kw: dict = {
        "sdk_root": args.sdk_root,
        "ignore_right_gripper_health": bool(
            getattr(args, "ignore_right_gripper_health", False)
        ),
    }
    if getattr(args, "robot_aid", None):
        kw["robot_aid"] = str(args.robot_aid)
    if getattr(args, "sdk_source_ip", None):
        kw["sdk_source_ip"] = str(args.sdk_source_ip)
    if getattr(args, "discovery_uri", None):
        kw["discovery_uri"] = str(args.discovery_uri)
    return GdkEnvConfig(**kw)


def _open_platform_backend(args: argparse.Namespace, *, mode: str):
    gdk_cfg = _gdk_env_from_args(args)
    reexec_with_gdk_env(gdk_cfg)
    sdk, robot = open_real_gdk(gdk_cfg)
    g2_cfg = AgibotG2Config(
        robot_aid=gdk_cfg.robot_aid,
        expected_joint_names=G2_JOINT_NAMES,
        camera_names=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        control_period_s=0.02,
        read_timeout_s=0.5,
        servo_timeout_s=0.05,
        joint_stall_timeout_s=0.15,
    )
    platform = AgibotG2Platform(robot=robot, config=g2_cfg, mode=mode)
    ws_cfg = OpenPiWsConfig(
        host=args.host,
        port=args.port,
        expect_checkpoint_step=89999,
        expect_norm_stats_name="badminton651",
        expect_prompt=args.prompt,
        expect_num_steps=10,
        expect_rtc=True,
        expect_action_shape=(30, 24),
        expect_image_transport="jpeg",
        horizon=30,
        action_dim=24,
    )
    transport = _openpi_transport(args.host, args.port)
    backend = OpenPiWebsocketBackend(config=ws_cfg, transport=transport)
    backend.warmup()
    return sdk, platform, backend, robot


class _TeeTextIO:
    """Mirror writes to console + run.log (keeps TTY semantics on the console)."""

    def __init__(self, console, log_file) -> None:  # noqa: ANN001
        self._console = console
        self._log = log_file

    def write(self, data: str) -> int:
        n = self._console.write(data)
        self._log.write(data)
        self._log.flush()
        return n

    def flush(self) -> None:
        self._console.flush()
        self._log.flush()

    def fileno(self) -> int:
        return self._console.fileno()

    def isatty(self) -> bool:
        return bool(self._console.isatty())

    @property
    def encoding(self) -> str:
        return getattr(self._console, "encoding", "utf-8")


def _attach_run_log(run_dir: Path) -> Path:
    """Tee stdout/stderr into ``run_dir/run.log`` for the rest of the process."""
    log_path = run_dir / "run.log"
    log_f = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 — kept open for process life
    sys.stdout = _TeeTextIO(sys.__stdout__, log_f)  # type: ignore[assignment]
    sys.stderr = _TeeTextIO(sys.__stderr__, log_f)  # type: ignore[assignment]
    return log_path


def _run_label_fields(args: argparse.Namespace, run_dir: Path | None) -> dict:
    tag = getattr(args, "tag", None)
    run_id_arg = getattr(args, "run_id", None)
    return {
        "tag": sanitize_run_label(tag) if tag else None,
        "run_id": (
            run_dir.name.removeprefix("robot-live-") if run_dir is not None else None
        ),
        "run_id_arg": sanitize_run_label(run_id_arg) if run_id_arg else None,
        "run_log": str(run_dir / "run.log") if run_dir is not None else None,
    }


def _make_recorder(args: argparse.Namespace, *, capacity_ticks: int) -> tuple[Path | None, TraceRecorder | None, PeriodicFlusher | None]:
    if args.no_trace:
        return None, None, None
    tag = getattr(args, "tag", None)
    run_id = getattr(args, "run_id", None)
    run_dir = make_run_dir(args.results_dir, tag=tag, run_id=run_id)
    log_path = _attach_run_log(run_dir)
    rec = TraceRecorder(run_dir, capacity_ticks=capacity_ticks, horizon=30)
    rid = run_dir.name.removeprefix("robot-live-")
    rec.set_experiment(
        tag=sanitize_run_label(tag) if tag else None,
        run_id=rid,
        run_id_arg=sanitize_run_label(run_id) if run_id else None,
        run_log=str(log_path),
    )
    flusher = None
    if float(args.flush_every_s) > 0:
        flusher = PeriodicFlusher(rec, every_s=float(args.flush_every_s))
        flusher.start()
    return run_dir, rec, flusher


def _shadow_loop(args: argparse.Namespace) -> int:
    sdk = None
    platform = None
    backend = None
    mapper = LayoutMapper()
    flusher: PeriodicFlusher | None = None
    recorder: TraceRecorder | None = None
    run_dir: Path | None = None

    try:
        sdk, platform, backend, robot = _open_platform_backend(args, mode="shadow")
        # Shadow is ~infer_period rate; size for duration / period + margin.
        period = max(1e-3, float(args.infer_period_s))
        n_est = int(float(args.duration_s) / period) + 32
        if args.max_infers:
            n_est = min(n_est, int(args.max_infers) + 8)
        run_dir, recorder, flusher = _make_recorder(args, capacity_ticks=max(64, n_est))

        print(
            json.dumps(
                {
                    "shadow_start": True,
                    "robot_aid": robot.aid,
                    "host": args.host,
                    "port": args.port,
                    "duration_s": args.duration_s,
                    "infer_period_s": args.infer_period_s,
                    "trace_dir": str(run_dir) if run_dir else None,
                    **_run_label_fields(args, run_dir),
                    "flush_every_s": args.flush_every_s,
                    "ignore_right_gripper_health": True,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        state0 = platform.state.read()
        waist = state0.q[:5].copy()
        t_end = monotonic_s() + float(args.duration_s)
        n_infer = 0
        commands_built = 0

        while monotonic_s() < t_end:
            if args.max_infers and n_infer >= args.max_infers:
                break
            t0 = monotonic_s()
            st = platform.state.read()
            t_state = monotonic_s()
            imgs = platform.images.read()
            t_img = monotonic_s()
            obs_age_ms = max(0.0, (t0 - float(st.timestamp)) * 1000.0)
            obs = Observation(
                images=imgs,
                image_ts=st.timestamp,
                state=st.q.astype(np.float32),
                state_ts=st.timestamp,
                prompt=args.prompt,
            )
            chunks = backend.sample(obs=obs, prefix=None, rtc_mode="NONE", n=1)
            t_infer = monotonic_s()
            waypoints = np.asarray(chunks[0].waypoints, dtype=np.float64)
            cmd: RobotCommand = mapper.from_policy(
                waypoints[0], waist_baseline=waist, timestamp=st.timestamp
            )
            platform.command.send(cmd)
            commands_built += 1
            n_infer += 1

            if recorder is not None:
                r2r_ms = (t_infer - t_img) * 1000.0
                recorder.set_experiment(
                    mode="shadow",
                    infer_period_s=float(args.infer_period_s),
                    host=args.host,
                    port=int(args.port),
                )
                recorder.record_tick(
                    monotonic_s=t0,
                    measured24=np.asarray(st.q, dtype=np.float64),
                    reference21=np.asarray(cmd.q_des, dtype=np.float64),
                    command21=np.asarray(cmd.q_des, dtype=np.float64),
                    progress=0.0,
                    plan_id=n_infer,
                    stage=STAGE_STEP,
                    action_frame=0,
                    accepted_prediction=n_infer,
                    servo_tick_ms=(monotonic_s() - t0) * 1000.0,
                )
                meta = chunks[0].meta if chunks[0].meta else {}
                recorder.record_chunk(
                    actions=waypoints.astype(np.float32),
                    actions_raw=waypoints.astype(np.float32),
                    requested_monotonic_s=t_img,
                    ready_monotonic_s=t_infer,
                    observation_monotonic_s=float(st.timestamp),
                    accepted_monotonic_s=t_infer,
                    action_origin_monotonic_s=t_infer,
                    d_hat=0,
                    d_actual=0.0,
                    plan_id=n_infer,
                    packet_id=n_infer,
                    rtc_applied=bool(meta.get("rtc_applied", False)),
                    rtc_d=0,
                    observation_age_ms=obs_age_ms,
                    image_age_ms=obs_age_ms,
                    observation_state=np.asarray(st.q, dtype=np.float64),
                    retime_status="shadow_none",
                    request_to_ready_ms=r2r_ms,
                    client_prep_ms=(t_img - t0) * 1000.0,
                    infer_call_ms=r2r_ms,
                    merge_accepted=True,
                )
                recorder.note_infer_record(
                    {
                        "packet_id": n_infer,
                        "accepted": True,
                        "request_to_ready_ms": r2r_ms,
                        "observation_age_ms": obs_age_ms,
                    }
                )

            tick = {
                "i": n_infer,
                "state_ms": round((t_state - t0) * 1000, 1),
                "image_ms": round((t_img - t_state) * 1000, 1),
                "infer_ms": round((t_infer - t_img) * 1000, 1),
                "r2r_ms": round((t_infer - t_img) * 1000, 1),
            }
            print(json.dumps({"shadow_tick": tick}, ensure_ascii=False), flush=True)

            sleep_s = float(args.infer_period_s) - (monotonic_s() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

        summary = {
            "shadow_done": True,
            "infer_calls": n_infer,
            "commands_built": commands_built,
            "commands_sent_hw": 0,
            "duration_s": round(monotonic_s() - (t_end - float(args.duration_s)), 2),
            "trace_dir": str(run_dir) if run_dir else None,
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    finally:
        if flusher is not None:
            flusher.stop()
        if recorder is not None:
            out = recorder.close()
            print(
                json.dumps(
                    {
                        "trace_closed": True,
                        "trace_dir": str(out),
                        "trace_drops": recorder.drops,
                        "trace_ticks": recorder.total_ticks,
                        "periodic_flushes": flusher.flush_count if flusher else 0,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if platform is not None:
            platform.close()
        if backend is not None:
            backend.close()
        if sdk is not None:
            close_real_gdk(sdk)


def _run_loop(args: argparse.Namespace) -> int:
    require_execute_tty(mode=args.mode)
    sdk = None
    platform = None
    backend = None
    flusher: PeriodicFlusher | None = None
    recorder: TraceRecorder | None = None
    run_dir: Path | None = None

    try:
        sdk, platform, backend, robot = _open_platform_backend(args, mode=args.mode)
        capacity = int(float(args.duration_s) * 100.0) + 200
        run_dir, recorder, flusher = _make_recorder(args, capacity_ticks=capacity)

        # Seed plan so control never starts empty.
        st0 = platform.state.read()
        imgs0 = platform.images.read()
        seed_obs = Observation(
            images=imgs0,
            image_ts=st0.timestamp,
            state=st0.q.astype(np.float32),
            state_ts=st0.timestamp,
            prompt=args.prompt,
        )
        seed = backend.sample(obs=seed_obs, prefix=None, rtc_mode="NONE", n=1)[0]
        seed_wp, seed_filt_meta = apply_arm_chunk_filter(
            np.asarray(seed.waypoints, dtype=np.float32),
            mode=str(args.arm_chunk_filter),
            cutoff_hz=float(args.arm_filter_cutoff_hz),
            fps=30.0,
        )
        seed = Chunk(
            waypoints=seed_wp,
            model_dt=float(seed.model_dt),
            obs_ts=float(seed.obs_ts),
            meta=dict(seed.meta),
        )
        if seed_filt_meta is not None:
            seed.meta["arm_chunk_filter"] = seed_filt_meta
        if args.retimer == "qp":
            retimer: UniformRetimer | TimeAxisQP = TimeAxisQP(
                TimeAxisQPConfig(
                    policy_fps=30.0,
                    kappa=float(args.kappa),
                    kappa_max=float(args.kappa_max),
                    kappa_min=float(args.kappa_min),
                    buffer_window_enabled=bool(args.qp_buffer_window),
                    contact_kappa=args.qp_contact_kappa,
                    contact_window_frames=int(args.qp_contact_window),
                    contact_window_scale_kappa=bool(args.qp_contact_window_scale_kappa),
                    contact_grip_delta=float(args.qp_contact_grip_delta),
                    transport_kappa_max=args.qp_transport_kappa_max,
                    dwell_kappa_max=args.qp_dwell_kappa_max,
                    dwell_step_rad=float(args.qp_dwell_step_rad),
                )
            )
            seed_timed = retimer.retime(seed, state=np.asarray(st0.q, dtype=np.float64))
        else:
            retimer = UniformRetimer(
                policy_fps=30.0,
                kappa=float(args.kappa),
                kappa_max=float(args.kappa_max),
                kappa_min=float(args.kappa_min),
            )
            seed_timed = retimer.retime(seed)
        buf = PlanBuffer(interpolation=str(args.arm_interp))
        buf.load(seed_timed, origin_monotonic=monotonic_s())

        cfg = InferenceLoopConfig(
            control_hz=100.0,
            policy_fps=30.0,
            kappa=float(args.kappa),
            duration_s=float(args.duration_s),
            t_infer_min_s=float(args.t_infer_min_s),
            t_switch_min_s=float(args.t_switch_min_s),
            rtc_mode="INPAINT",
            tracker=str(args.tracker),
            use_lead_tracker=str(args.tracker) != "none",
            lead_tau_s=float(args.lead_tau_s),
            lead_gain=float(args.lead_gain),
            lead_max_lead_rad=float(args.lead_max_lead_rad),
            lead_smooth_tau_s=float(args.lead_smooth_tau_s),
            lead_hold_during_blend=bool(args.lead_hold_during_blend),
            lead_closed_grip_max_lead_rad=args.lead_closed_grip_max_lead_rad,
            hold_gripper=bool(args.hold_gripper),
            execution_prefix_frames=int(args.prefix_frames),
            execution_buffer_frames=int(args.buffer_frames),
            arm_chunk_filter=str(args.arm_chunk_filter),
            arm_chunk_filter_cutoff_hz=float(args.arm_filter_cutoff_hz),
            arm_interpolation=str(args.arm_interp),
            mpc_preview=str(args.mpc_preview),
            qp_prefix_lock=bool(args.qp_prefix_lock),
            qp_lock_extra_frames=int(args.qp_lock_extra),
        )
        print(
            json.dumps(
                {
                    "run_start": True,
                    "mode": args.mode,
                    "robot_aid": robot.aid,
                    "kappa": args.kappa,
                    "retimer": args.retimer,
                    "tracker": args.tracker,
                    "lead_tau_s": float(args.lead_tau_s),
                    "lead_gain": float(args.lead_gain),
                    "lead_max_lead_rad": float(args.lead_max_lead_rad),
                    "lead_smooth_tau_s": float(args.lead_smooth_tau_s),
                    "lead_hold_during_blend": bool(args.lead_hold_during_blend),
                    "lead_closed_grip_max_lead_rad": args.lead_closed_grip_max_lead_rad,
                    "hold_gripper": bool(args.hold_gripper),
                    "prefix_frames": int(args.prefix_frames),
                    "buffer_frames": int(args.buffer_frames),
                    "arm_chunk_filter": str(args.arm_chunk_filter),
                    "arm_filter_cutoff_hz": float(args.arm_filter_cutoff_hz),
                    "arm_interp": str(args.arm_interp),
                    "mpc_preview": str(args.mpc_preview),
                    "qp_buffer_window": bool(getattr(args, "qp_buffer_window", False)),
                    "qp_contact_kappa": args.qp_contact_kappa,
                    "qp_contact_window": int(args.qp_contact_window),
                    "qp_contact_window_scale_kappa": bool(
                        args.qp_contact_window_scale_kappa
                    ),
                    "qp_contact_grip_delta": float(args.qp_contact_grip_delta),
                    "qp_transport_kappa_max": args.qp_transport_kappa_max,
                    "qp_dwell_kappa_max": args.qp_dwell_kappa_max,
                    "qp_dwell_step_rad": float(args.qp_dwell_step_rad),
                    "qp_prefix_lock": bool(args.qp_prefix_lock),
                    "qp_lock_extra": int(args.qp_lock_extra),
                    "duration_s": args.duration_s,
                    "trace_dir": str(run_dir) if run_dir else None,
                    **_run_label_fields(args, run_dir),
                    "flush_every_s": args.flush_every_s,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        loop = InferenceLoop(
            platform=platform,
            backend=backend,
            plan_buffer=buf,
            retimer=retimer,
            config=cfg,
            recorder=recorder,
        )
        if seed_filt_meta is not None:
            loop._prev_filtered = np.asarray(seed_wp, dtype=np.float32).copy()
        # InferenceLoop.close()'s recorder itself; avoid double-close in finally.
        summary = loop.run()
        recorder = None  # already closed inside loop
        mpc_info: dict = {}
        if getattr(loop, "_mpc", None) is not None:
            mpc = loop._mpc
            mpc_info = {
                "has_acados": bool(mpc.has_acados),
                "fallback_count": int(mpc.fallback_count),
                "last_status": mpc.last_status,
                "build_error": getattr(mpc, "_build_error", None),
                "preview": getattr(mpc.config, "preview", None),
            }
        print(
            json.dumps(
                {
                    "run_done": True,
                    "mode": args.mode,
                    "tracker": args.tracker,
                    "control_steps": summary.control_steps,
                    "infer_calls": summary.infer_calls,
                    "empty_buffer_ticks": summary.empty_buffer_ticks,
                    "starved_events": summary.starved_events,
                    "extras": summary.extras,
                    "mpc": mpc_info,
                    "periodic_flushes": flusher.flush_count if flusher else 0,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    finally:
        if flusher is not None:
            flusher.stop()
        if recorder is not None:
            recorder.close()
        if platform is not None:
            platform.close()
        if backend is not None:
            backend.close()
        if sdk is not None:
            close_real_gdk(sdk)


def _analyze_loop(args: argparse.Namespace) -> int:
    from fastervla.deploy.analysis import analyze_results_dir, analyze_run

    urdf = getattr(args, "urdf", None)
    ee_stride = int(getattr(args, "ee_stride", 5))
    wide = bool(getattr(args, "wide", False))

    if args.run is not None:
        summary = analyze_run(
            args.run,
            min_chunks=int(args.min_chunks),
            urdf=urdf,
            ee_stride=ee_stride,
        )
        payload: dict | list = summary["wide"] if wide else summary
        print(
            json.dumps(
                {
                    "run_dir": summary.get("run_dir"),
                    "headline": summary.get("headline"),
                    "wide": summary.get("wide") if not wide else None,
                },
                ensure_ascii=False,
                default=str,
            ),
            flush=True,
        )
    else:
        rows = analyze_results_dir(
            args.results_dir,
            min_chunks=int(args.min_chunks),
            urdf=urdf,
            ee_stride=ee_stride,
            wide_only=wide,
        )
        payload = rows
        for row in rows:
            if "error" in row:
                print(
                    json.dumps(
                        {"run_dir": row.get("run_dir"), "error": row["error"]},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            elif wide:
                print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
            else:
                print(
                    json.dumps(
                        {
                            "run_dir": row.get("run_dir"),
                            "headline": row.get("headline"),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                    flush=True,
                )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(json.dumps({"wrote": str(args.out)}, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Entry-point calls main() with argv=None; must read sys.argv for --config.
    argv_list = list(sys.argv[1:] if argv is None else argv)
    cfg_path = extract_config_path(argv_list)
    recipe: dict | None = None
    if cfg_path is not None:
        recipe = load_run_recipe(cfg_path)
        print(
            json.dumps(
                {"config_loaded": str(cfg_path), "recipe_keys": sorted(recipe)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    args = _build_parser(recipe_defaults=recipe).parse_args(argv_list)
    try:
        if args.cmd == "shadow":
            require_execute_tty(mode="shadow")
            return _shadow_loop(args)
        if args.cmd == "run":
            return _run_loop(args)
        if args.cmd == "analyze":
            return _analyze_loop(args)
        if args.cmd == "report":
            from fastervla.deploy.analysis.report import write_report

            write_report(
                args.run,
                out=args.out,
                control=not bool(args.no_control),
                urdf=args.urdf,
            )
            return 0
        raise SystemExit(f"unknown command {args.cmd}")
    except KeyboardInterrupt:
        # Exit 130 (128+SIGINT). Keep message short; shell must stay up.
        print(json.dumps({"interrupted": True, "signal": "SIGINT"}, ensure_ascii=False), flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
