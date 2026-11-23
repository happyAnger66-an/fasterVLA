"""Deploy CLI: shadow / run entrypoints for G2 live runs with TraceRecorder."""

from __future__ import annotations

import argparse
import json
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
)
from fastervla.deploy.client.runtime import require_execute_tty
from fastervla.deploy.postprocess.arm_chunk_filter import apply_arm_chunk_filter
from fastervla.deploy.postprocess.retime import (
    TimeAxisQP,
    TimeAxisQPConfig,
    UniformRetimer,
)


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


def _build_parser() -> argparse.ArgumentParser:
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
        "--tracker",
        choices=("none", "lead", "mpc"),
        default="lead",
        help="L3 tracker: passthrough / LeadTracker / AcadosMpcTracker",
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
        help="4+4 schedule: model frames to consume after accept before next replan (0=off)",
    )
    run.add_argument(
        "--buffer-frames",
        type=int,
        default=4,
        help="4+4 schedule: early-replan escape when remaining_steps <= this",
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
    run.add_argument("--t-infer-min-s", type=float, default=0.0)
    run.add_argument("--t-switch-min-s", type=float, default=0.0)

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
    return p


def _open_platform_backend(args: argparse.Namespace, *, mode: str):
    gdk_cfg = GdkEnvConfig(sdk_root=args.sdk_root, ignore_right_gripper_health=True)
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


def _make_recorder(args: argparse.Namespace, *, capacity_ticks: int) -> tuple[Path | None, TraceRecorder | None, PeriodicFlusher | None]:
    if args.no_trace:
        return None, None, None
    run_dir = make_run_dir(args.results_dir)
    rec = TraceRecorder(run_dir, capacity_ticks=capacity_ticks, horizon=30)
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
            hold_gripper=bool(args.hold_gripper),
            execution_prefix_frames=int(args.prefix_frames),
            execution_buffer_frames=int(args.buffer_frames),
            arm_chunk_filter=str(args.arm_chunk_filter),
            arm_chunk_filter_cutoff_hz=float(args.arm_filter_cutoff_hz),
            arm_interpolation=str(args.arm_interp),
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
                    "hold_gripper": bool(args.hold_gripper),
                    "prefix_frames": int(args.prefix_frames),
                    "buffer_frames": int(args.buffer_frames),
                    "arm_chunk_filter": str(args.arm_chunk_filter),
                    "arm_filter_cutoff_hz": float(args.arm_filter_cutoff_hz),
                    "arm_interp": str(args.arm_interp),
                    "qp_buffer_window": bool(getattr(args, "qp_buffer_window", False)),
                    "duration_s": args.duration_s,
                    "trace_dir": str(run_dir) if run_dir else None,
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
    args = _build_parser().parse_args(argv)
    if args.cmd == "shadow":
        require_execute_tty(mode="shadow")
        return _shadow_loop(args)
    if args.cmd == "run":
        return _run_loop(args)
    if args.cmd == "analyze":
        return _analyze_loop(args)
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
