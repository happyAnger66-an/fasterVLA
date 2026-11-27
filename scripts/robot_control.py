"""Named G2 target mapping and guarded, normal-mode GDK position servo."""
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from noise_initialization import add_noise_arguments, apply_noise_overrides

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "configs/common/robot_schema.json").read_text())
NAMES = tuple(SCHEMA["state_names"])
FINGERS = ("idx31_gripper_l_inner_joint1", "idx71_gripper_r_inner_joint1")
COMMAND_INDICES = np.asarray([*range(5), *range(8, 24)])
COMMAND_NAMES = tuple(NAMES[i] if i < 22 else FINGERS[i - 22] for i in COMMAND_INDICES)
ACTIVE_INDICES = np.asarray([*range(8, 24)])
ERROR_FIELDS = ("right_arm_error", "left_arm_error", "right_end_error", "left_end_error",
                "waist_error", "lift_error", "neck_error", "chassis_error")
# h6 mc_hal_app reactive_check.cpp:324, recorded threshold=0.300000.
# This deployment profile does not modify the manufacturer's controller.
NATIVE_ARM_ERROR_RAD = 0.3


def finite_vector(value, shape):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected finite shape {shape}, got {array.shape}")
    return array


def add_execution_arguments(parser):
    """Shared by launcher preflight and the live bridge."""
    parser.add_argument("--replan-steps", type=int, default=None, metavar="N",
                        help="Synchronous: execute N actions; async_buffer: infer after N then execute buffer; async_tail: replan at index N")
    parser.add_argument("--settle-ms", type=float, default=None, metavar="MS",
                        help="Endpoint wait in ms; policy_time: 0-500 fixed hold, feedback: (0,500] stable wait")
    parser.add_argument("--denoise-steps", type=int, default=None, metavar="N",
                        help="Online noise search: complete N-step denoising, 1-10 (reference remains 10)")
    parser.add_argument("--buffer-steps", type=int, default=None, metavar="N",
                        help="Async buffer: execute N old action frames after requesting inference; replan + buffer <= 30")
    parser.add_argument("--execution-speed", type=float, default=None, metavar="SCALE",
                        help="Policy-time playback speed in (0,1.5]; <1 slows, >1 speeds "
                             "(1.3≈30%% faster wall-clock); default 1")
    parser.add_argument("--arm-chunk-filter", default=None,
                        choices=["none", "zero_phase_dct", "zero_phase_dct_cross"],
                        help="Arm chunk post-filter; zero_phase_dct_cross stitches prev tail then ZP DCT (FP16/FP8/NVFP4)")
    parser.add_argument("--arm-chunk-filter-cutoff-hz", type=float, default=None, metavar="HZ",
                        help="Arm chunk filter cutoff Hz in [0.5, 7]; default from config (usually 3)")
    parser.add_argument("--arm-chunk-filter-context-frames", type=int, default=None, metavar="N",
                        help="Cross-chunk ZP: prev-tail frames to stitch, 3-30; default 10")
    add_noise_arguments(parser)


def apply_execution_overrides(config, *, replan_steps=None, settle_ms=None, denoise_steps=None, buffer_steps=None,
                              noise_mean_source=None, noise_mean_scale=None, noise_std=None, execution_speed=None,
                              arm_chunk_filter=None, arm_chunk_filter_cutoff_hz=None,
                              arm_chunk_filter_context_frames=None):
    """Return an effective config; never rewrite the on-disk defaults."""
    effective = dict(apply_noise_overrides(config, noise_mean_source=noise_mean_source,
                                          noise_mean_scale=noise_mean_scale, noise_std=noise_std))
    overrides = {}
    if execution_speed is not None:
        if (type(execution_speed) not in (int, float) or not math.isfinite(execution_speed)
                or not 0 < execution_speed <= 1.5):
            raise ValueError("--execution-speed must be finite and in (0,1.5]")
        if execution_speed != 1 and config.get("execution_progress") != "policy_time":
            raise ValueError("Scaled playback requires policy_time synchronous/async_buffer control")
        effective["execution_speed"] = float(execution_speed)
        overrides["execution_speed"] = float(execution_speed)
    if denoise_steps is not None:
        if type(denoise_steps) is not int or not 1 <= denoise_steps <= 10:
            raise ValueError("--denoise-steps must be an integer from 1 to 10")
        task = config.get("inference_task", {})
        if task.get("noise_mode") != "per_observation_search":
            raise ValueError("--denoise-steps requires the per-observation independent noise-search profile")
        effective["inference_task"] = dict(task, num_steps=denoise_steps)
        overrides["denoise_steps"] = denoise_steps
    if replan_steps is not None:
        if type(replan_steps) is not int or not 1 <= replan_steps <= 30:
            raise ValueError("--replan-steps must be an integer from 1 to 30")
        effective["execution_prefix_frames"] = replan_steps
        overrides["execution_prefix_frames"] = replan_steps
    if buffer_steps is not None:
        if config.get("inference_schedule") != "async_buffer":
            raise ValueError("--buffer-steps requires async_buffer scheduling")
        if type(buffer_steps) is not int or not 1 <= buffer_steps <= 29:
            raise ValueError("--buffer-steps must be an integer from 1 to 29; replan + buffer <= 30")
        effective["execution_buffer_frames"] = buffer_steps
        overrides["execution_buffer_frames"] = buffer_steps
    if effective.get("inference_schedule") == "async_buffer" and (replan_steps is not None or buffer_steps is not None):
        prefix = effective.get("execution_prefix_frames")
        buffer = effective.get("execution_buffer_frames")
        if (type(prefix) is not int or type(buffer) is not int or prefix < 1 or buffer < 1
                or prefix + buffer > 30):
            raise ValueError("--replan-steps + --buffer-steps must be <= 30, both positive integers")
    if arm_chunk_filter is not None:
        if arm_chunk_filter not in ("none", "zero_phase_dct", "zero_phase_dct_cross"):
            raise ValueError("--arm-chunk-filter must be none|zero_phase_dct|zero_phase_dct_cross")
        effective["arm_chunk_filter"] = arm_chunk_filter
        overrides["arm_chunk_filter"] = arm_chunk_filter
        if arm_chunk_filter == "zero_phase_dct_cross" and "arm_chunk_filter_context_frames" not in effective:
            effective["arm_chunk_filter_context_frames"] = 10
    if arm_chunk_filter_cutoff_hz is not None:
        hz = arm_chunk_filter_cutoff_hz
        if type(hz) not in (int, float) or not 0.5 <= float(hz) <= 7.0:
            raise ValueError("--arm-chunk-filter-cutoff-hz must be in [0.5, 7]")
        effective["arm_chunk_filter_cutoff_hz"] = float(hz)
        overrides["arm_chunk_filter_cutoff_hz"] = float(hz)
    if arm_chunk_filter_context_frames is not None:
        ctx = arm_chunk_filter_context_frames
        if type(ctx) is not int or not 3 <= ctx <= 30:
            raise ValueError("--arm-chunk-filter-context-frames must be an integer from 3 to 30")
        effective["arm_chunk_filter_context_frames"] = ctx
        overrides["arm_chunk_filter_context_frames"] = ctx
    if settle_ms is not None:
        policy_timed = config.get("execution_progress") == "policy_time"
        if (type(settle_ms) not in (int, float) or not math.isfinite(settle_ms)
                or not 0 <= settle_ms <= 500 or (settle_ms == 0 and not policy_timed)):
            raise ValueError("--settle-ms must be finite; feedback: (0, 500], policy_time: [0, 500]")
        if config.get("execution_progress") != "feedback" and not policy_timed:
            raise ValueError("--settle-ms requires feedback or policy_time execution")
        effective["settle_duration_s"] = settle_ms / 1000
        overrides["settle_duration_s"] = settle_ms / 1000
    if overrides:
        effective["runtime_overrides"] = overrides
    return effective


def execution_fps(config):
    """Playback clock; the checkpoint's training policy_fps stays unchanged."""
    return config["policy_fps"] * config.get("execution_speed", 1.0)


def validate_frame_times(frame_times, horizon=30):
    """Chunk node times relative to the action origin: T[0]=0, T[k] is where frame k-1 ends."""
    times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    if (times.shape != (horizon + 1,) or not np.all(np.isfinite(times)) or times[0] != 0.0
            or np.any(np.diff(times) <= 0)):
        raise ValueError(f"frame_times must be {horizon + 1} strictly increasing finite seconds from 0")
    return times.copy()


def frame_time(frame, fps, frame_times=None):
    """Seconds from the action origin to the start of ``frame``."""
    return frame / fps if frame_times is None else float(frame_times[frame])


def frame_at(elapsed, fps, frame_times=None):
    """Index of the frame active ``elapsed`` seconds after the action origin."""
    if frame_times is None:
        return int(elapsed * fps)
    return int(np.searchsorted(frame_times, elapsed, side="right")) - 1


def frame_position(elapsed, fps, frame_times=None):
    """Continuous frame coordinate; linear beyond the chunk ends."""
    if frame_times is None:
        return elapsed * fps
    times = np.asarray(frame_times, dtype=np.float64)
    if elapsed < 0:
        return elapsed / times[1]
    if elapsed > times[-1]:
        return len(times) - 1 + (elapsed - times[-1]) / (times[-1] - times[-2])
    return float(np.interp(elapsed, times, np.arange(len(times), dtype=np.float64)))


def validate_contract(config):
    """Refuse unverified semantics before importing the SDK or making requests."""
    task = config.get("inference_task")
    if isinstance(task, dict) and task.get("noise_mode") == "per_observation_search":
        if (config.get("inference_schedule") not in ("synchronous", "async_buffer")
                or type(task.get("num_steps")) is not int
                or not 1 <= task["num_steps"] <= 10
                or type(task.get("noise_search_candidates")) is not int or task["noise_search_candidates"] != 100
                or type(task.get("noise_search_reference_steps")) is not int or task["noise_search_reference_steps"] != 10
                or task.get("noise_search_metric") != "arms_rmse_30f_rad"):
            raise ValueError("Per-observation noise search requires synchronous/async_buffer, 100 candidates, 1-10 output steps, reference 10")
    if config.get("action_semantics") != "absolute_position_rad":
        raise ValueError("Training action semantics unverified: require absolute_position_rad")
    if config.get("camera_preprocessing") != "stretch_640x480":
        raise ValueError("Training camera preprocessing unverified")
    image_transport = config.get("image_transport", "rgb")
    if image_transport not in ("rgb", "jpeg"):
        raise ValueError("Unknown image transport")
    if image_transport == "jpeg":
        task = config.get("inference_task")
        if not isinstance(task, dict) or task.get("image_transport") != "jpeg":
            raise ValueError("JPEG transport requires an explicit matching inference_task.image_transport")
    observation_source = config.get("observation_source", "ros")
    if observation_source not in ("ros", "gdk"):
        raise ValueError("observation_source must be ros or gdk")
    if config.get("inference_worker", "thread") not in ("thread", "process"):
        raise ValueError("Unknown inference worker")
    if config.get("inference_schedule", "async_prefetch") not in ("async_prefetch", "synchronous", "async_tail", "async_buffer"):
        raise ValueError("Unknown inference schedule")
    for name in ("training_conversion_evidence", "camera_mapping_evidence", "execution_timing_evidence"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"Missing verified evidence: {name}")
    state_source = config.get("async_state_source", "measured")
    if state_source not in ("measured", "handoff_target"):
        raise ValueError("Unknown async inference state source")
    if state_source == "handoff_target" and (
            config.get("inference_schedule") != "async_buffer"
            or config.get("arm_command_interpolation", "none") != "none"
            or config.get("handoff_blend_frames", 0) != 0
            or config.get("handoff_blend_duration_s", 0) != 0
            or config.get("handoff_blend_mode") is not None):
        raise ValueError("Handoff target state requires direct, unblended async_buffer")
    rtc_cfg = config.get("inference_rtc")
    if rtc_cfg is not None:
        if not isinstance(rtc_cfg, dict) or "enabled" not in rtc_cfg or type(rtc_cfg["enabled"]) is not bool:
            raise ValueError("inference_rtc must be {\"enabled\": true|false}")
        if rtc_cfg["enabled"]:
            if config.get("inference_schedule") != "async_buffer":
                raise ValueError("RTC requires inference_schedule=async_buffer")
            if state_source != "measured":
                raise ValueError("RTC requires async_state_source=measured (handoff_target conflicts)")
    policy_fps = config.get("policy_fps")
    if type(policy_fps) not in (int, float) or policy_fps not in (15.0, 30.0):
        raise ValueError("This adapter requires policy_fps 15 or 30, matching training")
    speed = config.get("execution_speed", 1.0)
    if type(speed) not in (int, float) or not math.isfinite(speed) or not 0 < speed <= 1.5:
        raise ValueError("execution_speed must be finite and in (0,1.5]")
    if speed != 1 and config.get("execution_progress") != "policy_time":
        raise ValueError("Scaled playback requires policy_time synchronous/async_buffer control")
    expected = {"servo_hz": 100.0, "control_period_s": 0.02}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("This adapter requires 100 Hz normal servo and a 0.02 s control period")
    policy_timed = config.get("execution_progress") == "policy_time"
    bounds = {"max_observation_age_s": (0, 0.2), "max_prediction_age_s": (0, 0.8),
              "max_tracking_error_rad": (0, 0.1),
              "held_joint_tolerance_rad": (0, 0.03),
              "gripper_clip_tolerance_rad": (0, 0.01)}
    if not policy_timed:
        bounds["max_hold_s"] = (0, 2.0)
    for key, (lower, upper) in bounds.items():
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not lower < value <= upper:
            raise ValueError(f"Invalid conservative execution bound: {key}")
    prefix = config.get("execution_prefix_frames")
    arm_clip = config.get("arm_clip_tolerance_rad", 0.0)
    if config.get("prediction_limit_mode", "reject") not in ("reject", "clip"):
        raise ValueError("prediction_limit_mode must be reject or clip")
    if type(arm_clip) not in (int, float) or not math.isfinite(arm_clip) or not 0 <= arm_clip <= 0.02:
        raise ValueError("Invalid conservative execution bound: arm_clip_tolerance_rad")
    if type(prefix) is not int or not 1 <= prefix <= 30:
        raise ValueError("Execution prefix must contain 1 to 30 frames")
    filter_mode = config.get("arm_command_filter", "direct")
    if filter_mode not in ("direct", "bounded"):
        raise ValueError("Unknown arm command filter")
    if config.get("arm_smoothing", "critical_damping") not in ("critical_damping", "none"):
        raise ValueError("Unknown arm smoothing mode")
    chunk_filter = config.get("arm_chunk_filter", "none")
    if chunk_filter not in ("none", "zero_phase_dct", "zero_phase_dct_cross"):
        raise ValueError("Unknown arm_chunk_filter")
    if chunk_filter != "none":
        cutoff = config.get("arm_chunk_filter_cutoff_hz", 3.0)
        if type(cutoff) not in (int, float) or not 0.5 <= float(cutoff) <= 7.0:
            raise ValueError("arm_chunk_filter_cutoff_hz must be in [0.5, 7.0]")
        if config.get("inference_schedule") not in ("synchronous", "async_buffer", "async_prefetch", "async_tail"):
            raise ValueError("arm_chunk_filter requires a supported inference_schedule")
        if chunk_filter == "zero_phase_dct_cross":
            ctx = config.get("arm_chunk_filter_context_frames", 10)
            if type(ctx) is not int or not 3 <= ctx <= 30:
                raise ValueError("arm_chunk_filter_context_frames must be an integer from 3 to 30")
    error_profile = config.get("arm_error_profile", "conservative")
    if error_profile not in ("conservative", "native_0_3"):
        raise ValueError("Unknown arm error profile")
    native_errors = error_profile == "native_0_3"
    if native_errors and (filter_mode != "bounded" or config.get("execution_progress") != "feedback"
                          or config.get("inference_schedule") != "synchronous"):
        raise ValueError("Native arm error profile requires bounded synchronous feedback control")
    if filter_mode == "bounded":
        bounds = {"arm_rate_rad_s": (0, 0.2), "arm_acceleration_rad_s2": (0, 1.0),
                  "arm_filter_time_constant_s": (0.05, 0.5),
                  "max_arm_command_lead_rad": (0, 0.1),
                  "max_arm_tracking_error_rad": (0, 0.15),
                  "arm_stall_error_rad": (0, 0.08), "arm_stall_timeout_s": (0, 2.0)}
        if native_errors:
            for key in ("max_arm_command_lead_rad", "max_arm_tracking_error_rad", "arm_stall_error_rad"):
                bounds[key] = (0, NATIVE_ARM_ERROR_RAD)
        for key, (lower, upper) in bounds.items():
            value = config.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not lower < value <= upper:
                raise ValueError(f"Invalid bounded arm control setting: {key}")
        ordered = (config["arm_stall_error_rad"] <= config["max_arm_command_lead_rad"] <= config["max_arm_tracking_error_rad"]
                   if native_errors else
                   config["arm_stall_error_rad"] < config["max_arm_command_lead_rad"] < config["max_arm_tracking_error_rad"])
        if not ordered:
            raise ValueError("Arm stall/lead/tracking bounds must be " +
                             ("nondecreasing" if native_errors else "strictly increasing"))
    progress = config.get("execution_progress", "wall_time")
    if progress not in ("wall_time", "feedback", "policy_time"):
        raise ValueError("Unknown execution progress mode")
    if policy_timed:
        if (config.get("inference_schedule") not in ("synchronous", "async_buffer") or filter_mode != "direct"
                or config.get("gripper_command_filter") != "direct"
                or config.get("require_new_observation") is not True):
            raise ValueError("Policy time requires synchronous/async_buffer direct arm/gripper control and fresh observations")
        settle = config.get("settle_duration_s", 0.0)
        if type(settle) not in (int, float) or not math.isfinite(settle) or not 0 <= settle <= 0.5:
            raise ValueError("Invalid fixed endpoint wait: settle_duration_s")
    if config.get("inference_schedule") == "async_buffer":
        buffer = config.get("execution_buffer_frames")
        if (not policy_timed or config.get("inference_worker") != "process"
                or type(buffer) is not int or not 1 <= buffer <= 30 - prefix
                or config.get("settle_duration_s", 0.0) != 0.0):
            raise ValueError("async_buffer requires policy_time/process, 1 <= buffer <= 30 - replan steps, and --settle-ms 0")
    interpolation = config.get("arm_command_interpolation", "none")
    blend = config.get("handoff_blend_frames", 0)
    blend_seconds = config.get("handoff_blend_duration_s", 0.)
    blend_mode = config.get("handoff_blend_mode", "fixed")
    if blend_mode not in ("fixed", "adaptive_velocity"):
        raise ValueError("Unknown handoff blend mode")
    realtime_align = config.get("realtime_joint_align", "none")
    if realtime_align not in ("none", "nearest_measured"):
        raise ValueError("Unknown realtime_joint_align")
    align_tolerance = config.get("realtime_align_tolerance_rad", 0.02)
    if realtime_align == "nearest_measured" and (
            config.get("inference_schedule") != "async_buffer" or interpolation != "pchip"
            or blend_mode != "fixed" or blend
            or type(blend_seconds) not in (int, float) or not 0 <= blend_seconds <= 0.15
            or type(align_tolerance) not in (int, float) or not 0 <= align_tolerance <= 0.05):
        raise ValueError("nearest_measured alignment requires async_buffer/PCHIP, a fixed blend of at most 0.15 s, and tolerance in [0, 0.05] rad")
    if blend_mode == "adaptive_velocity" and (
            config.get("inference_schedule") != "async_buffer" or interpolation != "pchip"
            or "handoff_blend_duration_s" in config or "handoff_blend_frames" in config):
        raise ValueError("Adaptive velocity blending requires async_buffer/PCHIP and no fixed blend duration")
    if (interpolation not in ("none", "pchip") or type(blend) is not int or not 0 <= blend <= 10
            or type(blend_seconds) not in (int, float) or not math.isfinite(blend_seconds)
            or not 0 <= blend_seconds <= 1.
            or ("handoff_blend_frames" in config and "handoff_blend_duration_s" in config)
            or ((interpolation != "none" or blend or blend_seconds) and config.get("inference_schedule") != "async_buffer")):
        raise ValueError("Arm interpolation and handoff blending require async_buffer; blend frames must be 0-10")
    if progress == "feedback":
        if (config.get("inference_schedule") not in ("synchronous", "async_tail") or filter_mode != "bounded"
                or config.get("require_new_observation") is not True):
            raise ValueError("Feedback progress requires synchronous/async_tail bounded control and fresh observations")
        bounds = {"phase_arm_tolerance_rad": (0, 0.06), "phase_command_tolerance_rad": (0, 0.03),
                  "settle_arm_tolerance_rad": (0, 0.06),
                  "settle_command_tolerance_rad": (0, 0.02), "settle_velocity_rad_s": (0, 0.02),
                  "settle_duration_s": (0, 0.5), "segment_timeout_s": (0, 8.0),
                  "execution_no_progress_timeout_s": (0, 2.0),
                  "execution_progress_epsilon_rad": (0.001, 0.01)}
        if native_errors:
            bounds["phase_arm_tolerance_rad"] = (0, NATIVE_ARM_ERROR_RAD)
            bounds["settle_arm_tolerance_rad"] = (0, NATIVE_ARM_ERROR_RAD)
        for key, (lower, upper) in bounds.items():
            value = config.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not lower < value <= upper:
                raise ValueError(f"Invalid feedback progress setting: {key}")
        if config["segment_timeout_s"] <= prefix / config["policy_fps"] + config["settle_duration_s"]:
            raise ValueError("Segment timeout must allow nominal execution and settling")
        tolerances = (("phase_command_tolerance_rad", "phase_arm_tolerance_rad"),
                      ("settle_command_tolerance_rad", "settle_arm_tolerance_rad"))
        if not all(config[command] <= config[feedback] and
                   (config[feedback] <= config["arm_stall_error_rad"] if native_errors else
                    config[feedback] < config["arm_stall_error_rad"])
                   for command, feedback in tolerances):
            raise ValueError("Execution tolerances must stay " +
                             ("at or below" if native_errors else "below") + " the arm stall bound")
    if config.get("inference_schedule") == "async_tail":
        skip = config.get("async_max_skip_frames")
        gap = config.get("async_handoff_tolerance_rad")
        if (progress != "feedback" or config.get("inference_worker") != "process"
                or type(skip) is not int or not 0 <= skip < prefix < 30
                or skip > 10 or type(gap) not in (int, float)
                or not math.isfinite(gap) or not 0 < gap <= 0.1):
            raise ValueError("Invalid async_tail settings: feedback/process, skip <= 10 < horizon, handoff <= 0.1 rad required")
    gripper_filter = config.get("gripper_command_filter", "direct")
    if gripper_filter not in ("direct", "hysteresis"):
        raise ValueError("Unknown gripper command filter")
    if gripper_filter == "hysteresis":
        if progress != "feedback":
            raise ValueError("Gripper hysteresis requires coordinated feedback progress")
        for key, lower, upper in (("gripper_confirm_s", 0, 0.3), ("gripper_min_hold_s", 0, 1.0),
                                  ("gripper_open_fraction", 0, 0.4), ("gripper_close_fraction", 0.6, 1.0)):
            value = config.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not lower < value <= upper:
                raise ValueError(f"Invalid gripper hysteresis setting: {key}")
    lag = config.get("arm_lag_compensation")
    if lag is not None:
        if not isinstance(lag, dict):
            raise ValueError("arm_lag_compensation must be a dict")
        if lag.get("enabled"):
            if config.get("inference_schedule") != "async_buffer" or interpolation != "pchip":
                raise ValueError("arm_lag_compensation requires async_buffer + pchip")
            gain = lag.get("gain", 1.0)
            if type(gain) not in (int, float) or not math.isfinite(gain) or not 0 < float(gain) <= 1.5:
                raise ValueError("arm_lag_compensation.gain must be in (0, 1.5]")
            for key, lo, hi in (("tau_ms", 0.0, 200.0), ("delay_ms", 0.0, 120.0),
                                ("max_lead_rad", 0.0, 0.25), ("gripper_lookahead_ms", 0.0, 200.0),
                                ("lead_smooth_tau_ms", 0.0, 200.0)):
                value = lag.get(key, 0.0 if key != "tau_ms" else 57.0)
                if isinstance(value, (list, tuple)):
                    if len(value) != 14:
                        raise ValueError(f"arm_lag_compensation.{key} must be scalar or length 14")
                    values = value
                else:
                    values = [value]
                for item in values:
                    if type(item) not in (int, float) or not math.isfinite(item) or not lo <= float(item) <= hi:
                        raise ValueError(f"Invalid arm_lag_compensation.{key}")


def map_target(action24, baseline24):
    """Hold torso; omit head commands; map arms/fingers by verified native names."""
    action = finite_vector(action24, (24,)).copy()
    baseline = finite_vector(baseline24, (24,))
    action[:8] = baseline[:8]
    return action[COMMAND_INDICES]


def limit_violations(target, limits):
    target = finite_vector(target, (21,))
    errors = []
    for name, value in zip(COMMAND_NAMES, target, strict=True):
        interval = limits.get(name, {})
        lower, upper = interval.get("min"), interval.get("max")
        if (type(lower) not in (int, float) or type(upper) not in (int, float) or
                not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper):
            errors.append({"name": name, "error": "missing/invalid SDK interval"})
        elif not lower <= value <= upper:
            errors.append({"name": name, "target_rad": float(value), "min": lower, "max": upper})
    return errors


class TargetGuard:
    def __init__(self, baseline, limits, config):
        self.baseline = finite_vector(baseline, (24,)).copy()
        self.last = map_target(baseline, baseline)
        self.limits = limits
        self.config = config
        self.execution_fps = execution_fps(config)
        self.frame_times = None
        self.previous_tick = None
        self.accepted_at = None
        self.actions = None
        self.execution_started = None
        self.start_frame = 0
        self.switch_at = None
        self.inference_motion_rad = 0.0
        self.initial_arm_difference = None
        self.clipped_gripper_values = 0
        self.clipped_arm_values = 0
        self.last_arm_clipping = []
        self.last_tracking_diagnostic = None
        self.bounded = config.get("arm_command_filter") == "bounded"
        self.arm_velocity = np.zeros(14)
        self.arm_stall_since = None
        self.last_requested = self.last.copy()
        self.feedback_limited_cycles = 0
        self.previous_command_difference = None
        self.feedback_paced = config.get("execution_progress") == "feedback"
        self.policy_timed = config.get("execution_progress") == "policy_time"
        self.phase_elapsed = 0.0
        self.segment_complete = False
        self.settled_since = None
        self.completed_segments = 0
        self.phase_wait_cycles = 0
        self.last_action_frame = -1
        self.gripper_candidate = [None, None]
        self.gripper_evidence_s = np.zeros(2)
        self.gripper_changed_at = np.full(2, -math.inf)
        self.gripper_filtered_cycles = 0
        self.gripper_switches = np.zeros(2, dtype=int)
        self.last_execution_progress = None
        self.progress_phase = 0.0
        self.progress_goal = None
        self.progress_best_error = None
        self.last_execution_timeout = None

    def accept(self, actions, observation_started, measured, now, observation_state=None,
               *, action_origin=None, overlapping=False, frame_times=None):
        if self.feedback_paced and self.actions is not None and not self.replacement_ready(now):
            raise ValueError("Cannot replace an unfinished feedback-paced segment")
        if frame_times is not None and self.feedback_paced:
            raise ValueError("frame_times require policy_time or wall_time execution")
        self.frame_times = None if frame_times is None else validate_frame_times(frame_times)
        actions = finite_vector(actions, (30, 24)).copy()
        measured = finite_vector(measured, (24,))
        age = now - observation_started
        if not 0 <= age < self.config["max_prediction_age_s"]:
            raise ValueError("Prediction stale or timestamp in future")
        motion = 0.0
        if observation_state is not None:
            observed = finite_vector(observation_state, (24,))
            motion = float(np.max(np.abs(observed[ACTIVE_INDICES] - measured[ACTIVE_INDICES])))
        synchronous = self.config.get("inference_schedule") == "synchronous"
        origin, start_frame = self.prediction_start(now, action_origin)
        # Validate only executable frames; async_buffer deliberately discards 26-30.
        end = (self.execution_end() if self.config.get("inference_schedule") == "async_buffer"
               else self.config["execution_prefix_frames"] if synchronous else 30)
        remaining = slice(start_frame, end)
        clipped_count = 0
        clip_to_limits = self.config.get("prediction_limit_mode", "reject") == "clip"
        # Clip only model targets, never expand SDK limits or bypass send checks.
        arm_clipping = []
        for i in range(8, 22):
            name = NAMES[i]
            interval = self.limits.get(name, {})
            lower, upper = interval.get("min"), interval.get("max")
            if (type(lower) not in (int, float) or type(upper) not in (int, float) or
                    not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper):
                raise ValueError("Prediction outside SDK limits: missing/invalid SDK interval for " + name)
            raw = actions[remaining, i].copy()
            clipped = np.clip(raw, lower, upper)
            excess = float(np.max(np.abs(raw - clipped)))
            if not clip_to_limits and excess > self.config.get("arm_clip_tolerance_rad", 0.0):
                raise ValueError("Prediction outside SDK limits: arm overshoot exceeds tolerance: " +
                                 json.dumps({"name": name, "max_excess_rad": excess}))
            count = int(np.count_nonzero(raw != clipped))
            if count:
                arm_clipping.append({"name": name, "values": count, "max_excess_rad": excess,
                                     "prediction_limit_mode": self.config.get("prediction_limit_mode", "reject"),
                                     "raw_min_rad": float(raw.min()), "raw_max_rad": float(raw.max()),
                                     "min": lower, "max": upper})
            actions[remaining, i] = clipped
        # Clip mode also saturates grippers; reject mode retains its tolerance.
        for i, name in enumerate(FINGERS, start=22):
            interval = self.limits[name]
            lower, upper = interval["min"], interval["max"]
            if (type(lower) not in (int, float) or type(upper) not in (int, float) or
                    not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper):
                raise ValueError("Prediction outside SDK limits: missing/invalid SDK interval for " + name)
            raw = actions[remaining, i].copy()
            clipped = np.clip(raw, lower, upper)
            if not clip_to_limits and np.max(np.abs(raw - clipped)) > self.config["gripper_clip_tolerance_rad"]:
                raise ValueError("Gripper overshoot exceeds the bounded saturation tolerance")
            clipped_count += int(np.count_nonzero(raw != clipped))
            actions[remaining, i] = clipped
        for action in actions[remaining]:
            errors = limit_violations(map_target(action, self.baseline), self.limits)
            if errors:
                raise ValueError("Prediction outside SDK limits: " + json.dumps(errors))
        # A gripper may legitimately request its full stroke.
        arm_indices = np.arange(8, 22)
        differences = np.abs(actions[start_frame, arm_indices] - measured[arm_indices])
        worst = arm_indices[int(np.argmax(differences))]
        diagnostic = {"name": NAMES[worst], "measured_rad": float(measured[worst]),
                      "target_rad": float(actions[start_frame, worst]),
                      "difference_rad": float(np.max(differences))}
        command_difference = actions[start_frame, 8:22] - self.last[5:19]
        index = int(np.argmax(np.abs(command_difference)))
        self.previous_command_difference = {
            "name": NAMES[index + 8], "previous_command_rad": float(self.last[index + 5]),
            "raw_target_rad": float(actions[start_frame, index + 8]),
            "difference_rad": float(command_difference[index])}
        self.actions = actions.copy()
        self.accepted_at = observation_started
        self.execution_started = origin
        self.start_frame = start_frame
        self.switch_at = min(now + self._frame_time(self.config["execution_prefix_frames"]),
                             origin + self._frame_time(30))
        if self.policy_timed:
            self.switch_at += self.config.get("settle_duration_s", 0.0)
        self.inference_motion_rad = motion
        self.initial_arm_difference = diagnostic
        self.clipped_gripper_values += clipped_count
        self.clipped_arm_values += sum(row["values"] for row in arm_clipping)
        self.last_arm_clipping = arm_clipping
        self.phase_elapsed = start_frame / self.execution_fps if self.feedback_paced else 0.0
        self.segment_complete = False
        self.settled_since = None
        self.last_execution_progress = now
        self.progress_phase = self.phase_elapsed
        self.progress_goal = actions[start_frame, 8:22].copy()
        self.progress_best_error = float(np.max(np.abs(self.progress_goal - measured[8:22])))

    def replacement_ready(self, now):
        return self.segment_complete

    def prediction_start(self, now, action_origin):
        synchronous = self.config.get("inference_schedule") == "synchronous"
        origin = now if synchronous or action_origin is None else float(action_origin)
        if not math.isfinite(origin) or not 0 <= now - origin < self._frame_time(30):
            raise ValueError("No unexpired action frames or action origin in future")
        return origin, self._frame_at(now - origin)

    def _frame_time(self, frame):
        return frame_time(frame, self.execution_fps, self.frame_times)

    def _frame_at(self, elapsed):
        return frame_at(elapsed, self.execution_fps, self.frame_times)

    def execution_end(self):
        return self.config["execution_prefix_frames"] if self.config.get("inference_schedule") == "synchronous" else 30

    def check_execution_progress(self, goal, measured, now):
        """Allow slow, measured progress while bounding stalled and total time.

        Only a new full policy frame or cumulative measured approach to a fixed
        reference counts. Command motion alone and small feedback noise do not.
        """
        error = float(np.max(np.abs(self.progress_goal - measured[8:22])))
        if self.phase_elapsed - self.progress_phase >= 1 / self.execution_fps - 1e-12:
            self.progress_phase = self.phase_elapsed
            self.progress_goal = goal[5:19].copy()
            self.progress_best_error = float(np.max(np.abs(self.progress_goal - measured[8:22])))
            self.last_execution_progress = now
        elif self.progress_best_error - error >= self.config["execution_progress_epsilon_rad"]:
            self.progress_best_error = error
            self.last_execution_progress = now
        elapsed = now - self.execution_started
        stalled = now - self.last_execution_progress
        reason = ("segment_deadline" if elapsed >= self.config["segment_timeout_s"] else
                  "no_execution_progress" if stalled >= self.config["execution_no_progress_timeout_s"] else None)
        if reason is not None:
            differences = np.abs(goal[5:19] - measured[8:22])
            worst = int(np.argmax(differences))
            self.last_execution_timeout = {
                "reason": reason, "action_frame_zero_based": self.last_action_frame,
                "segment_elapsed_s": elapsed, "no_progress_s": stalled,
                "policy_elapsed_s": self.phase_elapsed,
                "worst_joint": NAMES[worst + 8], "goal_rad": float(goal[worst + 5]),
                "command_rad": float(self.last[worst + 5]), "measured_rad": float(measured[worst + 8]),
                "max_goal_feedback_error_rad": float(differences[worst]),
                "max_command_feedback_error_rad": float(np.max(np.abs(self.last[5:19] - measured[8:22]))),
                "max_command_velocity_rad_s": float(np.max(np.abs(self.arm_velocity)))}
            raise TimeoutError("Feedback-paced execution stopped: " + json.dumps(self.last_execution_timeout))

    def needs_prediction(self, now):
        if self.feedback_paced:
            return self.actions is None or self.segment_complete
        due = self.switch_at is None or now >= self.switch_at
        if self.policy_timed and self.actions is not None and due and not self.segment_complete:
            self.segment_complete = True
            self.completed_segments += 1
        return due

    def prefetch_due(self, now, lead_s):
        if self.config.get("inference_schedule") == "synchronous":
            return self.needs_prediction(now)
        return self.switch_at is None or now >= self.switch_at - lead_s

    def has_actions(self, now):
        if self.feedback_paced:
            return self.actions is not None and not self.segment_complete
        horizon = self.config["execution_prefix_frames"] if self.config.get("inference_schedule") == "synchronous" else 30
        return self.actions is not None and 0 <= now - self.execution_started < self._frame_time(horizon)

    def action_frame(self, now):
        if not self.has_actions(now):
            return -1
        end = self.execution_end()
        if self.feedback_paced:
            return min(int(self.phase_elapsed * self.execution_fps), end - 1)
        return min(self._frame_at(now - self.execution_started), end - 1)

    def filter_grippers(self, goal, progress_dt, now):
        """Confirm intent in executed policy time, never in wall-clock hold time.

        Retain the last accepted analog setpoint; do not invent a full-close
        command or treat contact with an object as a tracking failure.
        """
        result = self.last[-2:].copy()
        for j, name in enumerate(FINGERS):
            lower, upper = self.limits[name]["min"], self.limits[name]["max"]
            fraction = (goal[j] - lower) / (upper - lower)
            intent = ("open" if fraction <= self.config["gripper_open_fraction"] else
                      "close" if fraction >= self.config["gripper_close_fraction"] else None)
            if intent != self.gripper_candidate[j] or intent is None:
                self.gripper_candidate[j] = intent
                self.gripper_evidence_s[j] = 0.0
            if intent is not None:
                self.gripper_evidence_s[j] += progress_dt
                confirmed = self.gripper_evidence_s[j] + 1e-12 >= self.config["gripper_confirm_s"]
                old_fraction = (result[j] - lower) / (upper - lower)
                same_side = ((intent == "open" and old_fraction <= self.config["gripper_open_fraction"]) or
                             (intent == "close" and old_fraction >= self.config["gripper_close_fraction"]))
                if progress_dt > 0 and confirmed:
                    if same_side:
                        # Allow the rest of a gradual closure/opening. Never latch
                        # a partly closed target just because it crossed a threshold.
                        result[j] = max(result[j], goal[j]) if intent == "close" else min(result[j], goal[j])
                    elif now - self.gripper_changed_at[j] >= self.config["gripper_min_hold_s"]:
                        result[j] = goal[j]
                        self.gripper_changed_at[j] = now
                        self.gripper_switches[j] += 1
            if abs(result[j] - goal[j]) > 0.02:
                self.gripper_filtered_cycles += 1
        return result

    def hold(self, measured, now):
        """Continue an already bounded setpoint while inference retries."""
        measured = finite_vector(measured, (24,))
        if self.previous_tick is not None and not 0 < now - self.previous_tick <= 0.05:
            raise ValueError(f"Servo cadence missed while holding: gap_ms={(now - self.previous_tick) * 1000:.3f}, limit_ms=50")
        if np.max(np.abs(measured[:8] - self.baseline[:8])) > self.config["held_joint_tolerance_rad"]:
            raise ValueError("Held torso/head moved unexpectedly")
        self.last_tracking_diagnostic = self.check_tracking(measured)
        if self.bounded:
            self.check_arm_following(measured, now)
            # An empty buffer holds the last published position. Never extrapolate.
            self.arm_velocity[:] = 0
        self.last_requested = self.last.copy()
        self.last_action_frame = -1
        if limit_violations(self.last, self.limits):
            raise ValueError("Hold setpoint outside SDK limits")
        self.previous_tick = now
        return self.last.copy()

    def check_tracking(self, measured):
        feedback = map_target(measured, measured)
        differences = np.abs(feedback - self.last)
        worst = int(np.argmax(differences))
        if differences[worst] > self.config["max_tracking_error_rad"]:
            return {"name": COMMAND_NAMES[worst], "measured_rad": float(feedback[worst]),
                    "target_rad": float(self.last[worst]), "error_rad": float(differences[worst])}
        return None

    def check_arm_following(self, measured, now):
        """Stop publication on a diverged or persistently untracked arm command."""
        differences = np.abs(self.last[5:19] - measured[8:22])
        index = int(np.argmax(differences))
        error = float(differences[index])
        if error > self.config["max_arm_tracking_error_rad"]:
            raise RuntimeError(f"Arm feedback diverged: {NAMES[index + 8]} error={error:.6f} rad")
        if error > self.config["arm_stall_error_rad"]:
            if self.arm_stall_since is None:
                self.arm_stall_since = now
            if now - self.arm_stall_since >= self.config["arm_stall_timeout_s"]:
                raise RuntimeError(f"Arm command not tracked: {NAMES[index + 8]} error={error:.6f} rad")
        else:
            self.arm_stall_since = None

    def filter_arm_target(self, goal, measured, now):
        self.check_arm_following(measured, now)
        # Late ticks never accumulate a larger position step.
        dt = min(1 / self.config["servo_hz"],
                 now - self.previous_tick if self.previous_tick is not None else 1 / self.config["servo_hz"])
        error = goal[5:19] - self.last[5:19]
        if self.config.get("arm_smoothing", "critical_damping") == "none":
            # Follow the interpolated position directly when bounds allow it.
            # Retain acceleration-limited braking near the target, without a
            # damping/time-constant filter or a velocity reset at chunk joins.
            max_acceleration = self.config["arm_acceleration_rad_s2"]
            half_step = max_acceleration * dt / 2
            braking_speed = np.sqrt(2 * max_acceleration * np.abs(error) + half_step ** 2) - half_step
            desired_velocity = np.sign(error) * np.minimum(
                np.abs(error) / dt, np.minimum(braking_speed, self.config["arm_rate_rad_s"]))
            velocity = self.arm_velocity + np.clip(
                desired_velocity - self.arm_velocity, -max_acceleration * dt, max_acceleration * dt)
        else:
            omega = 1 / self.config["arm_filter_time_constant_s"]
            acceleration = np.clip(omega * omega * error - 2 * omega * self.arm_velocity,
                                   -self.config["arm_acceleration_rad_s2"], self.config["arm_acceleration_rad_s2"])
            velocity = np.clip(self.arm_velocity + acceleration * dt,
                               -self.config["arm_rate_rad_s"], self.config["arm_rate_rad_s"])
        candidate = self.last[5:19] + velocity * dt
        lo = np.asarray([self.limits[n]["min"] for n in NAMES[8:22]])
        hi = np.asarray([self.limits[n]["max"] for n in NAMES[8:22]])
        blocked = ((np.abs(candidate - measured[8:22]) > self.config["max_arm_command_lead_rad"]) |
                   (candidate < lo) | (candidate > hi))
        if np.any(blocked):
            # Safety braking takes priority over the normal acceleration bound.
            candidate[blocked] = self.last[5:19][blocked]
            velocity[blocked] = 0
            self.feedback_limited_cycles += 1
        self.arm_velocity = velocity
        target = goal.copy()
        target[5:19] = candidate
        return target

    def step(self, measured, now):
        measured = finite_vector(measured, (24,))
        if self.actions is None or self.accepted_at is None:
            raise ValueError("No accepted prediction; refuse initial servo publication")
        elapsed = now - self.execution_started
        if elapsed < 0 or (not self.feedback_paced and not self.policy_timed
                           and elapsed > self.config["max_hold_s"]):
            raise ValueError("Prediction expired; bounded hold deadline reached")
        if self.previous_tick is not None and not 0 < now - self.previous_tick <= 0.05:
            raise ValueError(f"Servo cadence missed; do not catch up delayed commands: gap_ms={(now - self.previous_tick) * 1000:.3f}, limit_ms=50")
        if np.max(np.abs(measured[:8] - self.baseline[:8])) > self.config["held_joint_tolerance_rad"]:
            raise ValueError("Held torso/head moved unexpectedly")
        self.last_tracking_diagnostic = self.check_tracking(measured)
        policy_elapsed = self.phase_elapsed if self.feedback_paced else elapsed
        end = self.execution_end()
        if self.feedback_paced:
            index = min(int(policy_elapsed * self.execution_fps), end - 1)
        else:
            index = self._frame_at(policy_elapsed)
        self.last_action_frame = index
        goal = self.last if index >= 30 else map_target(self.actions[index], self.baseline)
        if self.bounded and index < 30:
            following = min(index + 1, end - 1)
            if self.frame_times is None:
                fraction = min(1.0, policy_elapsed * self.execution_fps - index)
            else:
                start, stop = self.frame_times[index], self.frame_times[index + 1]
                fraction = min(1.0, (policy_elapsed - start) / (stop - start))
            goal = goal.copy()
            goal[5:19] += fraction * (self.actions[following, 8:22] - self.actions[index, 8:22])
        self.last_requested = goal.copy()
        if self.feedback_paced:
            self.check_execution_progress(goal, measured, now)
        target = self.filter_arm_target(goal, measured, now) if self.bounded else goal.copy()
        if self.feedback_paced:
            dt = min(1 / self.config["servo_hz"],
                     now - self.previous_tick if self.previous_tick is not None else 1 / self.config["servo_hz"])
            # Both command generation and actual feedback must reach this phase.
            # A lagging filtered arm can no longer leave the fingers running ahead.
            # Command generation and physical feedback have different tolerances.
            # Normal impedance control can retain a small steady position error;
            # do not tighten or offset the command to force that error to zero.
            feedback_error = float(np.max(np.abs(goal[5:19] - measured[8:22])))
            command_error = float(np.max(np.abs(goal[5:19] - target[5:19])))
            phase_ready = (feedback_error <= self.config["phase_arm_tolerance_rad"] and
                           command_error <= self.config["phase_command_tolerance_rad"])
            remaining = max(0.0, end / self.execution_fps - self.phase_elapsed)
            progress_dt = min(dt, remaining) if phase_ready else 0.0
            if progress_dt == 0 and remaining > 1e-12:
                self.phase_wait_cycles += 1
            if self.config.get("gripper_command_filter") == "hysteresis":
                target[-2:] = self.filter_grippers(goal[-2:], progress_dt, now)
            elif progress_dt == 0:
                target[-2:] = self.last[-2:]
            self.phase_elapsed += progress_dt
            reached_end = end / self.execution_fps - self.phase_elapsed <= 1e-12
            settled = (reached_end and
                       np.max(np.abs(goal[5:19] - measured[8:22])) <= self.config["settle_arm_tolerance_rad"] and
                       np.max(np.abs(goal[5:19] - target[5:19])) <= self.config["settle_command_tolerance_rad"] and
                       np.max(np.abs(self.arm_velocity)) <= self.config["settle_velocity_rad_s"])
            self.settled_since = (self.settled_since if self.settled_since is not None else now) if settled else None
            if self.settled_since is not None and now - self.settled_since >= self.config["settle_duration_s"]:
                self.segment_complete = True
                self.completed_segments += 1
        if limit_violations(target, self.limits):
            raise ValueError("Command outside SDK limits")
        self.last, self.previous_tick = target, now
        return target.copy()


class GdkBackend:
    """Single caller; no setters for modes/enables/calibration or low-latency mode."""
    def __init__(self, config, *, read_only=False):
        self.read_only = read_only
        if not read_only:
            validate_contract(config)
        if not read_only and not sys.stdin.isatty():
            raise ValueError("Physical execution requires the operator's interactive terminal")
        addresses = subprocess.run(["ip", "-j", "-4", "addr", "show", "dev", config["sdk_interface"]],
                                   check=True, capture_output=True, text=True)
        if not any(row.get("local") == config["sdk_source_ip"]
                   for interface in json.loads(addresses.stdout) for row in interface.get("addr_info", [])):
            raise ValueError("Configured SDK source IP is not assigned to the existing interface")
        if os.environ.get("LOCATOR_IP") != config["sdk_source_ip"] or os.environ.get("AORTA_DISCOVERY_URI") != config["discovery_uri"]:
            raise ValueError("SDK process environment differs from verified h6 interface")
        self.sdk = importlib.import_module("agibot_gdk")
        self.initialized = False
        self.robot = None
        self.last_stamp = 0
        self.control_mode = None
        self.advanced_at = time.monotonic()
        if self.sdk.gdk_init() != self.sdk.GDKRes.kSuccess:
            raise RuntimeError("SDK initialization failed")
        self.initialized = True
        try:
            if self.sdk.get_robot_aid() != config["robot_aid"]:
                raise ValueError("Robot identity mismatch")
            self.robot = self.sdk.Robot()
            self.limits = self.robot.get_joint_limits()
        except BaseException:
            self.close()
            raise

    def read(self, *, require_control_status=False):
        started = time.monotonic()
        health = self.robot.get_whole_body_status()
        if any(health.get(name) != 0 for name in ERROR_FIELDS):
            raise ValueError("Robot fault or missing health field")
        status = self.robot.get_motion_control_status()
        if not self.read_only or require_control_status:
            if status.error_code != 0 or status.control_mode not in (1, 3) or status.mode not in (0, 1):
                raise ValueError("Robot not in healthy supported normal control mode: " + json.dumps({
                    "mode": status.mode, "control_mode": status.control_mode,
                    "error_code": status.error_code, "error_msg": getattr(status, "error_msg", "")}))
            if self.control_mode is None:
                self.control_mode = status.control_mode
            elif status.control_mode != self.control_mode:
                raise ValueError("Robot control mode changed during execution")
        joints = self.robot.get_joint_states()
        rows = joints.get("states", [])
        if len(rows) != 22 or {row.get("name") for row in rows} != set(NAMES[:22]):
            raise ValueError("Body/head/arm joint names mismatch")
        joint_errors = [{key: row.get(key) for key in
                         ("name", "error_code", "motor_position", "motor_velocity", "effort")}
                        for row in rows if row.get("error_code") != 0]
        if joint_errors:
            raise ValueError("Joint error: " + json.dumps({
                "joint_timestamp": joints.get("timestamp"), "joints": joint_errors,
                "motion": {key: getattr(status, key, None) for key in
                           ("mode", "control_mode", "error_code", "error_msg")}}, ensure_ascii=False))
        positions = {row["name"]: row.get("motor_position") for row in rows}
        ends = self.robot.get_end_state()
        for side, model_name, native in zip(("left", "right"), NAMES[22:], FINGERS, strict=True):
            group = ends.get(side + "_end_state", {})
            values = group.get("end_states", [])
            if (group.get("type") != 1 or group.get("controlled") is not True or
                    group.get("names") != [native] or len(values) != 1 or
                    values[0].get("enable") is not True or values[0].get("err_code") != 0):
                raise ValueError("Omnipicker identity/control/health mismatch")
            positions[model_name] = values[0].get("position")
        stamp = joints.get("timestamp")
        if type(stamp) is not int or stamp <= 0 or stamp < self.last_stamp:
            raise ValueError("SDK source timestamp invalid or backwards")
        now = time.monotonic()
        if not self.read_only and now - started > 0.05:
            raise ValueError("SDK feedback acquisition exceeded 50 ms")
        if stamp > self.last_stamp:
            self.last_stamp, self.advanced_at = stamp, now
        elif now - self.advanced_at > 0.15:
            raise ValueError("SDK joint source stopped advancing for 150 ms")
        return finite_vector([positions[name] for name in NAMES], (24,))

    def send(self, target):
        if self.read_only:
            raise ValueError("Read-only backend cannot send robot commands")
        errors = limit_violations(target, self.limits)
        if errors:
            raise ValueError("Invalid SDK target: " + json.dumps(errors))
        request = self.sdk.JointServoControlReq()
        request.joint_names = list(COMMAND_NAMES)
        request.joint_positions = target.tolist()
        request.joint_velocities = []
        request.control_period = 0.02
        start = time.monotonic()
        result = self.robot.joint_servo_control(request, False)
        if result != 0 or time.monotonic() - start > 0.05:
            # A failed/timed-out call may already have published; never retry it.
            raise RuntimeError("SDK servo failed or blocked; publication state unknown")

    def close(self):
        self.robot = None
        if self.initialized:
            self.sdk.gdk_release()
            self.initialized = False
