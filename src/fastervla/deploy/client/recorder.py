"""TraceRecorder: per-tick / per-chunk capture aligned with h7 robot-live traces.

A/B double-buffer: ``flush()`` seals the active bank and writes it while recording
continues on the other bank. Drops only when both banks are full (or closed).
Disk I/O never runs inside ``record_tick``. See ``docs/trace-design.md``.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fastervla.core.clock import LatencyWindow

STAGE_STEP = 0
STAGE_INITIAL_HOLD = 1
STAGE_BUFFER_EMPTY_HOLD = 2
STAGE_NAMES = np.array(["step", "initial_hold", "buffer_empty_hold"], dtype=object)

TRACKER_PASSTHROUGH = 0
TRACKER_LEAD = 1
TRACKER_MPC = 2
TRACKER_FALLBACK = 3

_PRED_SCALAR_KEYS = (
    "requested_monotonic_s",
    "ready_monotonic_s",
    "observation_monotonic_s",
    "accepted_monotonic_s",
    "action_origin_monotonic_s",
    "handoff_at_monotonic_s",
    "d_hat",
    "d_actual",
    "anchor_step",
    "plan_id",
    "rtc_applied",
    "rtc_s",
    "rtc_d",
    "rtc_guided_gap",
    "overlap_s",
    "overlap_d",
    "overlap_gap",
    "observation_age_ms",
    "image_age_ms",
    "packet_id",
    "start_frame",
    "previous_command_difference",
    "request_to_ready_ms",
    "client_prep_ms",
    "infer_call_ms",
    "merge_accepted",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def make_run_dir(results_dir: str | Path, *, prefix: str = "robot-live") -> Path:
    """Create ``results_dir/robot-live-<stamp>-<hex>/`` for one run."""
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    path = root / f"{prefix}-{run_id}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class PeriodicFlusher:
    """Call ``recorder.flush()`` every ``every_s`` seconds on a daemon thread."""

    def __init__(self, recorder: TraceRecorder, *, every_s: float) -> None:
        if every_s <= 0:
            raise ValueError("every_s must be > 0")
        self._recorder = recorder
        self._every_s = float(every_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.flush_count = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.wait(self._every_s):
                try:
                    self._recorder.flush()
                    self.flush_count += 1
                except Exception:
                    # Keep trying; close() will final-write.
                    continue

        self._thread = threading.Thread(
            target=_loop, name="fastervla-trace-periodic-flush", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        th = self._thread
        if th is not None:
            th.join(timeout=max(1.0, self._every_s + 1.0))
        self._thread = None


@dataclass
class _TickBank:
    capacity: int
    action_dim: int
    command_dim: int
    n: int = 0
    flushing: bool = False
    monotonic_s: np.ndarray = field(init=False)
    measured: np.ndarray = field(init=False)
    reference: np.ndarray = field(init=False)
    command: np.ndarray = field(init=False)
    progress: np.ndarray = field(init=False)
    plan_id: np.ndarray = field(init=False)
    stage: np.ndarray = field(init=False)
    tracker_status: np.ndarray = field(init=False)
    contact: np.ndarray = field(init=False)
    servo_tick_ms: np.ndarray = field(init=False)
    action_frame: np.ndarray = field(init=False)
    accepted_prediction: np.ndarray = field(init=False)
    chunks: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        cap = int(self.capacity)
        ad, cd = int(self.action_dim), int(self.command_dim)
        self.monotonic_s = np.empty(cap, dtype=np.float64)
        self.measured = np.empty((cap, ad), dtype=np.float64)
        self.reference = np.empty((cap, cd), dtype=np.float64)
        self.command = np.empty((cap, cd), dtype=np.float64)
        self.progress = np.empty(cap, dtype=np.float64)
        self.plan_id = np.empty(cap, dtype=np.int32)
        self.stage = np.empty(cap, dtype=np.int8)
        self.tracker_status = np.empty(cap, dtype=np.int8)
        self.contact = np.empty(cap, dtype=np.int8)
        self.servo_tick_ms = np.empty(cap, dtype=np.float32)
        self.action_frame = np.empty(cap, dtype=np.int64)
        self.accepted_prediction = np.empty(cap, dtype=np.int64)

    def reset(self) -> None:
        self.n = 0
        self.chunks.clear()
        self.flushing = False

    def snapshot(self) -> dict[str, Any]:
        n = int(self.n)
        return {
            "n": n,
            "monotonic_s": self.monotonic_s[:n].copy(),
            "measured": self.measured[:n].copy(),
            "reference": self.reference[:n].copy(),
            "command": self.command[:n].copy(),
            "progress": self.progress[:n].copy(),
            "plan_id": self.plan_id[:n].copy(),
            "stage": self.stage[:n].copy(),
            "tracker_status": self.tracker_status[:n].copy(),
            "contact": self.contact[:n].copy(),
            "servo_tick_ms": self.servo_tick_ms[:n].copy(),
            "action_frame": self.action_frame[:n].copy(),
            "accepted_prediction": self.accepted_prediction[:n].copy(),
            "chunks": list(self.chunks),
        }


class TraceRecorder:
    """A/B double-buffered recorder; ``flush()`` does not stop recording."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        capacity_ticks: int,
        horizon: int = 30,
        action_dim: int = 24,
        command_dim: int = 21,
        max_chunks_per_bank: int = 4096,
        max_sealed_segments: int = 64,
    ) -> None:
        if capacity_ticks < 1:
            raise ValueError("capacity_ticks must be >= 1")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if max_sealed_segments < 0:
            raise ValueError("max_sealed_segments must be >= 0")
        self.out_dir = Path(out_dir)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.command_dim = int(command_dim)
        self.max_chunks_per_bank = int(max_chunks_per_bank)
        self.max_sealed_segments = int(max_sealed_segments)
        # Back-compat alias used by older call sites / tests.
        self.max_chunks = self.max_chunks_per_bank

        cap = int(capacity_ticks)
        self._lock = threading.Lock()
        self._banks = (
            _TickBank(capacity=cap, action_dim=action_dim, command_dim=command_dim),
            _TickBank(capacity=cap, action_dim=action_dim, command_dim=command_dim),
        )
        self._active = 0
        self._drops = 0
        self._total_ticks = 0
        self._total_chunks = 0
        self._sealed: list[dict[str, Any]] = []
        self._counters: dict[str, int] = {
            "buffer_empty_cycles": 0,
            "starved_events": 0,
            "tracking_warning_cycles": 0,
            "rejected_merges": 0,
        }
        self._closed = False
        self._flush_lock = threading.Lock()  # serialize disk writes
        self._infer_records: list[dict[str, Any]] = []
        self._experiment: dict[str, Any] = {}
        self._track_err_window = LatencyWindow(maxlen=2048)
        self._last_track_warn_mono = 0.0
        self.track_warn_threshold_rad = 0.10
        self.track_warn_throttle_s = 1.0

        self.r2r_ms = LatencyWindow(maxlen=500)
        self.image_age_ms = LatencyWindow(maxlen=128)
        self.servo_tick_ms_window = LatencyWindow(maxlen=256)

    def _idle_index(self) -> int:
        return 1 - self._active

    @property
    def tick_count(self) -> int:
        """Ticks still sitting in the active bank (not yet sealed)."""
        with self._lock:
            return int(self._banks[self._active].n)

    @property
    def total_ticks(self) -> int:
        with self._lock:
            return int(self._total_ticks)

    @property
    def drops(self) -> int:
        with self._lock:
            return int(self._drops)

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return sum(len(b.chunks) for b in self._banks) + sum(
                len(s["chunks"]) for s in self._sealed
            )

    def bump_counter(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] = int(self._counters.get(name, 0)) + int(n)

    def set_experiment(self, **fields: Any) -> None:
        with self._lock:
            self._experiment.update(fields)

    def note_infer_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._infer_records.append(dict(record))

    def note_tracking_error(
        self,
        err_rad: float,
        *,
        now: float | None = None,
    ) -> bool:
        """Record |cmd−meas| sample; return True if a throttled warning should fire."""
        err = float(err_rad)
        self._track_err_window.record(err)
        if err < self.track_warn_threshold_rad:
            return False
        self.bump_counter("tracking_warning_cycles", 1)
        t = float(time.monotonic() if now is None else now)
        with self._lock:
            if t - self._last_track_warn_mono < self.track_warn_throttle_s:
                return False
            self._last_track_warn_mono = t
            return True

    def record_tick(
        self,
        *,
        monotonic_s: float,
        measured24: np.ndarray,
        reference21: np.ndarray,
        command21: np.ndarray,
        progress: float,
        plan_id: int,
        stage: int = STAGE_STEP,
        tracker_status: int = 0,
        contact: int = 0,
        servo_tick_ms: float = float("nan"),
        action_frame: int = -1,
        accepted_prediction: int = -1,
    ) -> None:
        """Copy one control sample into the active bank. Never touches disk."""
        measured = np.asarray(measured24, dtype=np.float64).reshape(-1)
        reference = np.asarray(reference21, dtype=np.float64).reshape(-1)
        command = np.asarray(command21, dtype=np.float64).reshape(-1)
        if measured.shape[0] != self.action_dim:
            raise ValueError(f"measured24 must be length {self.action_dim}")
        if reference.shape[0] != self.command_dim or command.shape[0] != self.command_dim:
            raise ValueError(f"reference/command must be length {self.command_dim}")

        auto_seal: dict[str, Any] | None = None
        with self._lock:
            if self._closed:
                return
            bank = self._banks[self._active]
            if bank.n >= bank.capacity:
                # Rotate onto idle bank when free and sealed queue not saturated.
                idle = self._banks[self._idle_index()]
                can_rotate = (
                    not idle.flushing
                    and idle.n == 0
                    and not idle.chunks
                    and len(self._sealed) < self.max_sealed_segments
                )
                if can_rotate:
                    auto_seal = bank.snapshot()
                    bank.reset()
                    self._active = self._idle_index()
                    bank = self._banks[self._active]
                else:
                    self._drops += 1
                    return
            i = bank.n
            bank.monotonic_s[i] = float(monotonic_s)
            bank.measured[i] = measured
            bank.reference[i] = reference
            bank.command[i] = command
            bank.progress[i] = float(progress)
            bank.plan_id[i] = int(plan_id)
            bank.stage[i] = int(stage)
            bank.tracker_status[i] = int(tracker_status)
            bank.contact[i] = int(contact)
            bank.servo_tick_ms[i] = float(servo_tick_ms)
            bank.action_frame[i] = int(action_frame)
            bank.accepted_prediction[i] = int(accepted_prediction)
            bank.n = i + 1
            self._total_ticks += 1

        if auto_seal is not None:
            with self._lock:
                self._sealed.append(auto_seal)
            # Best-effort background write of sealed bank (non-final).
            threading.Thread(
                target=self._write_sealed_unlocked,
                name="fastervla-trace-writer",
                daemon=True,
            ).start()

        if np.isfinite(servo_tick_ms):
            self.servo_tick_ms_window.record(float(servo_tick_ms))

    def record_chunk(self, **fields: Any) -> None:
        """Store one accepted prediction on the active bank."""
        actions = np.asarray(fields.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(
                f"actions must be (H,{self.action_dim}), got {getattr(actions, 'shape', None)}"
            )
        if actions.shape[0] != self.horizon:
            h = self.horizon
            padded = np.zeros((h, self.action_dim), dtype=np.float32)
            n = min(h, actions.shape[0])
            padded[:n] = actions[:n]
            if n > 0 and actions.shape[0] < h:
                padded[n:] = actions[-1]
            actions = padded

        raw = fields.get("actions_raw", actions)
        actions_raw = np.asarray(raw, dtype=np.float32)
        if actions_raw.shape != actions.shape:
            padded = np.zeros_like(actions)
            n = min(actions.shape[0], actions_raw.shape[0])
            padded[:n] = actions_raw[:n]
            actions_raw = padded

        entry: dict[str, Any] = {
            "actions": actions.copy(),
            "actions_raw": actions_raw.copy(),
        }
        for key in _PRED_SCALAR_KEYS:
            if key in fields and fields[key] is not None:
                entry[key] = fields[key]
        if "step_dt" in fields and fields["step_dt"] is not None:
            entry["step_dt"] = np.asarray(fields["step_dt"], dtype=np.float64).copy()
        if "retime_status" in fields and fields["retime_status"] is not None:
            entry["retime_status"] = str(fields["retime_status"])
        if "observation_state" in fields and fields["observation_state"] is not None:
            entry["observation_state"] = np.asarray(
                fields["observation_state"], dtype=np.float64
            ).reshape(self.action_dim)
        if "frame_times_s" in fields and fields["frame_times_s"] is not None:
            entry["frame_times_s"] = np.asarray(
                fields["frame_times_s"], dtype=np.float64
            ).reshape(-1)

        req = entry.get("requested_monotonic_s")
        ready = entry.get("ready_monotonic_s")
        if req is not None and ready is not None:
            self.r2r_ms.record((float(ready) - float(req)) * 1000.0)
        age = entry.get("observation_age_ms")
        if age is not None and np.isfinite(float(age)):
            self.image_age_ms.record(float(age))

        with self._lock:
            if self._closed:
                return
            bank = self._banks[self._active]
            if len(bank.chunks) >= self.max_chunks_per_bank:
                self._drops += 1
                return
            bank.chunks.append(entry)
            self._total_chunks += 1

    def _seal_active_locked(self) -> dict[str, Any] | None:
        """Move active bank contents into sealed list; switch to idle bank."""
        bank = self._banks[self._active]
        if bank.n == 0 and not bank.chunks:
            return None
        snap = bank.snapshot()
        bank.reset()
        idle = self._banks[self._idle_index()]
        # Prefer writing into the idle bank next; if idle is flushing, stay on same
        # after reset (active already cleared).
        if not idle.flushing:
            self._active = self._idle_index()
        self._sealed.append(snap)
        return snap

    def flush(self, *, final: bool = False) -> Path:
        """Seal active bank and write all sealed segments to disk.

        Recording may continue on the other bank unless ``final=True`` (close).
        """
        with self._lock:
            if self._closed and not self._sealed:
                return self.out_dir
            self._seal_active_locked()
            sealed = list(self._sealed)
            self._sealed.clear()
            drops = int(self._drops)
            counters = dict(self._counters)
            total_ticks = int(self._total_ticks)
            infer_records = list(self._infer_records)
            experiment = dict(self._experiment)
            if final:
                self._closed = True

        self._write_segments(
            sealed,
            drops=drops,
            counters=counters,
            total_ticks=total_ticks,
            infer_records=infer_records,
            experiment=experiment,
        )
        return self.out_dir

    def close(self) -> Path:
        """Final flush; reject further record_* calls."""
        return self.flush(final=True)

    def _write_sealed_unlocked(self) -> None:
        with self._lock:
            if not self._sealed or self._closed:
                return
            sealed = list(self._sealed)
            self._sealed.clear()
            drops = int(self._drops)
            counters = dict(self._counters)
            total_ticks = int(self._total_ticks)
            infer_records = list(self._infer_records)
            experiment = dict(self._experiment)
        self._write_segments(
            sealed,
            drops=drops,
            counters=counters,
            total_ticks=total_ticks,
            infer_records=infer_records,
            experiment=experiment,
        )

    def _write_segments(
        self,
        segments: list[dict[str, Any]],
        *,
        drops: int,
        counters: dict[str, int],
        total_ticks: int,
        infer_records: list[dict[str, Any]] | None = None,
        experiment: dict[str, Any] | None = None,
    ) -> None:
        """Merge segments with any existing on-disk arrays (append semantics)."""
        with self._flush_lock:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            prev_ctl = self._load_existing_control()
            prev_pred = self._load_existing_prediction()

            mono_parts = [prev_ctl["monotonic_s"]] if prev_ctl else []
            meas_parts = [prev_ctl["measured_state24"]] if prev_ctl else []
            ref_parts = [prev_ctl["reference21"]] if prev_ctl else []
            cmd_parts = [prev_ctl["command21"]] if prev_ctl else []
            prog_parts = [prev_ctl["progress"]] if prev_ctl else []
            plan_parts = [prev_ctl["plan_id"]] if prev_ctl else []
            stage_parts = [prev_ctl["stage"]] if prev_ctl else []
            trk_parts = [prev_ctl["tracker_status"]] if prev_ctl else []
            contact_parts = [prev_ctl["contact"]] if prev_ctl else []
            servo_parts = [prev_ctl["servo_tick_ms"]] if prev_ctl else []
            frame_parts = [prev_ctl["action_frame"]] if prev_ctl else []
            ap_parts = [prev_ctl["accepted_prediction"]] if prev_ctl else []

            all_chunks: list[dict[str, Any]] = []
            if prev_pred is not None:
                all_chunks.extend(self._chunks_from_pred(prev_pred))

            for seg in segments:
                if seg["n"] > 0:
                    mono_parts.append(seg["monotonic_s"])
                    meas_parts.append(seg["measured"])
                    ref_parts.append(seg["reference"])
                    cmd_parts.append(seg["command"])
                    prog_parts.append(seg["progress"])
                    plan_parts.append(seg["plan_id"])
                    stage_parts.append(seg["stage"])
                    trk_parts.append(seg["tracker_status"])
                    contact_parts.append(seg["contact"])
                    servo_parts.append(seg["servo_tick_ms"])
                    frame_parts.append(seg["action_frame"])
                    ap_parts.append(seg["accepted_prediction"])
                all_chunks.extend(seg["chunks"])

            if mono_parts:
                mono = np.concatenate(mono_parts)
                measured = np.concatenate(meas_parts, axis=0)
                reference = np.concatenate(ref_parts, axis=0)
                command = np.concatenate(cmd_parts, axis=0)
                progress = np.concatenate(prog_parts)
                plan_id = np.concatenate(plan_parts)
                stage = np.concatenate(stage_parts)
                tracker_status = np.concatenate(trk_parts)
                contact = np.concatenate(contact_parts)
                servo_tick_ms = np.concatenate(servo_parts)
                action_frame = np.concatenate(frame_parts)
                accepted_prediction = np.concatenate(ap_parts)
            else:
                mono = np.empty(0, dtype=np.float64)
                measured = np.empty((0, self.action_dim), dtype=np.float64)
                reference = np.empty((0, self.command_dim), dtype=np.float64)
                command = np.empty((0, self.command_dim), dtype=np.float64)
                progress = np.empty(0, dtype=np.float64)
                plan_id = np.empty(0, dtype=np.int32)
                stage = np.empty(0, dtype=np.int8)
                tracker_status = np.empty(0, dtype=np.int8)
                contact = np.empty(0, dtype=np.int8)
                servo_tick_ms = np.empty(0, dtype=np.float32)
                action_frame = np.empty(0, dtype=np.int64)
                accepted_prediction = np.empty(0, dtype=np.int64)

            n = int(mono.shape[0])
            _atomic_savez(
                self.out_dir / "control_trace.npz",
                monotonic_s=mono,
                measured_state24=measured,
                reference21=reference,
                command21=command,
                progress=progress,
                plan_id=plan_id,
                stage=stage,
                stage_names=STAGE_NAMES,
                tracker_status=tracker_status,
                contact=contact,
                servo_tick_ms=servo_tick_ms,
                action_frame=action_frame,
                accepted_prediction=accepted_prediction,
                controller_feedback24=measured,
                requested_target21=command,
                arm_velocity14=np.zeros((n, 14), dtype=np.float64),
            )

            self._write_prediction(all_chunks)

            # Tracking |cmd−meas| on arm command dims if we have samples.
            track_p50 = self._track_err_window.p50()
            track_p95 = self._track_err_window.p95()
            report = {
                "mode": "fastervla",
                "control_trace_samples": n,
                "prediction_chunks": len(all_chunks),
                "trace_drops": drops,
                "total_ticks_recorded": total_ticks,
                "buffer_empty_cycles": counters.get("buffer_empty_cycles", 0),
                "starved_events": counters.get("starved_events", 0),
                "tracking_warning_cycles": counters.get("tracking_warning_cycles", 0),
                "rejected_merges": counters.get("rejected_merges", 0),
                "r2r_p50_ms": self.r2r_ms.p50(),
                "r2r_p95_ms": self.r2r_ms.p95(),
                "servo_tick_p95_ms": self.servo_tick_ms_window.p95(),
                "image_age_p95_ms": self.image_age_ms.p95(),
                "track_err_p50_rad": track_p50,
                "track_err_p95_rad": track_p95,
                "out_dir": str(self.out_dir),
                "double_buffer": True,
                "experiment": experiment or {},
                "inference_records": infer_records or [],
            }
            report_path = self.out_dir / "robot_bridge_report.json"
            temporary = report_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(report, indent=2, default=_json_default),
                encoding="utf-8",
            )
            temporary.replace(report_path)

    def _load_existing_control(self) -> dict[str, np.ndarray] | None:
        path = self.out_dir / "control_trace.npz"
        if not path.is_file():
            return None
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    def _load_existing_prediction(self) -> dict[str, np.ndarray] | None:
        path = self.out_dir / "prediction_trace.npz"
        if not path.is_file():
            return None
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    def _chunks_from_pred(self, pred: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        k = int(pred["actions"].shape[0]) if "actions" in pred else 0
        out: list[dict[str, Any]] = []
        for i in range(k):
            entry: dict[str, Any] = {
                "actions": np.asarray(pred["actions"][i], dtype=np.float32),
                "actions_raw": np.asarray(pred["actions_raw"][i], dtype=np.float32)
                if "actions_raw" in pred
                else np.asarray(pred["actions"][i], dtype=np.float32),
            }
            for key in _PRED_SCALAR_KEYS:
                if key in pred and pred[key].shape[0] > i:
                    entry[key] = pred[key][i].item() if pred[key].ndim == 1 else pred[key][i]
            if "step_dt" in pred and pred["step_dt"].shape[0] > i:
                entry["step_dt"] = np.asarray(pred["step_dt"][i], dtype=np.float64)
            if "retime_status" in pred and pred["retime_status"].shape[0] > i:
                entry["retime_status"] = str(pred["retime_status"][i])
            if "observation_state" in pred and pred["observation_state"].shape[0] > i:
                entry["observation_state"] = np.asarray(
                    pred["observation_state"][i], dtype=np.float64
                )
            if "frame_times_s" in pred and pred["frame_times_s"].shape[0] > i:
                entry["frame_times_s"] = np.asarray(
                    pred["frame_times_s"][i], dtype=np.float64
                )
            out.append(entry)
        return out

    def _write_prediction(self, chunks: list[dict[str, Any]]) -> None:
        h, d = self.horizon, self.action_dim
        k = len(chunks)

        def col(key: str, dtype: Any = np.float64, default: float = np.nan) -> np.ndarray:
            if k == 0:
                return np.empty(0, dtype=dtype)
            vals = []
            for c in chunks:
                if key not in c or c[key] is None:
                    vals.append(default)
                else:
                    vals.append(c[key])
            return np.asarray(vals, dtype=dtype)

        if k == 0:
            actions = np.empty((0, h, d), dtype=np.float32)
            actions_raw = np.empty((0, h, d), dtype=np.float32)
            obs_state = np.empty((0, d), dtype=np.float64)
            frame_times = np.empty((0, h + 1), dtype=np.float64)
            step_dt = np.empty((0, h), dtype=np.float64)
            retime_status = np.empty(0, dtype=object)
        else:
            actions = np.stack([c["actions"] for c in chunks], axis=0)
            actions_raw = np.stack([c["actions_raw"] for c in chunks], axis=0)
            obs_state = np.stack(
                [
                    c.get("observation_state")
                    if c.get("observation_state") is not None
                    else np.zeros(d, dtype=np.float64)
                    for c in chunks
                ],
                axis=0,
            )
            ft_list = []
            for c in chunks:
                ft = c.get("frame_times_s")
                if ft is None:
                    ft_list.append(np.full(h + 1, np.nan))
                else:
                    row = np.full(h + 1, np.nan)
                    m = min(h + 1, len(ft))
                    row[:m] = ft[:m]
                    ft_list.append(row)
            frame_times = np.stack(ft_list, axis=0)
            sd_list = []
            for c in chunks:
                sd = c.get("step_dt")
                if sd is None:
                    sd_list.append(np.full(h, np.nan))
                else:
                    row = np.full(h, np.nan)
                    m = min(h, len(sd))
                    row[:m] = sd[:m]
                    sd_list.append(row)
            step_dt = np.stack(sd_list, axis=0)
            retime_status = np.asarray(
                [c.get("retime_status", "") for c in chunks], dtype=object
            )

        _atomic_savez(
            self.out_dir / "prediction_trace.npz",
            actions=actions,
            actions_raw=actions_raw,
            observation_state=obs_state,
            inference_state=obs_state,
            observation_monotonic_s=col("observation_monotonic_s"),
            accepted_monotonic_s=col("accepted_monotonic_s"),
            requested_monotonic_s=col("requested_monotonic_s"),
            ready_monotonic_s=col("ready_monotonic_s"),
            action_origin_monotonic_s=col("action_origin_monotonic_s"),
            handoff_at_monotonic_s=col("handoff_at_monotonic_s"),
            packet_id=col("packet_id", dtype=np.int64, default=-1),
            start_frame=col("start_frame", dtype=np.int64, default=-1),
            rtc_applied=col("rtc_applied", dtype=bool, default=False),
            rtc_s=col("rtc_s", dtype=np.int64, default=-1),
            rtc_d=col("rtc_d", dtype=np.int64, default=-1),
            rtc_guided_gap=col("rtc_guided_gap"),
            overlap_s=col("overlap_s", dtype=np.int64, default=-1),
            overlap_d=col("overlap_d", dtype=np.int64, default=-1),
            overlap_gap=col("overlap_gap"),
            d_hat=col("d_hat", dtype=np.int64, default=-1),
            d_actual=col("d_actual"),
            anchor_step=col("anchor_step", dtype=np.int64, default=-1),
            plan_id=col("plan_id", dtype=np.int64, default=-1),
            observation_age_ms=col("observation_age_ms"),
            image_age_ms=col("image_age_ms"),
            previous_command_difference=col("previous_command_difference"),
            request_to_ready_ms=col("request_to_ready_ms"),
            client_prep_ms=col("client_prep_ms"),
            infer_call_ms=col("infer_call_ms"),
            merge_accepted=col("merge_accepted", dtype=bool, default=True),
            frame_times_s=frame_times,
            step_dt=step_dt,
            retime_status=retime_status,
        )
