"""PlanBuffer: model-step ruler for L1 async execution (F2)."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from fastervla.core.types import Prefix, TimedChunk
from fastervla.deploy.postprocess.pchip import pchip_eval


class PlanBuffer:
    def __init__(self, *, interpolation: str = "linear") -> None:
        mode = str(interpolation or "linear").lower()
        if mode not in ("linear", "pchip"):
            raise ValueError(f"unknown interpolation {interpolation!r}")
        self._lock = threading.RLock()
        self._interpolation = mode
        self._waypoints: np.ndarray | None = None
        self._step_dt: np.ndarray | None = None
        self._speed: np.ndarray | None = None
        self._progress = 0.0
        self._origin_monotonic = 0.0
        self._plan_id = 0
        self._generation = 0
        self._last_merge: dict[str, Any] = {}
        self._last_d_hat: int | None = None

    @property
    def interpolation(self) -> str:
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: str) -> None:
        mode = str(value or "linear").lower()
        if mode not in ("linear", "pchip"):
            raise ValueError(f"unknown interpolation {value!r}")
        with self._lock:
            self._interpolation = mode

    # ---- public properties ----

    @property
    def progress(self) -> float:
        with self._lock:
            return float(self._progress)

    @property
    def horizon(self) -> int:
        with self._lock:
            self._require_loaded()
            assert self._waypoints is not None
            return int(self._waypoints.shape[0])

    @property
    def action_dim(self) -> int:
        with self._lock:
            self._require_loaded()
            assert self._waypoints is not None
            return int(self._waypoints.shape[1])

    @property
    def waypoints(self) -> np.ndarray:
        with self._lock:
            self._require_loaded()
            assert self._waypoints is not None
            return self._waypoints.copy()

    @property
    def step_dt(self) -> np.ndarray:
        with self._lock:
            self._require_loaded()
            assert self._step_dt is not None
            return self._step_dt.copy()

    @property
    def origin_monotonic(self) -> float:
        with self._lock:
            return float(self._origin_monotonic)

    @property
    def plan_id(self) -> int:
        with self._lock:
            return int(self._plan_id)

    @property
    def last_merge(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_merge)

    def remaining_steps(self) -> float:
        with self._lock:
            self._require_loaded()
            assert self._waypoints is not None
            return max(0.0, float(self._waypoints.shape[0]) - self._progress)

    # ---- API ----

    def load(self, timed: TimedChunk, *, origin_monotonic: float = 0.0) -> None:
        with self._lock:
            wp = np.asarray(timed.chunk.waypoints, dtype=np.float64)
            step_dt = np.asarray(timed.step_dt, dtype=np.float64)
            speed = np.asarray(timed.speed, dtype=np.float64)
            if wp.ndim != 2 or step_dt.shape[0] != wp.shape[0]:
                raise ValueError("TimedChunk shape mismatch")
            self._waypoints = wp.copy()
            self._step_dt = step_dt.copy()
            self._speed = speed.copy()
            self._progress = 0.0
            self._origin_monotonic = float(origin_monotonic)
            self._plan_id = 0
            gen = timed.chunk.meta.get("plan_generation")
            self._generation = int(gen) if gen is not None else 0
            self._last_merge = {}
            self._last_d_hat = None

    def advance(self, dt_real: float, alpha: float = 1.0) -> float:
        with self._lock:
            self._require_loaded()
            assert self._waypoints is not None and self._step_dt is not None
            dt = float(dt_real) * float(alpha)
            if dt <= 0 or alpha == 0.0:
                return float(self._progress)
            h = int(self._waypoints.shape[0])
            remaining = dt
            p = float(self._progress)
            while remaining > 1e-15 and p < h - 1e-15:
                idx = int(np.floor(p))
                if idx >= h:
                    break
                # Duration of segment idx → idx+1. Last index uses last dt as hold.
                seg_i = min(idx, h - 1)
                seg = float(self._step_dt[seg_i])
                if seg <= 0:
                    p = float(idx + 1)
                    continue
                frac = p - idx
                rem_in_seg = seg * (1.0 - frac)
                if remaining < rem_in_seg - 1e-15:
                    p = p + remaining / seg
                    remaining = 0.0
                else:
                    remaining -= rem_in_seg
                    p = float(idx + 1)
            self._progress = min(p, float(h))
            return float(self._progress)

    def estimate_delay_steps(self, latency_s: float) -> int:
        with self._lock:
            self._require_loaded()
            assert self._waypoints is not None and self._step_dt is not None
            if latency_s <= 0:
                return 0
            h = int(self._waypoints.shape[0])
            remaining = float(latency_s)
            p = float(self._progress)
            steps = 0
            last_seg = float(self._step_dt[-1]) if h > 0 else 1.0 / 30.0
            while remaining > 1e-15:
                if p >= h - 1e-15:
                    # Extrapolate past plan end so d̂ can exceed remaining (starved).
                    if last_seg <= 0:
                        break
                    extra = int(np.ceil(remaining / last_seg - 1e-15))
                    steps += max(0, extra)
                    break
                idx = int(np.floor(p))
                seg_i = min(idx, h - 1)
                seg = float(self._step_dt[seg_i])
                last_seg = seg if seg > 0 else last_seg
                if seg <= 0:
                    p = float(idx + 1)
                    continue
                frac = p - idx
                rem_in_seg = seg * (1.0 - frac)
                if rem_in_seg <= 1e-15:
                    p = float(idx + 1)
                    continue
                # Any positive coverage of this segment counts one (ceil).
                steps += 1
                if remaining <= rem_in_seg + 1e-15:
                    break
                remaining -= rem_in_seg
                p = float(idx + 1)
            return int(steps)

    def prefix(self, d_hat: int) -> Prefix:
        with self._lock:
            self._require_loaded()
            assert self._waypoints is not None
            d_hat = int(d_hat)
            if d_hat < 0:
                raise ValueError("d_hat must be >= 0")
            start = int(np.floor(self._progress))
            h = int(self._waypoints.shape[0])
            end = min(h, start + d_hat)
            actions = self._waypoints[start:end].astype(np.float32)
            if actions.shape[0] < d_hat:
                # Pad by holding last waypoint if near end of plan.
                pad_n = d_hat - actions.shape[0]
                last = (
                    actions[-1:]
                    if actions.shape[0] > 0
                    else np.zeros((1, self._waypoints.shape[1]), dtype=np.float32)
                )
                actions = np.concatenate(
                    [actions, np.repeat(last, pad_n, axis=0)], axis=0
                )
            self._last_d_hat = d_hat
            return Prefix(actions=actions, anchor_step=start)

    def merge(self, timed: TimedChunk, anchor_step: int) -> bool:
        with self._lock:
            self._require_loaded()
            assert self._waypoints is not None and self._step_dt is not None
            incoming_gen = timed.chunk.meta.get("plan_generation")
            if incoming_gen is not None and int(incoming_gen) < self._generation:
                return False

            d_actual = float(self._progress) - float(anchor_step)
            if d_actual < -1e-6:
                # Result older than current progress relative to its anchor.
                return False

            new_wp = np.asarray(timed.chunk.waypoints, dtype=np.float64).copy()
            new_dt = np.asarray(timed.step_dt, dtype=np.float64).copy()
            new_speed = np.asarray(timed.speed, dtype=np.float64).copy()
            self._waypoints = new_wp
            self._step_dt = new_dt
            self._speed = new_speed
            # Re-index: current execution point is index d within the new chunk.
            self._progress = max(0.0, d_actual)
            self._plan_id += 1
            if incoming_gen is not None:
                self._generation = int(incoming_gen)
            else:
                self._generation += 1
            self._last_merge = {
                "d_actual": float(d_actual),
                "d_hat": self._last_d_hat,
                "anchor_step": int(anchor_step),
                "plan_id": int(self._plan_id),
            }
            return True

    def reference(self, t_now: float, horizon_s: float = 0.0) -> np.ndarray:
        """Interpolate absolute waypoints at the current model-step progress.

        ``linear``: lerp between floor/ceil knots.
        ``pchip``: monotone cubic on wall-clock node times from ``step_dt``
        (matches old-stack soft arm path when ``step_dt`` is uniform).
        ``t_now`` / ``horizon_s`` reserved for absolute wall-clock queries.
        """
        with self._lock:
            self._require_loaded()
            assert self._waypoints is not None and self._step_dt is not None
            _ = (t_now, horizon_s)
            p = float(self._progress)
            h = int(self._waypoints.shape[0])
            if p <= 0:
                return self._waypoints[0].copy()
            if p >= h - 1:
                return self._waypoints[-1].copy()
            if self._interpolation == "pchip" and h >= 2:
                return self._reference_pchip(p)
            i0 = int(np.floor(p))
            i1 = min(i0 + 1, h - 1)
            frac = p - i0
            return (1.0 - frac) * self._waypoints[i0] + frac * self._waypoints[i1]

    def _reference_pchip(self, progress: float) -> np.ndarray:
        assert self._waypoints is not None and self._step_dt is not None
        h = int(self._waypoints.shape[0])
        # Node times: waypoint k is the pose at cumulative time of segments 0..k-1.
        # progress advances along those same segments (see advance()).
        knots = np.zeros(h, dtype=np.float64)
        if h >= 2:
            knots[1:] = np.cumsum(self._step_dt[: h - 1])
        # Map progress → wall time along the same ruler.
        p = float(progress)
        i0 = int(np.floor(p))
        i0 = max(0, min(i0, h - 2))
        frac = p - i0
        seg = float(self._step_dt[i0])
        t_query = float(knots[i0] + frac * seg)
        return pchip_eval(knots, self._waypoints, t_query)

    def _require_loaded(self) -> None:
        if self._waypoints is None or self._step_dt is None:
            raise RuntimeError("PlanBuffer.load() required before use")
