"""Soft-lead tracker aligned with h7_deploy arm_lag_compensation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LeadTrackerConfig:
    tau_s: float = 0.057
    gain: float = 0.25
    max_lead_rad: float = 0.06
    lead_smooth_tau_s: float = 0.040
    hold_during_blend: bool = True


class LeadTracker:
    """Apply clipped lead on command dims [5:19] (14-DoF arms)."""

    ARM_SLICE = slice(5, 19)

    def __init__(self, config: LeadTrackerConfig | None = None) -> None:
        self.config = config or LeadTrackerConfig()
        self._lead_filt = np.zeros(14, dtype=np.float64)
        self._prev_now: float | None = None

    def step(
        self,
        reference: np.ndarray,
        *,
        ref_ahead: np.ndarray,
        ref_vel: np.ndarray,
        now: float,
        blending: bool = False,
    ) -> np.ndarray:
        ref = np.asarray(reference, dtype=np.float64).reshape(-1).copy()
        if ref.shape[0] != 21:
            raise ValueError(f"reference must be length 21, got {ref.shape}")
        r_ahead = np.asarray(ref_ahead, dtype=np.float64).reshape(14)
        v_ahead = np.asarray(ref_vel, dtype=np.float64).reshape(14)
        cfg = self.config

        if cfg.hold_during_blend and blending:
            lead = np.zeros(14, dtype=np.float64)
        else:
            desired = r_ahead + cfg.tau_s * v_ahead
            lead = cfg.gain * (desired - ref[self.ARM_SLICE])
            lead = np.clip(lead, -cfg.max_lead_rad, cfg.max_lead_rad)

        if (
            cfg.lead_smooth_tau_s > 0.0
            and self._prev_now is not None
            and now > self._prev_now
        ):
            dt = float(now - self._prev_now)
            alpha = 1.0 - math.exp(-dt / cfg.lead_smooth_tau_s)
            self._lead_filt = self._lead_filt + alpha * (lead - self._lead_filt)
            lead = self._lead_filt
        else:
            self._lead_filt = np.asarray(lead, dtype=np.float64).copy()

        self._prev_now = float(now)
        ref[self.ARM_SLICE] = ref[self.ARM_SLICE] + lead
        return ref
