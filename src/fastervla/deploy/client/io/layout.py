"""G2 policy↔command layout (24 → 21 with waist baseline hold)."""

from __future__ import annotations

import numpy as np

from fastervla.core.clock import monotonic_s
from fastervla.core.types import RobotCommand

G2_ACTION_DIM = 24
G2_COMMAND_DIM = 21
COMMAND_INDICES = np.asarray([*range(5), *range(8, 24)], dtype=np.int64)
# Policy indices for dual-arm joints entering L2/L3 (14 dims).
ARM_POLICY_SLICE = (8, 22)


class LayoutMapper:
    """Map between policy layout (24) and G2 command layout (21)."""

    def to_policy(self, q: np.ndarray) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape[0] != G2_ACTION_DIM:
            raise ValueError(f"to_policy expects dim {G2_ACTION_DIM}, got {arr.shape}")
        return arr

    def from_policy(
        self,
        action: np.ndarray,
        *,
        waist_baseline: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> RobotCommand:
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.shape[0] != G2_ACTION_DIM:
            raise ValueError(f"from_policy expects dim {G2_ACTION_DIM}, got {arr.shape}")
        q_des = arr[COMMAND_INDICES].copy()
        if waist_baseline is not None:
            wb = np.asarray(waist_baseline, dtype=np.float64).reshape(-1)
            if wb.shape[0] != 5:
                raise ValueError(f"waist_baseline must have length 5, got {wb.shape}")
            q_des[:5] = wb
        ts = float(monotonic_s() if timestamp is None else timestamp)
        return RobotCommand(q_des=q_des, qd_des=None, timestamp=ts)
