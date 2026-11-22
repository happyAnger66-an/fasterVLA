"""Load per-joint velocity limits from YAML for training-time frame skip."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class JointVelocityLimits:
    joint_names: list[str]
    v_max: np.ndarray
    velocity_scale: float
    ignore_velocity_dims: list[int] = field(default_factory=list)


def load_joint_velocity_limits(path: str | Path) -> JointVelocityLimits:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        data: dict[str, Any] = yaml.safe_load(fp) or {}

    required = ("joint_names", "v_max", "velocity_scale")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")

    joint_names = list(data["joint_names"])
    v_max = np.asarray(data["v_max"], dtype=np.float64).reshape(-1)
    if len(joint_names) != v_max.shape[0]:
        raise ValueError(
            f"joint_names length ({len(joint_names)}) != v_max length ({v_max.shape[0]})"
        )

    ignore = list(data.get("ignore_velocity_dims") or [])
    return JointVelocityLimits(
        joint_names=joint_names,
        v_max=v_max,
        velocity_scale=float(data["velocity_scale"]),
        ignore_velocity_dims=ignore,
    )


def effective_v_max(limits: JointVelocityLimits) -> np.ndarray:
    out = np.asarray(limits.v_max, dtype=np.float64) * float(limits.velocity_scale)
    out = out.copy()
    for dim in limits.ignore_velocity_dims:
        out[int(dim)] = np.inf
    return out
