"""Shared fixtures for fasterVLA tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture
def tmp_limits_yaml(tmp_path: Path) -> Path:
    """Write a minimal joint velocity limits YAML and return its path."""
    path = tmp_path / "joint_velocity_limits.yaml"
    data = {
        "joint_names": ["j0", "j1", "j2"],
        "v_max": [1.0, 2.0, 0.5],
        "velocity_scale": 1.0,
        "ignore_velocity_dims": [],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def constant_speed_traj() -> np.ndarray:
    """
    Synthetic joint trajectory with constant per-frame deltas.

    Shape [T, 3]. Frame i has q = i * delta, so a jump of k frames
    changes q by k * delta.
    """
    t = 40
    delta = np.asarray([0.10, 0.05, 0.02], dtype=np.float64)
    idx = np.arange(t, dtype=np.float64)[:, None]
    return idx * delta
