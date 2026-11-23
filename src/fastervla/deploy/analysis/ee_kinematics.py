"""G2 end-effector FK for offline EE speed metrics (reuses g2_tools urdf_fk)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# measured_state24 = 22 body/head/arm + 2 grippers (same order as Gdk G2_JOINT_NAMES+fingers).
_MEAS24_NAMES: tuple[str, ...] = (
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
    "idx31_gripper_l_inner_joint1",
    "idx71_gripper_r_inner_joint1",
)


def _g2_scripts_dir() -> Path | None:
    here = Path(__file__).resolve()
    # .../r-infra/robot/fasterVLA/src/fastervla/deploy/analysis/this.py
    for parent in here.parents:
        cand = parent / "g2_tools" / "scripts"
        if (cand / "urdf_fk.py").is_file():
            return cand
    return None


def load_urdf_fk(urdf_hint: str | Path | None = None) -> tuple[Any, list[str]] | None:
    """Return ``(UrdfFK, joint_names_24)`` or None if URDF / helper unavailable."""
    scripts = _g2_scripts_dir()
    if scripts is None:
        return None
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        from urdf_fk import UrdfFK, resolve_urdf_path  # type: ignore
    except ImportError:
        return None
    path = resolve_urdf_path(urdf_hint)
    if path is None:
        return None
    return UrdfFK(path), list(_MEAS24_NAMES)


def ee_xyz_series(
    q24: np.ndarray,
    *,
    urdf_hint: str | Path | None = None,
    stride: int = 5,
    ee_links: tuple[str, ...] = ("arm_l_end_link", "arm_r_end_link"),
) -> dict[str, np.ndarray] | None:
    """Subsampled EE xyz for each link. ``q24`` shape ``(T, 24)``."""
    loaded = load_urdf_fk(urdf_hint)
    if loaded is None:
        return None
    fk, names = loaded
    from urdf_fk import fk_batch  # type: ignore

    q = np.asarray(q24, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 22:
        return None
    if q.shape[1] < 24:
        pad = np.zeros((q.shape[0], 24 - q.shape[1]), dtype=np.float64)
        q = np.concatenate([q, pad], axis=1)
    stride = max(1, int(stride))
    idx = np.arange(0, q.shape[0], stride)
    batch = fk_batch(fk, names, q[idx], ee_links)
    out: dict[str, np.ndarray] = {"_index": idx.astype(np.int64)}
    for ee in ee_links:
        out[ee] = np.asarray(batch[ee]["xyz"], dtype=np.float64)
    return out


def ee_speed_metrics(
    t: np.ndarray,
    q24: np.ndarray,
    *,
    urdf_hint: str | Path | None = None,
    stride: int = 5,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Linear EE speed (m/s) from measured joints via URDF FK.

    Reports per-arm and max-of-arms percentiles. Missing URDF → empty dict.
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    q = np.asarray(q24, dtype=np.float64)
    if t.size < 3 or q.shape[0] != t.size:
        return {}
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape[0] != t.size or not np.any(mask):
            return {}
        t = t[mask]
        q = q[mask]
    series = ee_xyz_series(q, urdf_hint=urdf_hint, stride=stride)
    if series is None:
        return {"ee_fk_available": False}
    idx = series["_index"]
    ts = t[idx]
    speeds: list[np.ndarray] = []
    out: dict[str, Any] = {"ee_fk_available": True, "ee_stride": int(stride)}
    for ee in ("arm_l_end_link", "arm_r_end_link"):
        xyz = series[ee]
        if xyz.shape[0] < 2:
            continue
        dt = np.diff(ts)
        disp = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        ok = dt > 1e-6
        if not np.any(ok):
            continue
        spd = disp[ok] / dt[ok]
        speeds.append(spd)
        tag = "L" if ee.startswith("arm_l") else "R"
        out[f"ee_speed_{tag}_p50_mps"] = float(np.percentile(spd, 50))
        out[f"ee_speed_{tag}_p95_mps"] = float(np.percentile(spd, 95))
        out[f"ee_speed_{tag}_mean_mps"] = float(np.mean(spd))
    if not speeds:
        return out
    both = np.concatenate(speeds)
    # Pairwise max of L/R when aligned lengths match; else pooled.
    if len(speeds) == 2 and speeds[0].shape == speeds[1].shape:
        both = np.maximum(speeds[0], speeds[1])
    out["ee_speed_p50_mps"] = float(np.percentile(both, 50))
    out["ee_speed_p95_mps"] = float(np.percentile(both, 95))
    out["ee_speed_mean_mps"] = float(np.mean(both))
    return out
