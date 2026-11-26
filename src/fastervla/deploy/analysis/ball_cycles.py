"""Ball-cycle timing from control_trace right-gripper command.

Stages (wall-clock, from command21 right grip):
  · grasp→place（搬运）: first close → last open of a ball
  · release→grasp（找球）: previous open → next first close
  · release→release（单球周期）: adjacent successful opens

Short release→release (&lt; ``min_release_to_release_s``, default 3s) is treated
as a regrasp retry: merged into the same ball and excluded from cycle stats.
"""

from __future__ import annotations

from typing import Any

import numpy as np

GRIP_CLOSED = -0.39
CMD_GRIP_R = 20  # command21 right gripper
SEEK_DWELL_MPS = 0.03
SEEK_SLOW_MPS = 0.15
# Below this, release→release is a retry, not a real single-ball period.
MIN_RELEASE_TO_RELEASE_S = 3.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(idx[::2].tolist(), idx[1::2].tolist()))


def _clamp_idx(i: int, n: int) -> int:
    """Clamp run index for timestamp lookup.

    ``_runs`` ends are exclusive and may equal ``n`` when the trace ends closed.
    """
    if n <= 0:
        return 0
    return int(min(max(int(i), 0), n - 1))


def ball_events(
    g: np.ndarray,
    fps: float,
    *,
    merge_gap_s: float = 0.4,
    min_close_s: float = 0.5,
    regrasp_s: float = 1.0,
) -> list[dict[str, Any]]:
    raw = _runs(np.asarray(g, dtype=np.float64) > GRIP_CLOSED)
    merged: list[list[int]] = []
    for s, e in raw:
        if merged and (s - merged[-1][1]) / fps < merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    held = [(s, e) for s, e in merged if (e - s) / fps >= min_close_s]

    balls: list[dict[str, Any]] = []
    cur: list[tuple[int, int]] = []
    for s, e in held:
        if cur and (s - cur[-1][1]) / fps < regrasp_s:
            cur.append((s, e))
        else:
            if cur:
                balls.append(_ball(cur, fps))
            cur = [(s, e)]
    if cur:
        balls.append(_ball(cur, fps))
    return balls


def _ball(holds: list[tuple[int, int]], fps: float) -> dict[str, Any]:
    s0, e_last = holds[0][0], holds[-1][1]
    hold_s = sum((e - s) / fps for s, e in holds)
    return {
        "first_grasp_i": s0,
        "final_release_i": e_last,
        "n_attempts": len(holds),
        "grasp_to_place_s": (e_last - s0) / fps,
        "hold_s": hold_s,
        "gap_s": (e_last - s0) / fps - hold_s,
    }


def merge_short_release_cycles(
    balls: list[dict[str, Any]],
    fps: float,
    *,
    min_release_to_release_s: float = MIN_RELEASE_TO_RELEASE_S,
) -> tuple[list[dict[str, Any]], int]:
    """Merge consecutive balls whose release→release gap is below the floor.

    A sub-3s 松→松 is almost always a failed place + quick regrasp, not a new ball.
    """
    if not balls:
        return [], 0
    out: list[dict[str, Any]] = [dict(balls[0])]
    n_merged = 0
    min_s = float(min_release_to_release_s)
    for b in balls[1:]:
        prev = out[-1]
        r2r = (int(b["final_release_i"]) - int(prev["final_release_i"])) / fps
        if r2r < min_s:
            prev["n_attempts"] = int(prev["n_attempts"]) + int(b["n_attempts"])
            prev["final_release_i"] = int(b["final_release_i"])
            prev["hold_s"] = float(prev["hold_s"]) + float(b["hold_s"])
            prev["grasp_to_place_s"] = (
                int(prev["final_release_i"]) - int(prev["first_grasp_i"])
            ) / fps
            prev["gap_s"] = float(prev["grasp_to_place_s"]) - float(prev["hold_s"])
            prev["short_cycle_merged"] = True
            n_merged += 1
        else:
            out.append(dict(b))
    return out, n_merged


def summarize_times(xs: list[float]) -> dict[str, Any]:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _ee_speed_ref_mps(ctl: dict[str, np.ndarray], dt: float) -> np.ndarray | None:
    """Right EE speed of policy reference (cmd layout via measured+reference)."""
    try:
        from fastervla.deploy.analysis.control_quality import _policy24
        from fastervla.deploy.analysis.ee_kinematics import load_urdf_fk

        loaded = load_urdf_fk(None)
        if loaded is None:
            return None
        fk, names = loaded
        from urdf_fk import fk_batch  # type: ignore

        ref = _policy24(ctl["measured_state24"], ctl["reference21"])
        p = np.asarray(
            fk_batch(fk, names, ref, ("arm_r_end_link",))["arm_r_end_link"]["xyz"]
        )
        return np.linalg.norm(np.gradient(p, dt, axis=0), axis=1)
    except Exception:  # noqa: BLE001
        return None


def seek_motion_share(
    *,
    t: np.ndarray,
    balls: list[dict[str, Any]],
    ee_speed_mps: np.ndarray | None,
    dwell_mps: float = SEEK_DWELL_MPS,
    slow_mps: float = SEEK_SLOW_MPS,
) -> dict[str, Any]:
    """Per-ball release→grasp（找球）dwell/slow/fast time shares."""
    if ee_speed_mps is None or len(balls) < 2:
        return {"n": 0, "per_ball": [], "dwell_frac_p50": None, "definition": {
            "window": "release→grasp（找球）",
            "dwell_mps": dwell_mps,
            "slow_mps": slow_mps,
        }}
    v = np.asarray(ee_speed_mps, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    n = int(t.size)
    for i in range(1, len(balls)):
        i0 = int(balls[i - 1]["final_release_i"])
        i1 = int(balls[i]["first_grasp_i"])
        if i1 <= i0:
            continue
        seg = v[i0:i1]
        dur = float(t[_clamp_idx(i1, n)] - t[_clamp_idx(i0, n)])
        if seg.size == 0 or dur <= 0:
            continue
        # Equal-tick approx within the window.
        dwell = float(np.mean(seg < dwell_mps))
        slow = float(np.mean((seg >= dwell_mps) & (seg < slow_mps)))
        fast = float(np.mean(seg >= slow_mps))
        rows.append(
            {
                "ball_i": i,
                "release_to_grasp_s": round(dur, 3),
                "dwell_frac": round(dwell, 3),
                "slow_frac": round(slow, 3),
                "fast_frac": round(fast, 3),
                "dwell_s": round(dur * dwell, 3),
                "slow_s": round(dur * slow, 3),
                "fast_s": round(dur * fast, 3),
            }
        )
    dwell_fracs = [r["dwell_frac"] for r in rows]
    return {
        "n": len(rows),
        "dwell_frac_p50": float(np.median(dwell_fracs)) if dwell_fracs else None,
        "slow_frac_p50": float(np.median([r["slow_frac"] for r in rows])) if rows else None,
        "fast_frac_p50": float(np.median([r["fast_frac"] for r in rows])) if rows else None,
        "dwell_s_p50": float(np.median([r["dwell_s"] for r in rows])) if rows else None,
        "per_ball": rows,
        "definition": {
            "window": "release→grasp（找球）",
            "dwell_mps": dwell_mps,
            "slow_mps": slow_mps,
        },
    }


def analyze_ball_cycles(
    ctl: dict[str, np.ndarray],
    *,
    merge_gap_s: float = 0.4,
    min_close_s: float = 0.5,
    regrasp_s: float = 1.0,
    min_release_to_release_s: float = MIN_RELEASE_TO_RELEASE_S,
    ee_speed: bool = True,
) -> dict[str, Any]:
    t = np.asarray(ctl["monotonic_s"], dtype=np.float64)
    cmd = np.asarray(ctl["command21"], dtype=np.float64)
    if t.size < 2 or cmd.ndim != 2 or cmd.shape[1] <= CMD_GRIP_R:
        return {"error": "too_short_or_bad_cmd"}
    dt = float(np.median(np.diff(t)))
    fps = 1.0 / max(dt, 1e-6)
    g = cmd[:, CMD_GRIP_R]
    raw_balls = ball_events(
        g, fps, merge_gap_s=merge_gap_s, min_close_s=min_close_s, regrasp_s=regrasp_s
    )
    balls, n_short_merged = merge_short_release_cycles(
        raw_balls, fps, min_release_to_release_s=min_release_to_release_s
    )

    g2p = [b["grasp_to_place_s"] for b in balls]
    hold = [b["hold_s"] for b in balls]
    r2g: list[float] = []
    r2r_period: list[float] = []
    r2r_excluded: list[float] = []
    per_gaps: list[dict[str, Any]] = [{"release_to_grasp_s": None, "release_to_release_s": None}]
    for i in range(1, len(balls)):
        prev_rel = balls[i - 1]["final_release_i"]
        cur_g = balls[i]["first_grasp_i"]
        cur_rel = balls[i]["final_release_i"]
        gap_g = (cur_g - prev_rel) / fps
        gap_r = (cur_rel - prev_rel) / fps
        # Belt-and-suspenders: never let sub-floor gaps into cycle summaries.
        if gap_r < float(min_release_to_release_s):
            r2r_excluded.append(gap_r)
            per_gaps.append(
                {
                    "release_to_grasp_s": round(gap_g, 3),
                    "release_to_release_s": round(gap_r, 3),
                    "excluded_short_cycle": True,
                }
            )
            continue
        r2g.append(gap_g)
        r2r_period.append(gap_r)
        per_gaps.append(
            {
                "release_to_grasp_s": round(gap_g, 3),
                "release_to_release_s": round(gap_r, 3),
                "excluded_short_cycle": False,
            }
        )

    attempts = [b["n_attempts"] for b in balls]
    wall = float(t[-1] - t[0])
    n_balls = len(balls)

    ee_v = _ee_speed_ref_mps(ctl, dt) if ee_speed else None
    seek = seek_motion_share(t=t, balls=balls, ee_speed_mps=ee_v)

    return {
        "fps": fps,
        "dur_s": wall,
        "n_balls": n_balls,
        "n_raw_balls": len(raw_balls),
        "n_short_cycles_merged": int(n_short_merged),
        "n_short_cycles_excluded": len(r2r_excluded),
        "min_release_to_release_s": float(min_release_to_release_s),
        "balls_per_min": (n_balls / wall * 60.0) if wall > 0 and n_balls else 0.0,
        "grasp_to_place": summarize_times(g2p),
        "hold_only": summarize_times(hold),
        "release_to_grasp": summarize_times(r2g),
        "release_to_release": summarize_times(r2r_period),
        "release_to_release_excluded_short": summarize_times(r2r_excluded),
        "attempts": {
            "mean": float(np.mean(attempts)) if attempts else 0.0,
            "first_try": float(np.mean([a == 1 for a in attempts])) if attempts else 0.0,
            "within_2": float(np.mean([a <= 2 for a in attempts])) if attempts else 0.0,
            "exact": {
                str(k): int(sum(1 for a in attempts if a == k))
                for k in sorted(set(attempts))
            },
        },
        "seek_motion": seek,
        "per_ball": [
            {
                "i": i,
                "grasp_to_place_s": round(b["grasp_to_place_s"], 3),
                "hold_s": round(b["hold_s"], 3),
                "n_attempts": b["n_attempts"],
                "short_cycle_merged": bool(b.get("short_cycle_merged")),
                "t_grasp": round(
                    float(t[_clamp_idx(b["first_grasp_i"], t.size)] - t[0]), 3
                ),
                # final_release_i is an exclusive end from _runs; may equal len(t).
                "t_release": round(
                    float(t[_clamp_idx(b["final_release_i"], t.size)] - t[0]), 3
                ),
                **per_gaps[i],
            }
            for i, b in enumerate(balls)
        ],
        "definition": {
            "grasp_to_place": "夹→放（搬运）：首次夹住→末次松开",
            "release_to_grasp": "松→夹（找球）：上次松开→下次首次夹住（仅计入真周期）",
            "release_to_release": (
                f"松→松（单球周期）：相邻成功松开间隔；"
                f"<{min_release_to_release_s:g}s 视为重试，合并/不入统计"
            ),
            "min_release_to_release_s": float(min_release_to_release_s),
        },
    }
