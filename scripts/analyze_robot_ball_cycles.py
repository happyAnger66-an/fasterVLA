#!/usr/bin/env python3
"""从 robot-live control_trace 统计每球周期（与示范 / FAST60 口径对齐）。

右手夹爪命令 command21[-1]（= policy dim 23）：
  · 闭合：grip > -0.39；张开间隙 < merge_gap_s 合并；夹持 ≥ min_close_s 才算尝试
  · 松开后 regrasp_s 内又夹 = 同球重试
  · 松→松 < min_release_to_release_s（默认 3s）= 重试，合并且不入真周期统计

阶段（加速对照用）：
  · grasp→place  夹→放（搬运）：该球首次夹住 → 末次松开
  · release→grasp 松→夹（找球）：上次松开 → 下次首次夹住
  · release→release 松→松（单球周期）：相邻两次成功松开间隔（≥3s）

例：
  python3 tools/analyze_robot_ball_cycles.py \\
    --runs /data/project_r/fastervla/results/robot-live-20261124-213252-1f6b6d \\
           /data/project_r/fastervla/results/robot-live-20261124-213731-f6aab0 \\
    --label baseline_1x --label k1p1_u44
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from fastervla.deploy.analysis.ball_cycles import (  # noqa: E402
    GRIP_CLOSED,
    analyze_ball_cycles,
)


def analyze_run(
    run_dir: Path,
    *,
    merge_gap_s: float,
    min_close_s: float,
    regrasp_s: float,
    min_release_to_release_s: float,
) -> dict:
    ctl_path = run_dir / "control_trace.npz"
    if not ctl_path.is_file():
        return {"run": run_dir.name, "error": "missing_control_trace"}
    ctl = dict(np.load(ctl_path, allow_pickle=True))
    out = analyze_ball_cycles(
        ctl,
        merge_gap_s=merge_gap_s,
        min_close_s=min_close_s,
        regrasp_s=regrasp_s,
        min_release_to_release_s=min_release_to_release_s,
    )
    if "error" in out:
        return {"run": run_dir.name, "error": out["error"]}
    out["run"] = run_dir.name.replace("robot-live-", "")
    out["run_dir"] = str(run_dir)
    g = np.asarray(ctl["command21"], dtype=np.float64)[:, -1]
    out["grip_cmd_range"] = [float(g.min()), float(g.max())]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--label", action="append", default=None, help="optional label per run")
    ap.add_argument("--merge-gap-s", type=float, default=0.4)
    ap.add_argument("--min-close-s", type=float, default=0.5)
    ap.add_argument("--regrasp-s", type=float, default=1.0)
    ap.add_argument(
        "--min-release-to-release-s",
        type=float,
        default=3.0,
        help="松→松 below this is a retry (merge; exclude from cycle stats)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    labels = list(args.label or [])
    rows = []
    for i, p in enumerate(args.runs):
        r = analyze_run(
            p,
            merge_gap_s=args.merge_gap_s,
            min_close_s=args.min_close_s,
            regrasp_s=args.regrasp_s,
            min_release_to_release_s=args.min_release_to_release_s,
        )
        if i < len(labels):
            r["label"] = labels[i]
        rows.append(r)
        _print_row(r)

    refs = {
        "demo_30hz_full": {
            "grasp_to_place_mean": 3.75,
            "grasp_to_place_p50": 3.30,
            "release_to_release_mean": 8.37,
            "release_to_release_p50": 7.67,
            "note": "results/g2-fangqiu-opt/ball-cycle-stats/SUMMARY.json",
        },
        "robot_fp4_ep0_all": {
            "grasp_to_place_mean": 2.24,
            "grasp_to_place_p50": 2.29,
            "release_to_release_mean": 8.46,
            "release_to_release_p50": 6.30,
            "note": "ROBOT_GRASP_SUMMARY.json · ≥3球",
        },
        "robot_grasp_speed_ref": {
            "grasp_to_place_mean": 2.85,
            "release_to_release_mean": 6.85,
            "note": "37EnzHdr tube_fp4_ep0_grasp_speed_1_0",
        },
    }
    payload = {
        "params": {
            "merge_gap_s": args.merge_gap_s,
            "min_close_s": args.min_close_s,
            "regrasp_s": args.regrasp_s,
            "min_release_to_release_s": args.min_release_to_release_s,
            "grip_closed_thr": GRIP_CLOSED,
        },
        "definition": {
            "grasp_to_place": "夹→放（搬运）：首次夹住→末次松开",
            "release_to_grasp": "松→夹（找球）：上次松开→下次首次夹住（仅真周期）",
            "release_to_release": (
                f"松→松（单球周期）：<{args.min_release_to_release_s:g}s 视为重试"
            ),
            "hold_only": "有效夹持时长之和（不含重抓间隙）",
            "seek_motion": "每个松→夹窗口内末端停/慢/快时间占比",
        },
        "reference": refs,
        "runs": rows,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def _print_row(r: dict) -> None:
    if "error" in r:
        print(f"== {r.get('run')} ERROR {r['error']}")
        return
    lab = r.get("label") or r["run"]
    g = r["grasp_to_place"]
    f = r["release_to_grasp"]
    p = r["release_to_release"]
    a = r["attempts"]
    sm = r.get("seek_motion") or {}
    print(
        f"== {lab}  balls={r['n_balls']} (raw={r.get('n_raw_balls')}, "
        f"short_merged={r.get('n_short_cycles_merged')}) "
        f"dur={r['dur_s']:.1f}s bpm={r['balls_per_min']:.2f}"
    )
    print(
        f"   夹→放（搬运） mean/p50/p90 = {g.get('mean', float('nan')):.2f}/"
        f"{g.get('p50', float('nan')):.2f}/{g.get('p90', float('nan')):.2f}  n={g.get('n', 0)}"
    )
    print(
        f"   松→夹（找球） mean/p50/p90 = {f.get('mean', float('nan')):.2f}/"
        f"{f.get('p50', float('nan')):.2f}/{f.get('p90', float('nan')):.2f}  n={f.get('n', 0)}"
    )
    print(
        f"   松→松（单球周期） mean/p50/p90 = {p.get('mean', float('nan')):.2f}/"
        f"{p.get('p50', float('nan')):.2f}/{p.get('p90', float('nan')):.2f}  n={p.get('n', 0)}  "
        f"(<{r.get('min_release_to_release_s', 3):g}s 已排除)"
    )
    print(
        f"   找球停/慢/快 p50 = {sm.get('dwell_frac_p50')}/"
        f"{sm.get('slow_frac_p50')}/{sm.get('fast_frac_p50')}  "
        f"dwell_s_p50={sm.get('dwell_s_p50')}"
    )
    print(
        f"   attempts mean={a['mean']:.2f} first_try={a['first_try']:.0%} "
        f"within_2={a['within_2']:.0%} exact={a['exact']}"
    )
    print(f"   per_ball: {r['per_ball']}")


if __name__ == "__main__":
    raise SystemExit(main())
