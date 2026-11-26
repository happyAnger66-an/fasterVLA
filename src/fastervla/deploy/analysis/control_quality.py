"""Control-quality metrics: how fast and how well measured joints follow commands.

All statistics use ``stage == step`` ticks, per arm (7 joints, L2 over joints),
pooled over arms while they move. Handoff windows (just after a plan switch) are
reported separately from steady-state.

Groups
  lag     path-matching lag per tick, binned by command joint speed
          (vs cmd = plant + transport; vs ref = net lag after lead);
          sliding-window velocity cross-correlation lag + peak rho
  error   following (tangential) / normal / contour (path distance) /
          shift-aligned residual
  gain    |v_meas(t+tau*)| / |v_cmd(t)| per speed bin
  freq    Welch FRF cmd->meas and ref->meas: -3 dB bandwidth, phase delay,
          command 95 % power frequency
  task    EE path lag / contour (mm), EE error at right-gripper events,
          stop settling time / overshoot; per task-phase (seek/grasp/carry/place)
          lag & contour (joint + EE), with a fast subset for high-speed analysis
  effort  command acc / jerk, tracker status mix, saturation excess
  smooth  SPARC on arm joint-speed profiles (cmd vs meas)
  timing  tick period jitter, tick compute, state read / send / state age
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fastervla.deploy.client.io.layout import COMMAND_INDICES

# command21 / measured24 column ranges per arm.
ARMS: dict[str, tuple[slice, slice]] = {
    "L": (slice(5, 12), slice(8, 15)),
    "R": (slice(12, 19), slice(15, 22)),
}
EE_LINKS = {"L": "arm_l_end_link", "R": "arm_r_end_link"}
CMD_GRIP_R = 20
GRIP_CLOSED = -0.39
# Align with TimeAxisQP contact_window_frames=10 @ 30 Hz policy.
CONTACT_WINDOW_S = 10.0 / 30.0
PHASE_NAMES = ("seek", "grasp", "carry", "place")
PHASE_FAST_JOINT_RAD_S = 1.0
PHASE_FAST_EE_MPS = 0.30
# G2_t2v2_crsB URDF velocity limits for idx61..67_arm_r_joint1..7 (rad/s).
# Phase speed uses max_|q̇|; util = max_i(|q̇_i|/vlim_i). Target util≥0.8 on seek/carry.
G2_ARM_R_VEL_LIMIT_RAD_S: tuple[float, ...] = (3.05, 3.05, 3.76, 3.76, 3.35, 3.35, 3.35)
VEL_LIMIT_UTIL_TARGET = 0.80

SPEED_BINS_RAD_S: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, np.inf)
EE_SPEED_BINS_MPS: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50, np.inf)
LAG_SEARCH_S = 0.40
HANDOFF_WINDOW_S = 0.05
XCORR_WIN_S = 1.0
XCORR_HOP_S = 0.25
XCORR_TAU_S = (-0.05, 0.30)
XCORR_MIN_RHO = 0.6
WELCH_NPERSEG = 256


# ---------------------------------------------------------------- helpers


def _pct(a: np.ndarray, q: float) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else None


def _p50_p95(a: np.ndarray, scale: float = 1.0) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "p50": None, "p95": None}
    return {
        "n": int(a.size),
        "p50": float(np.percentile(a, 50) * scale),
        "p95": float(np.percentile(a, 95) * scale),
    }


def _bin_labels(edges: tuple[float, ...]) -> list[str]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">={lo:g}")
    return out


def _binned(
    values: np.ndarray, key: np.ndarray, edges: tuple[float, ...], scale: float = 1.0
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label, lo, hi in zip(_bin_labels(edges), edges[:-1], edges[1:]):
        m = (key >= lo) & (key < hi)
        out[label] = _p50_p95(values[m], scale)
    return out


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or x.shape[0] < n:
        return x
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x, ((pad, n - 1 - pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def _step_mask(ctl: dict[str, np.ndarray], n: int) -> np.ndarray:
    if "stage" not in ctl or "stage_names" not in ctl:
        return np.ones(n, dtype=bool)
    names = [str(x) for x in np.asarray(ctl["stage_names"]).tolist()]
    if "step" not in names:
        return np.ones(n, dtype=bool)
    return np.asarray(ctl["stage"]) == names.index("step")


def _handoff_mask(plan_id: np.ndarray | None, t: np.ndarray, window_s: float) -> np.ndarray:
    n = t.shape[0]
    out = np.zeros(n, dtype=bool)
    if plan_id is None:
        return out
    pid = np.asarray(plan_id).reshape(-1)
    sw = np.flatnonzero(pid[1:] != pid[:-1]) + 1
    for i in sw:
        out[i : int(np.searchsorted(t, t[i] + window_s))] = True
    return out


def _policy24(measured24: np.ndarray, command21: np.ndarray) -> np.ndarray:
    q = np.asarray(measured24, dtype=np.float64).copy()
    q[:, COMMAND_INDICES] = command21
    return q


# ---------------------------------------------------------------- lag


def path_lag(
    target: np.ndarray, meas: np.ndarray, max_lag_ticks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-tick lag (ticks, fractional) and distance from ``meas[i]`` to the past target path.

    Searches ``target[i-k]`` for ``k in [0, max_lag_ticks]``, then refines on the
    two adjacent segments by projection. Returns ``(lag_ticks, dist, at_bound)``.
    """
    n = target.shape[0]
    k_max = int(max_lag_ticks)
    d2 = np.full((n, k_max + 1), np.inf)
    idx = np.arange(n)
    for k in range(k_max + 1):
        src = idx - k
        ok = src >= 0
        diff = target[src[ok]] - meas[ok]
        d2[ok, k] = np.einsum("ij,ij->i", diff, diff)
    k_star = np.argmin(d2, axis=1)
    s = idx - k_star

    best_lag = k_star.astype(np.float64)
    best_d = np.sqrt(d2[idx, k_star])
    for a_off, b_off in ((-1, 0), (0, 1)):
        a = s + a_off
        b = s + b_off
        ok = (a >= 0) & (b <= idx) & (b < n)
        if not np.any(ok):
            continue
        pa = target[a[ok]]
        ab = target[b[ok]] - pa
        ap = meas[ok] - pa
        den = np.einsum("ij,ij->i", ab, ab)
        u = np.where(den > 1e-18, np.einsum("ij,ij->i", ap, ab) / np.maximum(den, 1e-18), 0.0)
        u = np.clip(u, 0.0, 1.0)
        dist = np.linalg.norm(ap - u[:, None] * ab, axis=1)
        src_f = a[ok] + u
        lag = idx[ok] - src_f
        sel = np.flatnonzero(ok)
        better = dist < best_d[sel]
        best_d[sel[better]] = dist[better]
        best_lag[sel[better]] = lag[better]
    at_bound = (k_star == 0) | (k_star == k_max)
    return best_lag, best_d, at_bound


def _xcorr_windows(
    vc: np.ndarray,
    vm: np.ndarray,
    speed: np.ndarray,
    valid: np.ndarray,
    dt: float,
) -> dict[str, np.ndarray]:
    """Sliding-window velocity cross-correlation; positive lag = meas behind."""
    n = vc.shape[0]
    w = max(8, int(round(XCORR_WIN_S / dt)))
    hop = max(1, int(round(XCORR_HOP_S / dt)))
    t_lo = int(np.floor(XCORR_TAU_S[0] / dt))
    t_hi = int(np.ceil(XCORR_TAU_S[1] / dt))
    taus = np.arange(t_lo, t_hi + 1)
    lags, rhos, spd = [], [], []
    for w0 in range(max(0, -t_lo), n - w - t_hi, hop):
        sl = slice(w0, w0 + w)
        if np.mean(valid[sl]) < 0.9:
            continue
        mean_spd = float(np.mean(speed[sl]))
        if mean_spd < SPEED_BINS_RAD_S[0]:
            continue
        a = vc[sl]
        ea = float(np.sum(a * a))
        r = np.empty(taus.shape[0])
        for j, tau in enumerate(taus):
            b = vm[w0 + tau : w0 + tau + w]
            eb = float(np.sum(b * b))
            r[j] = float(np.sum(a * b)) / np.sqrt(max(ea * eb, 1e-18))
        j = int(np.argmax(r))
        frac = 0.0
        if 0 < j < r.shape[0] - 1:
            den = r[j - 1] - 2 * r[j] + r[j + 1]
            if abs(den) > 1e-12:
                frac = 0.5 * (r[j - 1] - r[j + 1]) / den
        lags.append((taus[j] + frac) * dt)
        rhos.append(r[j])
        spd.append(mean_spd)
    return {"lag_s": np.asarray(lags), "rho": np.asarray(rhos), "speed": np.asarray(spd)}


# ---------------------------------------------------------------- freq


def _welch_frf(x: np.ndarray, y: np.ndarray, dt: float, nperseg: int) -> dict[str, np.ndarray] | None:
    """H1 FRF x->y summed over columns (numpy Welch, Hann, linear detrend)."""
    n = x.shape[0]
    if n < nperseg:
        return None
    win = np.hanning(nperseg)
    hop = nperseg // 2
    tt = np.arange(nperseg, dtype=np.float64)
    a_mat = np.stack([tt, np.ones_like(tt)], 1)
    pinv = np.linalg.pinv(a_mat)
    f = np.fft.rfftfreq(nperseg, dt)
    pxx = np.zeros(f.shape[0])
    pyy = np.zeros(f.shape[0])
    pxy = np.zeros(f.shape[0], dtype=np.complex128)
    for s0 in range(0, n - nperseg + 1, hop):
        xs = x[s0 : s0 + nperseg]
        ys = y[s0 : s0 + nperseg]
        xs = xs - a_mat @ (pinv @ xs)
        ys = ys - a_mat @ (pinv @ ys)
        xf = np.fft.rfft(xs * win[:, None], axis=0)
        yf = np.fft.rfft(ys * win[:, None], axis=0)
        pxx += np.sum(np.abs(xf) ** 2, axis=1)
        pyy += np.sum(np.abs(yf) ** 2, axis=1)
        pxy += np.sum(np.conj(xf) * yf, axis=1)
    h = pxy / np.maximum(pxx, 1e-30)
    coh = np.abs(pxy) ** 2 / np.maximum(pxx * pyy, 1e-30)
    return {"f": f, "h": h, "coh": coh, "pxx": pxx}


def _frf_summary(frf: dict[str, np.ndarray] | None) -> dict[str, Any]:
    if frf is None:
        return {}
    f, h, coh, pxx = frf["f"], frf["h"], frf["coh"], frf["pxx"]
    mag = np.abs(h)
    phase = np.unwrap(np.angle(h))
    out: dict[str, Any] = {"df_hz": float(f[1] - f[0])}
    ok = (f >= 0.2) & (coh >= 0.5)
    below = np.flatnonzero(ok & (mag < 1.0 / np.sqrt(2.0)))
    out["bw_3db_hz"] = float(f[below[0]]) if below.size else None
    out["bw_lower_bound_hz"] = float(f[ok][-1]) if np.any(ok) and not below.size else None
    for fq in (1.0, 2.0, 3.0):
        j = int(np.argmin(np.abs(f - fq)))
        out[f"phase_delay_{fq:g}hz_ms"] = (
            float(-phase[j] / (2 * np.pi * f[j]) * 1e3) if f[j] > 0 else None
        )
        out[f"gain_{fq:g}hz"] = float(mag[j])
        out[f"coh_{fq:g}hz"] = float(coh[j])
    cum = np.cumsum(pxx[1:])
    if cum.size and cum[-1] > 0:
        cum = cum / cum[-1]
        out["cmd_f95_hz"] = float(f[1:][np.searchsorted(cum, 0.95)])
        out["cmd_f99_hz"] = float(f[1:][min(np.searchsorted(cum, 0.99), cum.size - 1)])
    return out


# ---------------------------------------------------------------- smooth


def sparc(speed: np.ndarray, fs: float, fc: float = 10.0, amp_th: float = 0.05, padlevel: int = 4) -> float:
    """Spectral arc length (Balasubramanian 2015); closer to 0 = smoother."""
    v = np.asarray(speed, dtype=np.float64).reshape(-1)
    if v.size < 4 or np.max(np.abs(v)) < 1e-9:
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(v.size)) + padlevel))
    f = np.arange(nfft) * fs / nfft
    m = np.abs(np.fft.fft(v, nfft))
    m = m / max(np.max(m), 1e-18)
    sel = f <= fc
    f_sel, m_sel = f[sel], m[sel]
    keep = np.flatnonzero(m_sel >= amp_th)
    if keep.size < 2:
        return float("nan")
    f_sel = f_sel[: keep[-1] + 1]
    m_sel = m_sel[: keep[-1] + 1]
    f_sel = f_sel / f_sel[-1]
    return float(-np.sum(np.sqrt(np.diff(f_sel) ** 2 + np.diff(m_sel) ** 2)))


# ---------------------------------------------------------------- task


def _grip_events(g: np.ndarray, t: np.ndarray, min_hold_s: float = 0.2) -> list[tuple[int, str]]:
    closed = g > GRIP_CLOSED
    edges = np.flatnonzero(closed[1:] != closed[:-1]) + 1
    out: list[tuple[int, str]] = []
    for k, i in enumerate(edges):
        nxt = edges[k + 1] if k + 1 < edges.size else closed.size
        if t[min(nxt, closed.size - 1)] - t[i] < min_hold_s:
            continue
        out.append((int(i), "close" if closed[i] else "open"))
    return out



def _phase_labels(
    g: np.ndarray,
    t: np.ndarray,
    *,
    window_s: float = CONTACT_WINDOW_S,
) -> np.ndarray:
    """Label each tick: seek / grasp / carry / place from right-gripper command.

    - seek: gripper open (找球 / 接近)
    - grasp: ±window around close edges (夹取接触窗)
    - carry: gripper held closed (夹→放中段)
    - place: ±window around open edges (放球接触窗)

    Windows override seek/carry. Overlapping grasp+place → nearer event wins.
    """
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    n = g.shape[0]
    closed = g > GRIP_CLOSED
    labels = np.where(closed, "carry", "seek").astype(object)
    # edge indices + kind
    edges: list[tuple[int, str]] = []
    for i in range(1, n):
        if closed[i] and not closed[i - 1]:
            edges.append((i, "grasp"))
        elif (not closed[i]) and closed[i - 1]:
            edges.append((i, "place"))
    if not edges:
        return labels
    # paint windows; resolve overlaps by nearest edge
    win = np.full(n, np.inf)
    kind = np.empty(n, dtype=object)
    kind[:] = None
    for i, k in edges:
        lo = int(np.searchsorted(t, t[i] - window_s, side="left"))
        hi = int(np.searchsorted(t, t[i] + window_s, side="right"))
        d = np.abs(t[lo:hi] - t[i])
        closer = d < win[lo:hi]
        if np.any(closer):
            sl = slice(lo, hi)
            idx = np.flatnonzero(closer) + lo
            win[idx] = d[closer]
            kind[idx] = k
    hit = kind != None  # noqa: E711
    labels[hit] = kind[hit]
    return labels


def joint_vel_limit_util(
    v_joints: np.ndarray,
    limits_rad_s: tuple[float, ...] | np.ndarray = G2_ARM_R_VEL_LIMIT_RAD_S,
) -> np.ndarray:
    """Per-tick max_i(|q̇_i| / vlim_i). >1 means command exceeds URDF limit on some joint."""
    v = np.asarray(v_joints, dtype=np.float64)
    lim = np.asarray(limits_rad_s, dtype=np.float64).reshape(1, -1)
    if v.ndim != 2 or v.shape[1] != lim.shape[1]:
        raise ValueError(f"v_joints shape {v.shape} vs limits {lim.shape[1]}")
    return np.max(np.abs(v) / np.maximum(lim, 1e-9), axis=1)


def _summarize_phase(
    phase: np.ndarray,
    *,
    mask: np.ndarray,
    lag_s: np.ndarray,
    contour: np.ndarray,
    err: np.ndarray,
    speed: np.ndarray,
    fast_speed: float,
    lag_scale: float,
    err_scale: float,
    vel_util: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-phase p50/p95 for lag/contour/err, plus a fast-speed subset."""
    out: dict[str, Any] = {"phases": list(PHASE_NAMES), "fast_speed": float(fast_speed)}
    for name in PHASE_NAMES:
        m = mask & (phase == name)
        mf = m & (speed >= fast_speed)
        block = {
            "n": int(np.count_nonzero(m)),
            "n_fast": int(np.count_nonzero(mf)),
            "frac": float(np.mean(m)) if mask.any() else None,
            "lag_ms": _p50_p95(lag_s[m], lag_scale),
            "contour": _p50_p95(contour[m], err_scale),
            "err": _p50_p95(err[m], err_scale),
            "speed_p50": _pct(speed[m], 50),
            "speed_p95": _pct(speed[m], 95),
            "fast": {
                "lag_ms": _p50_p95(lag_s[mf], lag_scale),
                "contour": _p50_p95(contour[mf], err_scale),
                "err": _p50_p95(err[mf], err_scale),
            },
        }
        if vel_util is not None:
            u50 = _pct(vel_util[m], 50)
            u95 = _pct(vel_util[m], 95)
            block["vel_limit_util_p50"] = u50
            block["vel_limit_util_p95"] = u95
            # Positive = still headroom to URDF; negative = command overshoots limit.
            block["vel_limit_gap_p95"] = None if u95 is None else float(1.0 - u95)
        out[name] = block
    return out


def _stop_settling(
    ref: np.ndarray, meas: np.ndarray, speed_ref: np.ndarray, valid: np.ndarray, dt: float
) -> dict[str, Any]:
    """Settle time / overshoot after the reference comes (nearly) to rest.

    Policy references rarely reach exact zero speed, so a stop is entered below
    0.10 rad/s and held while below 0.15 rad/s (hysteresis), after a peak above
    0.8 rad/s within the previous 0.6 s.
    """
    n = ref.shape[0]
    enter = speed_ref < 0.10
    still = speed_ref < 0.15
    run_n = max(1, int(round(0.10 / dt)))
    look = max(1, int(round(0.60 / dt)))
    win = int(round(1.0 / dt))
    settle_ms, overshoot, unsettled = [], [], 0
    i = look
    while i < n - run_n:
        if not (enter[i] and not enter[i - 1]):
            i += 1
            continue
        if np.max(speed_ref[i - look : i]) < 0.8 or not np.all(still[i : i + run_n]):
            i += 1
            continue
        if not np.all(valid[i - look : i + run_n]):
            i += 1
            continue
        end = i
        while end < min(n, i + win) and still[end]:
            end += 1
        seg_err = np.max(np.abs(ref[i:end] - meas[i:end]), axis=1)
        if seg_err.size < run_n:
            i = end
            continue
        tail = max(1, int(round(0.05 / dt)))
        eps = max(0.02, 1.5 * float(np.median(seg_err[-tail:])))
        ok = seg_err < eps
        hold = max(1, int(round(0.05 / dt)))
        settled = None
        for j in range(ok.size - hold + 1):
            if np.all(ok[j : j + hold]):
                settled = j
                break
        if settled is None:
            unsettled += 1
        else:
            settle_ms.append(settled * dt * 1e3)
        d = ref[i] - ref[max(0, i - look)]
        dn = float(np.linalg.norm(d))
        if dn > 1e-6:
            proj = (meas[i:end] - ref[i]) @ (d / dn)
            overshoot.append(max(0.0, float(np.max(proj))) * 1e3)
        i = end
    total = len(settle_ms) + unsettled
    return {
        "n_stops": int(total),
        "settle_ms": _p50_p95(np.asarray(settle_ms)),
        "frac_unsettled_1s": float(unsettled / total) if total else None,
        "overshoot_mrad": _p50_p95(np.asarray(overshoot)),
    }


# ---------------------------------------------------------------- main


def control_quality_metrics(
    ctl: dict[str, np.ndarray],
    *,
    urdf_hint: str | None = None,
    ee: bool = True,
) -> dict[str, Any]:
    """Compute the control-quality block for one ``control_trace``."""
    need = ("monotonic_s", "measured_state24", "reference21", "command21")
    if any(k not in ctl for k in need):
        return {}
    t = np.asarray(ctl["monotonic_s"], dtype=np.float64).reshape(-1)
    n = t.shape[0]
    if n < 200:
        return {"n": int(n)}
    dt = float(np.median(np.diff(t)))
    meas24 = np.asarray(ctl["measured_state24"], dtype=np.float64)
    ref21 = np.asarray(ctl["reference21"], dtype=np.float64)
    cmd21 = np.asarray(ctl["command21"], dtype=np.float64)

    step = _step_mask(ctl, n)
    handoff = _handoff_mask(ctl.get("plan_id"), t, HANDOFF_WINDOW_S)
    k_max = int(round(LAG_SEARCH_S / dt))

    qd24 = ctl.get("measured_velocity24")
    qd_ratio: list[float] = []

    pool: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "speed", "lag_cmd", "lag_ref", "bound_cmd", "contour_cmd", "contour_ref",
            "follow_ref", "normal_ref", "err_ref", "resid_ref", "gain", "handoff", "steady",
        )
    }
    xc: dict[str, list[np.ndarray]] = {"cmd_lag": [], "cmd_rho": [], "cmd_spd": [],
                                       "ref_lag": [], "ref_rho": [], "ref_spd": []}
    frf_in: dict[str, list[np.ndarray]] = {"cmd": [], "ref": [], "meas": []}
    acc_p95, jerk_p95, sparc_c, sparc_m = [], [], [], []
    settle_parts: list[dict[str, Any]] = []
    arm_energy: dict[str, float] = {}
    phase = _phase_labels(cmd21[:, CMD_GRIP_R], t)
    right_joint: dict[str, np.ndarray] | None = None

    for arm, (c_sl, m_sl) in ARMS.items():
        cmd = cmd21[:, c_sl]
        ref = ref21[:, c_sl]
        meas = meas24[:, m_sl]
        v_cmd = np.gradient(cmd, dt, axis=0)
        v_ref = np.gradient(ref, dt, axis=0)
        v_meas = _moving_average(np.gradient(meas, dt, axis=0), 5)
        if qd24 is not None:
            qd = np.asarray(qd24, dtype=np.float64)[:, m_sl]
            ok = step & np.all(np.isfinite(qd), axis=1)
            den = float(np.sum(np.abs(v_meas[ok])))
            if np.any(ok) and den > 1e-9:
                qd_ratio.append(float(np.sum(np.abs(qd[ok]))) / den)
        speed = np.max(np.abs(v_cmd), axis=1)
        speed_ref = np.max(np.abs(v_ref), axis=1)
        arm_energy[arm] = float(np.sum(speed[step] ** 2))
        moving = step & (speed >= SPEED_BINS_RAD_S[0])

        lag_c, dist_c, bnd_c = path_lag(cmd, meas, k_max)
        lag_r, dist_r, _ = path_lag(ref, meas, k_max)

        e = ref - meas
        vn = np.linalg.norm(v_ref, axis=1, keepdims=True)
        that = v_ref / np.maximum(vn, 1e-9)
        follow = np.einsum("ij,ij->i", e, that)
        normal = np.linalg.norm(e - follow[:, None] * that, axis=1)

        xr_c = _xcorr_windows(v_cmd, v_meas, speed, step, dt)
        xr_r = _xcorr_windows(v_ref, v_meas, speed, step, dt)
        for tag, xr in (("cmd", xr_c), ("ref", xr_r)):
            xc[f"{tag}_lag"].append(xr["lag_s"])
            xc[f"{tag}_rho"].append(xr["rho"])
            xc[f"{tag}_spd"].append(xr["speed"])

        good = xr_r["rho"] >= XCORR_MIN_RHO
        tau_star = int(round(np.median(xr_r["lag_s"][good]) / dt)) if np.any(good) else 0
        tau_star = max(0, tau_star)
        shifted_ref = np.roll(ref, tau_star, axis=0)
        resid = np.linalg.norm(shifted_ref - meas, axis=1)
        resid[:tau_star] = np.nan
        v_meas_fwd = np.roll(v_meas, -tau_star, axis=0)
        gain = np.linalg.norm(v_meas_fwd, axis=1) / np.maximum(np.linalg.norm(v_cmd, axis=1), 1e-9)
        if tau_star:
            gain[-tau_star:] = np.nan

        sel = moving
        pool["speed"].append(speed[sel])
        pool["lag_cmd"].append(lag_c[sel] * dt)
        pool["lag_ref"].append(lag_r[sel] * dt)
        pool["bound_cmd"].append(bnd_c[sel].astype(np.float64))
        pool["contour_cmd"].append(dist_c[sel])
        pool["contour_ref"].append(dist_r[sel])
        pool["follow_ref"].append(follow[sel])
        pool["normal_ref"].append(normal[sel])
        pool["err_ref"].append(np.linalg.norm(e[sel], axis=1))
        pool["resid_ref"].append(resid[sel])
        pool["gain"].append(gain[sel])
        pool["handoff"].append(np.linalg.norm(e[moving & handoff], axis=1))
        pool["steady"].append(np.linalg.norm(e[moving & ~handoff], axis=1))

        frf_in["cmd"].append(cmd[step])
        frf_in["ref"].append(ref[step])
        frf_in["meas"].append(meas[step])

        acc = np.gradient(v_cmd, dt, axis=0)
        jerk = np.gradient(acc, dt, axis=0)
        acc_p95.append(_pct(np.max(np.abs(acc[step]), axis=1), 95))
        jerk_p95.append(_pct(np.max(np.abs(jerk[step]), axis=1), 95))

        w = int(round(2.0 / dt))
        sp_c = np.linalg.norm(v_cmd, axis=1)
        sp_m = np.linalg.norm(v_meas, axis=1)
        for w0 in range(0, n - w, w // 2):
            sl = slice(w0, w0 + w)
            if np.mean(step[sl]) < 0.95 or np.mean(sp_c[sl]) < SPEED_BINS_RAD_S[0]:
                continue
            sparc_c.append(sparc(sp_c[sl], 1.0 / dt))
            sparc_m.append(sparc(sp_m[sl], 1.0 / dt))

        st = _stop_settling(ref, meas, speed_ref, step, dt)
        st["arm"] = arm
        settle_parts.append(st)

        if arm == "R":
            right_joint = {
                "lag_s": lag_r * dt,
                "contour": dist_r,
                "err": np.linalg.norm(e, axis=1),
                "speed": speed,
                "vel_util": joint_vel_limit_util(v_cmd, G2_ARM_R_VEL_LIMIT_RAD_S),
                "mask": step & (speed >= SPEED_BINS_RAD_S[0]),
            }

    cat = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in pool.items()}
    spd = cat["speed"]
    edges = SPEED_BINS_RAD_S
    out: dict[str, Any] = {
        "dt_s": dt,
        "n_moving_ticks": int(spd.size),
        "speed_bins_rad_s": _bin_labels(edges),
        "lag": {
            "path_cmd_ms": {**_p50_p95(cat["lag_cmd"], 1e3), "bins": _binned(cat["lag_cmd"], spd, edges, 1e3)},
            "path_ref_ms": {**_p50_p95(cat["lag_ref"], 1e3), "bins": _binned(cat["lag_ref"], spd, edges, 1e3)},
            "path_cmd_frac_at_bound": float(np.mean(cat["bound_cmd"])) if cat["bound_cmd"].size else None,
        },
        "error": {
            "contour_cmd_mrad": {**_p50_p95(cat["contour_cmd"], 1e3), "bins": _binned(cat["contour_cmd"], spd, edges, 1e3)},
            "contour_ref_mrad": {**_p50_p95(cat["contour_ref"], 1e3), "bins": _binned(cat["contour_ref"], spd, edges, 1e3)},
            "follow_ref_mrad": _p50_p95(cat["follow_ref"], 1e3),
            "normal_ref_mrad": _p50_p95(cat["normal_ref"], 1e3),
            "err_ref_l2_mrad": {**_p50_p95(cat["err_ref"], 1e3), "bins": _binned(cat["err_ref"], spd, edges, 1e3)},
            "aligned_resid_ref_mrad": _p50_p95(cat["resid_ref"], 1e3),
            "handoff_err_ref_l2_mrad": _p50_p95(cat["handoff"], 1e3),
            "steady_err_ref_l2_mrad": _p50_p95(cat["steady"], 1e3),
        },
        "gain": {"speed_gain": {**_p50_p95(cat["gain"]), "bins": _binned(cat["gain"], spd, edges)}},
    }

    xcat = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in xc.items()}
    xcorr: dict[str, Any] = {}
    for tag in ("cmd", "ref"):
        lag, rho, wsp = xcat[f"{tag}_lag"], xcat[f"{tag}_rho"], xcat[f"{tag}_spd"]
        good = rho >= XCORR_MIN_RHO
        xcorr[f"{tag}_ms"] = {
            **_p50_p95(lag[good], 1e3),
            "n_windows": int(lag.size),
            "frac_rho_ok": float(np.mean(good)) if lag.size else None,
            "rho_p50": _pct(rho, 50),
            "bins": _binned(lag[good], wsp[good], edges, 1e3),
        }
    out["lag"]["xcorr"] = xcorr

    main_arm = max(arm_energy, key=arm_energy.get) if arm_energy else "R"
    ai = list(ARMS).index(main_arm)
    freq: dict[str, Any] = {"arm": main_arm}
    for tag in ("cmd", "ref"):
        frf = _welch_frf(frf_in[tag][ai], frf_in["meas"][ai], dt, WELCH_NPERSEG)
        freq[tag] = _frf_summary(frf)
    out["freq"] = freq

    task: dict[str, Any] = {"settling": [s for s in settle_parts if s["n_stops"]]}
    if ee:
        task.update(_ee_metrics(t, meas24, ref21, cmd21, step, urdf_hint, dt, phase=phase))
    out["task"] = task

    if right_joint is not None:
        out["phases"] = {
            "definition": {
                "seek": "gripper open (找球/接近)",
                "grasp": f"±{CONTACT_WINDOW_S:.2f}s around close (夹取窗)",
                "carry": "gripper held closed (搬运)",
                "place": f"±{CONTACT_WINDOW_S:.2f}s around open (放球窗)",
                "fast_joint_rad_s": PHASE_FAST_JOINT_RAD_S,
                "fast_ee_mps": PHASE_FAST_EE_MPS,
                "arm": "R",
                "speed": "max_|q̇_cmd| over R-arm joints (rad/s)",
                "vel_limit_rad_s": list(G2_ARM_R_VEL_LIMIT_RAD_S),
                "vel_limit_util": "max_i(|q̇_i|/vlim_i); target≥0.8 on seek/carry",
                "vel_limit_util_target": VEL_LIMIT_UTIL_TARGET,
                "vel_limit_source": "G2_t2v2_crsB URDF idx61..67_arm_r_joint*",
            },
            "joint": _summarize_phase(
                phase,
                mask=right_joint["mask"],
                lag_s=right_joint["lag_s"],
                contour=right_joint["contour"],
                err=right_joint["err"],
                speed=right_joint["speed"],
                fast_speed=PHASE_FAST_JOINT_RAD_S,
                lag_scale=1e3,
                err_scale=1e3,
                vel_util=right_joint["vel_util"],
            ),
            "ee": task.get("phases_ee"),
            "frac": {
                name: float(np.mean(phase == name)) for name in PHASE_NAMES
            },
        }
        task.pop("phases_ee", None)

    trk = np.asarray(ctl.get("tracker_status", np.zeros(n)), dtype=np.int64)
    effort: dict[str, Any] = {
        "cmd_acc_p95_rad_s2": max([a for a in acc_p95 if a is not None], default=None),
        "cmd_jerk_p95_rad_s3": max([a for a in jerk_p95 if a is not None], default=None),
        "tracker_status_frac": {
            str(int(v)): float(np.mean(trk[step] == v)) for v in np.unique(trk[step])
        },
    }
    if "sat_excess_rad" in ctl:
        sat = np.asarray(ctl["sat_excess_rad"], dtype=np.float64)
        effort["sat_frac"] = float(np.mean(sat[step] > 1e-6))
        effort["sat_excess_p99_mrad"] = (_pct(sat[step], 99) or 0.0) * 1e3
    if "measured_effort24" in ctl:
        eff = np.abs(np.asarray(ctl["measured_effort24"], dtype=np.float64)[:, 8:22])
        if np.any(np.isfinite(eff)):
            effort["effort_p95_max_joint"] = _pct(np.nanmax(eff[step], axis=1), 95)
    out["effort"] = effort
    out["smooth"] = {
        "sparc_cmd_p50": _pct(np.asarray(sparc_c), 50),
        "sparc_meas_p50": _pct(np.asarray(sparc_m), 50),
        "n_windows": len(sparc_c),
    }
    out["timing"] = _timing_metrics(ctl, t, dt)
    out["measured_velocity_vs_diff"] = float(np.mean(qd_ratio)) if qd_ratio else None
    return out


def _ee_metrics(
    t: np.ndarray,
    meas24: np.ndarray,
    ref21: np.ndarray,
    cmd21: np.ndarray,
    step: np.ndarray,
    urdf_hint: str | None,
    dt: float,
    phase: np.ndarray | None = None,
) -> dict[str, Any]:
    from fastervla.deploy.analysis.ee_kinematics import load_urdf_fk

    loaded = load_urdf_fk(urdf_hint)
    if loaded is None:
        return {"ee_fk_available": False}
    fk, names = loaded
    from urdf_fk import fk_batch  # type: ignore

    links = tuple(EE_LINKS.values())
    xyz = {
        "meas": fk_batch(fk, names, meas24, links),
        "ref": fk_batch(fk, names, _policy24(meas24, ref21), links),
        "cmd": fk_batch(fk, names, _policy24(meas24, cmd21), links),
    }
    k_max = int(round(LAG_SEARCH_S / dt))
    edges = EE_SPEED_BINS_MPS
    lag_ref, contour_ref, err_ref, spd_pool = [], [], [], []
    # Right EE kept for phase metrics (task arm).
    right_ee: dict[str, np.ndarray] | None = None
    for link in links:
        p_ref = np.asarray(xyz["ref"][link]["xyz"], dtype=np.float64)
        p_meas = np.asarray(xyz["meas"][link]["xyz"], dtype=np.float64)
        spd = np.linalg.norm(np.gradient(p_ref, dt, axis=0), axis=1)
        sel = step & (spd >= edges[0])
        lag, dist, _ = path_lag(p_ref, p_meas, k_max)
        lag_ref.append(lag[sel] * dt)
        contour_ref.append(dist[sel])
        err_ref.append(np.linalg.norm(p_ref - p_meas, axis=1)[sel])
        spd_pool.append(spd[sel])
        if link == EE_LINKS["R"]:
            right_ee = {
                "lag_s": lag * dt,
                "contour": dist,
                "err": np.linalg.norm(p_ref - p_meas, axis=1),
                "speed": spd,
                "mask": step & (spd >= edges[0]),
            }
    lag_a = np.concatenate(lag_ref)
    con_a = np.concatenate(contour_ref)
    err_a = np.concatenate(err_ref)
    spd_a = np.concatenate(spd_pool)

    events = []
    g = cmd21[:, CMD_GRIP_R]
    link_r = EE_LINKS["R"]
    p_ref_r = np.asarray(xyz["ref"][link_r]["xyz"], dtype=np.float64)
    p_cmd_r = np.asarray(xyz["cmd"][link_r]["xyz"], dtype=np.float64)
    p_meas_r = np.asarray(xyz["meas"][link_r]["xyz"], dtype=np.float64)
    for i, kind in _grip_events(g, t):
        events.append(
            {
                "t_s": float(t[i] - t[0]),
                "kind": kind,
                "ee_err_ref_mm": float(np.linalg.norm(p_ref_r[i] - p_meas_r[i]) * 1e3),
                "ee_err_cmd_mm": float(np.linalg.norm(p_cmd_r[i] - p_meas_r[i]) * 1e3),
            }
        )

    def _ev(kind: str) -> dict[str, Any]:
        v = np.asarray([e["ee_err_ref_mm"] for e in events if e["kind"] == kind])
        return {**_p50_p95(v), "max": float(np.max(v)) if v.size else None}

    out_ee: dict[str, Any] = {
        "ee_fk_available": True,
        "ee_speed_bins_mps": _bin_labels(edges),
        "ee_path_lag_ref_ms": {**_p50_p95(lag_a, 1e3), "bins": _binned(lag_a, spd_a, edges, 1e3)},
        "ee_contour_ref_mm": {**_p50_p95(con_a, 1e3), "bins": _binned(con_a, spd_a, edges, 1e3)},
        "ee_err_ref_mm": {**_p50_p95(err_a, 1e3), "bins": _binned(err_a, spd_a, edges, 1e3)},
        "grip_event_err_mm": {"close": _ev("close"), "open": _ev("open")},
        "grip_events": events,
    }
    if phase is not None and right_ee is not None:
        out_ee["phases_ee"] = _summarize_phase(
            phase,
            mask=right_ee["mask"],
            lag_s=right_ee["lag_s"],
            contour=right_ee["contour"],
            err=right_ee["err"],
            speed=right_ee["speed"],
            fast_speed=PHASE_FAST_EE_MPS,
            lag_scale=1e3,
            err_scale=1e3,
        )
    return out_ee


def _timing_metrics(ctl: dict[str, np.ndarray], t: np.ndarray, dt: float) -> dict[str, Any]:
    period = np.diff(t) * 1e3
    out: dict[str, Any] = {
        "period_p50_ms": _pct(period, 50),
        "period_p99_ms": _pct(period, 99),
        "deadline_miss_frac": float(np.mean(period > 1.5 * dt * 1e3)) if period.size else None,
    }
    if "servo_tick_ms" in ctl:
        st = np.asarray(ctl["servo_tick_ms"], dtype=np.float64)
        out["tick_compute_p50_ms"] = _pct(st, 50)
        out["tick_compute_p99_ms"] = _pct(st, 99)
    for key in ("state_read_ms", "send_ms", "tracker_solve_ms"):
        if key in ctl:
            v = np.asarray(ctl[key], dtype=np.float64)
            out[f"{key[:-3]}_p50_ms"] = _pct(v, 50)
            out[f"{key[:-3]}_p99_ms"] = _pct(v, 99)
    if "state_stamp_ms" in ctl and "state_wall_ms" in ctl:
        stamp = np.asarray(ctl["state_stamp_ms"], dtype=np.float64)
        wall = np.asarray(ctl["state_wall_ms"], dtype=np.float64)
        age = wall - stamp
        plausible = (stamp > 1e12) & (np.abs(age) < 5_000)
        if np.any(plausible):
            out["state_age_p50_ms"] = _pct(age[plausible], 50)
            out["state_age_p99_ms"] = _pct(age[plausible], 99)
        st = stamp[stamp > 0]
        if st.size > 2:
            ds = np.diff(st)
            out["state_stamp_stale_frac"] = float(np.mean(ds <= 0))
    return out


def control_headline(cq: dict[str, Any]) -> dict[str, Any]:
    """Flat key numbers for dashboards / daily tables."""
    if not cq or "lag" not in cq:
        return {}
    lag, err, gain = cq["lag"], cq["error"], cq["gain"]
    task = cq.get("task", {})
    freq = cq.get("freq", {})
    out: dict[str, Any] = {
        "lag_cmd_p50_ms": lag["path_cmd_ms"]["p50"],
        "lag_cmd_p95_ms": lag["path_cmd_ms"]["p95"],
        "lag_ref_p50_ms": lag["path_ref_ms"]["p50"],
        "lag_ref_p95_ms": lag["path_ref_ms"]["p95"],
        "xcorr_cmd_p50_ms": lag["xcorr"]["cmd_ms"]["p50"],
        "xcorr_ref_p50_ms": lag["xcorr"]["ref_ms"]["p50"],
        "xcorr_rho_p50": lag["xcorr"]["ref_ms"]["rho_p50"],
        "contour_ref_p95_mrad": err["contour_ref_mrad"]["p95"],
        "follow_ref_p95_mrad": err["follow_ref_mrad"]["p95"],
        "aligned_resid_p95_mrad": err["aligned_resid_ref_mrad"]["p95"],
        "speed_gain_p50": gain["speed_gain"]["p50"],
        "bw_cmd_hz": freq.get("cmd", {}).get("bw_3db_hz"),
        "phase_delay_ref_1hz_ms": freq.get("ref", {}).get("phase_delay_1hz_ms"),
        "cmd_f95_hz": freq.get("cmd", {}).get("cmd_f95_hz"),
    }
    if task.get("ee_fk_available"):
        out["ee_lag_ref_p50_ms"] = task["ee_path_lag_ref_ms"]["p50"]
        out["ee_contour_ref_p95_mm"] = task["ee_contour_ref_mm"]["p95"]
        out["grip_close_err_p50_mm"] = task["grip_event_err_mm"]["close"]["p50"]
    phases = cq.get("phases") or {}
    for name in PHASE_NAMES:
        j = (phases.get("joint") or {}).get(name) or {}
        e = (phases.get("ee") or {}).get(name) or {}
        out[f"phase_{name}_lag_p50_ms"] = (j.get("lag_ms") or {}).get("p50")
        out[f"phase_{name}_contour_p95_mrad"] = (j.get("contour") or {}).get("p95")
        out[f"phase_{name}_speed_p95"] = j.get("speed_p95")
        out[f"phase_{name}_vel_util_p50"] = j.get("vel_limit_util_p50")
        out[f"phase_{name}_vel_util_p95"] = j.get("vel_limit_util_p95")
        out[f"phase_{name}_vel_gap_p50"] = (
            None if j.get("vel_limit_util_p50") is None else float(1.0 - j["vel_limit_util_p50"])
        )
        out[f"phase_{name}_vel_gap_p95"] = j.get("vel_limit_gap_p95")
        out[f"phase_{name}_ee_lag_p50_ms"] = (e.get("lag_ms") or {}).get("p50")
        out[f"phase_{name}_ee_contour_p95_mm"] = (e.get("contour") or {}).get("p95")
        out[f"phase_{name}_ee_err_p95_mm"] = (e.get("err") or {}).get("p95")
        fj = j.get("fast") or {}
        fe = e.get("fast") or {}
        out[f"phase_{name}_fast_lag_p50_ms"] = (fj.get("lag_ms") or {}).get("p50")
        out[f"phase_{name}_fast_ee_err_p95_mm"] = (fe.get("err") or {}).get("p95")
    out["vel_limit_util_target"] = VEL_LIMIT_UTIL_TARGET
    out["vel_limit_rad_s"] = list(G2_ARM_R_VEL_LIMIT_RAD_S)
    for label, b in lag["path_ref_ms"]["bins"].items():
        out[f"lag_ref_p50_ms@{label}"] = b["p50"]
        out[f"lag_ref_p95_ms@{label}"] = b["p95"]
    return out
