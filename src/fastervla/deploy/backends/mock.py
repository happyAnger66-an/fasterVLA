"""Deterministic mock policy backend for CPU-only deploy tests."""

from __future__ import annotations

import numpy as np

from fastervla.core.types import Chunk, Observation, Prefix

_SUPPORTED_RTC = {"NONE", "INPAINT"}


class MockBackend:
    """Sine / seeded fake chunks; F0 supports ``rtc_mode=NONE`` (and INPAINT no-op)."""

    def __init__(
        self,
        *,
        horizon: int = 30,
        action_dim: int = 24,
        seed: int = 0,
        mode: str = "sine",
    ) -> None:
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.seed = int(seed)
        self.mode = str(mode)
        self._warm = False
        self._call = 0

    def warmup(self) -> None:
        self._warm = True

    def sample(
        self,
        *,
        obs: Observation,
        prefix: Prefix | None,
        rtc_mode: str,
        n: int,
    ) -> list[Chunk]:
        mode = str(rtc_mode).upper()
        if mode not in _SUPPORTED_RTC:
            raise ValueError(f"MockBackend unsupported rtc_mode={rtc_mode!r}")
        if mode == "INPAINT" and prefix is None:
            # Allowed: still return full horizon for F0.
            pass

        out: list[Chunk] = []
        for i in range(int(n)):
            waypoints = self._make_waypoints(obs, sample_i=i)
            if prefix is not None and prefix.actions.shape[0] > 0:
                # Drop prefix length so caller gets the effective suffix (optional F0 path).
                # Keep full horizon when preferred — tests accept either 26 or 30 for prefix=4.
                # We keep full horizon for simpler sync loop; shape stays [H, D].
                pass
            out.append(
                Chunk(
                    waypoints=waypoints,
                    model_dt=1.0 / 30.0,
                    obs_ts=float(obs.image_ts),
                    meta={"mock": True, "call": self._call},
                )
            )
            self._call += 1
        return out

    def _make_waypoints(self, obs: Observation, *, sample_i: int) -> np.ndarray:
        h, d = self.horizon, self.action_dim
        rng = np.random.default_rng(self.seed + sample_i)
        t = np.arange(h, dtype=np.float64)[:, None]
        if self.mode == "sine":
            phase = float(np.sum(obs.state.astype(np.float64)) * 0.01)
            base = 0.05 * np.sin(2.0 * np.pi * (t / max(h, 1) + phase + self.seed * 0.001))
            waypoints = np.broadcast_to(base, (h, d)).astype(np.float32).copy()
            waypoints += 0.001 * rng.normal(size=(h, d)).astype(np.float32)
        else:
            waypoints = rng.normal(scale=0.02, size=(h, d)).astype(np.float32)
        # Anchor first step near current state for smoother sync tracking.
        state = np.asarray(obs.state, dtype=np.float32).reshape(-1)
        if state.shape[0] == d:
            waypoints[0] = state
            for k in range(1, h):
                waypoints[k] = waypoints[0] + 0.01 * k * np.sin(
                    np.linspace(0, np.pi, d, dtype=np.float32) + self.seed
                )
        return waypoints
