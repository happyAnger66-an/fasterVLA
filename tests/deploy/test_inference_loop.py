"""模块: fastervla.deploy.client.inference_loop
功能: L1 异步推理；估 d̂；RTC INPAINT；与 PlanBuffer merge；T_infer_min。

F2 测试（先测后码）。用 DelayedBackend + MockRobotPlatform，不连真机。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from fastervla.core.types import Chunk, Observation, Prefix
from fastervla.deploy.backends.mock import MockBackend
from fastervla.deploy.client.inference_loop import InferenceLoop, InferenceLoopConfig
from fastervla.deploy.client.io.mock_sim import MockRobotPlatform
from fastervla.deploy.client.plan_buffer import PlanBuffer
from fastervla.deploy.postprocess.retime import UniformRetimer


class DelayedBackend:
    """Wrap MockBackend with a fixed wall-clock delay on sample (simulates r2r)."""

    def __init__(self, inner: MockBackend, delay_s: float) -> None:
        self.inner = inner
        self.delay_s = float(delay_s)
        self.infer_calls = 0
        self.last_rtc: dict[str, Any] | None = None

    def warmup(self) -> None:
        self.inner.warmup()

    def sample(
        self,
        *,
        obs: Observation,
        prefix: Prefix | None,
        rtc_mode: str,
        n: int,
        rtc: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        self.infer_calls += 1
        self.last_rtc = None if rtc is None else dict(rtc)
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        # MockBackend ignores rtc payload; INPAINT is accepted as a mode flag.
        return self.inner.sample(
            obs=obs, prefix=prefix, rtc_mode=rtc_mode, n=n
        )


def _platform() -> MockRobotPlatform:
    return MockRobotPlatform(
        state_dim=24,
        command_dim=21,
        camera_names=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        plant_tau_s=0.057,
        plant_delay_s=0.036,
        seed=0,
    )


def _loop(
    *,
    delay_s: float,
    kappa: float = 1.0,
    duration_s: float = 2.0,
    t_infer_min_s: float = 0.0,
    t_switch_min_s: float = 0.0,
    cold_start_latency_ms: float = 100.0,
) -> tuple[InferenceLoop, DelayedBackend, MockRobotPlatform]:
    plat = _platform()
    inner = MockBackend(horizon=30, action_dim=24, seed=0, mode="sine")
    backend = DelayedBackend(inner, delay_s=delay_s)
    backend.warmup()
    buf = PlanBuffer()
    # Seed an initial plan so control never starts empty.
    seed_chunks = inner.sample(
        obs=Observation(
            images={"base_0_rgb": b"x"},
            image_ts=0.0,
            state=np.zeros(24, dtype=np.float32),
            state_ts=0.0,
        ),
        prefix=None,
        rtc_mode="NONE",
        n=1,
    )
    timed = UniformRetimer(policy_fps=30.0, kappa=kappa).retime(seed_chunks[0])
    buf.load(timed, origin_monotonic=0.0)

    cfg = InferenceLoopConfig(
        control_hz=100.0,
        policy_fps=30.0,
        kappa=kappa,
        kappa_max=1.5,
        kappa_min=1.0,
        t_infer_min_s=t_infer_min_s,
        t_switch_min_s=t_switch_min_s,
        latency_margin_ms=10.0,
        cold_start_latency_ms=cold_start_latency_ms,
        duration_s=duration_s,
        rtc_mode="INPAINT",
    )
    loop = InferenceLoop(
        platform=plat,
        backend=backend,
        plan_buffer=buf,
        retimer=UniformRetimer(policy_fps=30.0, kappa=kappa),
        config=cfg,
    )
    return loop, backend, plat


# 用例: 配置 kappa 超上限拒绝
# 参数: 无
def test_inference_loop_config_rejects_kappa_above_max() -> None:
    with pytest.raises((ValueError, TypeError)):
        InferenceLoopConfig(kappa=1.6, kappa_max=1.5, kappa_min=1.0)


# 用例: κ=1.0、延迟 60 ms，跑若干秒：无空缓冲；至少 1 次推理；RTC 载荷含 inference_delay
# 参数: 无
def test_inference_loop_no_starvation_delay_60ms_kappa_1() -> None:
    loop, backend, plat = _loop(delay_s=0.06, kappa=1.0, duration_s=1.5)
    summary = loop.run()
    assert summary.infer_calls >= 1
    assert backend.infer_calls >= 1
    assert summary.empty_buffer_ticks == 0
    assert summary.control_steps > 0
    # INPAINT path should attach rtc dict at least once.
    assert backend.last_rtc is not None
    assert "inference_delay" in backend.last_rtc
    assert "prev_chunk_left_over" in backend.last_rtc
    assert "execution_horizon" in backend.last_rtc
    plat.close()


# 用例: κ=1.0、延迟 120 ms，仍无空缓冲
# 参数: 无
def test_inference_loop_no_starvation_delay_120ms_kappa_1() -> None:
    loop, backend, plat = _loop(delay_s=0.12, kappa=1.0, duration_s=2.0)
    summary = loop.run()
    assert summary.empty_buffer_ticks == 0
    assert summary.infer_calls >= 1
    plat.close()


# 用例: κ=1.4、延迟 60 ms，Uniform 时间轴下无空缓冲
# 参数: 无
def test_inference_loop_no_starvation_delay_60ms_kappa_1p4() -> None:
    loop, backend, plat = _loop(delay_s=0.06, kappa=1.4, duration_s=1.5)
    summary = loop.run()
    assert summary.empty_buffer_ticks == 0
    assert summary.infer_calls >= 1
    plat.close()


# 用例: 结果早到（delay≈0）也不出现停顿 / 早到闲置计数为 0
# 参数: 无
def test_inference_loop_early_arrival_no_idle() -> None:
    loop, backend, plat = _loop(
        delay_s=0.0,
        kappa=1.0,
        duration_s=1.0,
        t_infer_min_s=0.0,
        t_switch_min_s=0.0,
    )
    summary = loop.run()
    assert summary.empty_buffer_ticks == 0
    # early_idle_s: wall time spent waiting for a fixed handoff that already arrived.
    assert float(summary.early_idle_s) == pytest.approx(0.0, abs=1e-3)
    assert summary.infer_calls >= 1
    plat.close()


# 用例: T_infer_min 强制两次推理间隔 ≥ 配置值
# 参数: 无
def test_inference_loop_respects_t_infer_min() -> None:
    loop, backend, plat = _loop(
        delay_s=0.0,
        kappa=1.0,
        duration_s=0.9,
        t_infer_min_s=0.25,
        cold_start_latency_ms=20.0,
    )
    t0 = time.monotonic()
    summary = loop.run()
    elapsed = time.monotonic() - t0
    assert summary.infer_calls >= 2
    # With T_infer_min=0.25 and duration=0.9, infer count should be bounded.
    assert summary.infer_calls <= 1 + int(elapsed / 0.25) + 2
    if len(summary.infer_intervals_s) >= 1:
        assert min(summary.infer_intervals_s) >= 0.25 - 0.05
    plat.close()


# 用例: d̂ 超过剩余计划 → starved 事件至少记 1 次（构造极短剩余）
# 参数: 无
def test_inference_loop_records_starved_when_d_hat_exceeds_remaining() -> None:
    plat = _platform()
    inner = MockBackend(horizon=30, action_dim=24, seed=1, mode="sine")
    backend = DelayedBackend(inner, delay_s=0.0)
    backend.warmup()
    buf = PlanBuffer()
    # Tiny remaining plan: load then advance almost to end.
    seed = inner.sample(
        obs=Observation(
            images={"base_0_rgb": b"x"},
            image_ts=0.0,
            state=np.zeros(24, dtype=np.float32),
            state_ts=0.0,
        ),
        prefix=None,
        rtc_mode="NONE",
        n=1,
    )[0]
    timed = UniformRetimer(policy_fps=30.0, kappa=1.0).retime(seed)
    buf.load(timed, origin_monotonic=0.0)
    # Leave only ~1 model step.
    buf.advance(29.0 / 30.0, alpha=1.0)

    cfg = InferenceLoopConfig(
        control_hz=100.0,
        policy_fps=30.0,
        kappa=1.0,
        duration_s=0.4,
        cold_start_latency_ms=200.0,  # large L̂ → large d̂
        latency_margin_ms=50.0,
        t_infer_min_s=0.3,
        rtc_mode="INPAINT",
    )
    loop = InferenceLoop(
        platform=plat,
        backend=backend,
        plan_buffer=buf,
        retimer=UniformRetimer(policy_fps=30.0, kappa=1.0),
        config=cfg,
    )
    summary = loop.run()
    assert summary.starved_events >= 1
    plat.close()
