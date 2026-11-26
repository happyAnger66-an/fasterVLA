"""模块: fastervla.deploy.client.recorder
功能: TraceRecorder — A/B 双缓冲；flush 中可继续 record；字段对齐老栈 robot-live。

见 docs/trace-design.md。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from fastervla.deploy.client.recorder import (
    STAGE_BUFFER_EMPTY_HOLD,
    STAGE_NAMES,
    STAGE_STEP,
    PeriodicFlusher,
    TraceRecorder,
    make_run_dir,
)


def _meas(i: float = 0.0) -> np.ndarray:
    return np.full(24, i, dtype=np.float64)


def _cmd(i: float = 0.0) -> np.ndarray:
    return np.full(21, i, dtype=np.float64)


def _actions(h: int = 30, base: float = 0.0) -> np.ndarray:
    t = np.arange(h, dtype=np.float32)[:, None]
    j = np.arange(24, dtype=np.float32)[None, :]
    return (base + 0.01 * t + 0.001 * j).astype(np.float32)


def _tick(rec: TraceRecorder, i: float, **kw: object) -> None:
    rec.record_tick(
        monotonic_s=float(i),
        measured24=_meas(i),
        reference21=_cmd(i),
        command21=_cmd(i),
        progress=float(i) * 0.01,
        plan_id=1,
        stage=STAGE_STEP,
        **kw,  # type: ignore[arg-type]
    )


# 用例: record_tick 不满容量；flush 写出 control_trace.npz，字段名对齐老栈
def test_recorder_flush_control_trace_fields(tmp_path: Path) -> None:
    rec = TraceRecorder(tmp_path / "robot-live-test", capacity_ticks=16, horizon=30)
    t0 = time.monotonic()
    for i in range(5):
        rec.record_tick(
            monotonic_s=t0 + 0.01 * i,
            measured24=_meas(i),
            reference21=_cmd(i + 0.1),
            command21=_cmd(i + 0.2),
            progress=float(i) * 0.1,
            plan_id=1,
            stage=STAGE_STEP,
            action_frame=i,
            accepted_prediction=0,
            servo_tick_ms=1.5,
        )
    out = rec.flush()
    assert out == tmp_path / "robot-live-test"
    ctl_path = out / "control_trace.npz"
    assert ctl_path.is_file()
    ctl = np.load(ctl_path, allow_pickle=True)
    for key in (
        "monotonic_s",
        "measured_state24",
        "reference21",
        "command21",
        "stage",
        "stage_names",
        "progress",
        "plan_id",
    ):
        assert key in ctl.files, key
    assert ctl["monotonic_s"].shape == (5,)
    assert ctl["measured_state24"].shape == (5, 24)
    assert ctl["reference21"].shape == (5, 21)
    assert ctl["command21"].shape == (5, 21)
    assert ctl["stage"].shape == (5,)
    assert list(ctl["stage_names"]) == list(STAGE_NAMES)
    assert int(ctl["stage"][0]) == STAGE_STEP
    assert rec.total_ticks == 5
    assert rec.drops == 0


# 用例: record_chunk + flush → prediction_trace.npz，含 r2r 时间戳与 rtc / d_hat
def test_recorder_flush_prediction_trace_fields(tmp_path: Path) -> None:
    rec = TraceRecorder(tmp_path / "robot-live-pred", capacity_ticks=8, horizon=30)
    acts = _actions(base=1.0)
    t_req, t_ready = 10.0, 10.085
    rec.record_chunk(
        actions=acts,
        actions_raw=acts + 0.01,
        requested_monotonic_s=t_req,
        ready_monotonic_s=t_ready,
        observation_monotonic_s=t_req - 0.02,
        accepted_monotonic_s=t_ready,
        action_origin_monotonic_s=t_ready,
        handoff_at_monotonic_s=t_ready + 0.2,
        d_hat=4,
        d_actual=3.2,
        anchor_step=2,
        plan_id=1,
        rtc_applied=True,
        rtc_d=4,
        rtc_s=5,
        rtc_guided_gap=0.01,
        observation_age_ms=25.0,
        retime_status="uniform",
        step_dt=np.full(30, 1.0 / 30.0),
    )
    out = rec.flush()
    pred = np.load(out / "prediction_trace.npz", allow_pickle=True)
    assert pred["actions"].shape == (1, 30, 24)
    assert pred["actions_raw"].shape == (1, 30, 24)
    assert pred["requested_monotonic_s"].shape == (1,)
    assert pred["ready_monotonic_s"][0] == pytest.approx(t_ready)
    r2r_ms = (pred["ready_monotonic_s"][0] - pred["requested_monotonic_s"][0]) * 1000.0
    assert r2r_ms == pytest.approx(85.0, abs=0.1)
    assert bool(pred["rtc_applied"][0]) is True
    assert int(pred["rtc_d"][0]) == 4
    assert "d_hat" in pred.files
    assert int(pred["d_hat"][0]) == 4
    assert float(pred["d_actual"][0]) == pytest.approx(3.2)


# 用例: 禁止 sealed 排队时，单 bank 满则 drops（不抛错、不阻塞）
def test_recorder_tick_capacity_drops(tmp_path: Path) -> None:
    rec = TraceRecorder(
        tmp_path / "robot-live-drop",
        capacity_ticks=3,
        horizon=30,
        max_sealed_segments=0,  # 不能自动轮转 → 满则丢
    )
    for i in range(7):
        _tick(rec, i)
    assert rec.total_ticks == 3
    assert rec.drops == 4
    rec.flush()
    ctl = np.load(tmp_path / "robot-live-drop" / "control_trace.npz")
    assert ctl["monotonic_s"].shape == (3,)


# 用例: 默认 A/B 轮转 — 超过单 bank 容量仍可不丢（sealed 未饱和）
def test_recorder_ab_rotate_avoids_drops(tmp_path: Path) -> None:
    rec = TraceRecorder(
        tmp_path / "robot-live-ab",
        capacity_ticks=3,
        horizon=30,
        max_sealed_segments=8,
    )
    for i in range(7):
        _tick(rec, i)
    assert rec.drops == 0
    assert rec.total_ticks == 7
    rec.close()
    ctl = np.load(tmp_path / "robot-live-ab" / "control_trace.npz")
    assert ctl["monotonic_s"].shape == (7,)


# 用例: flush 过程中另一线程可继续 record；最终数据包含 flush 前后的拍
def test_recorder_record_during_flush(tmp_path: Path) -> None:
    rec = TraceRecorder(tmp_path / "robot-live-during", capacity_ticks=64, horizon=30)
    for i in range(10):
        _tick(rec, i)

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    mid_ticks = {"n": 0}

    def flusher() -> None:
        try:
            barrier.wait(timeout=2.0)
            # Slow disk path simulated by large-ish flush + sleep inside is hard;
            # just flush while the other thread keeps writing.
            time.sleep(0.01)
            rec.flush()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def writer() -> None:
        try:
            barrier.wait(timeout=2.0)
            for i in range(10, 40):
                _tick(rec, i)
                mid_ticks["n"] += 1
                time.sleep(0.001)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    th1 = threading.Thread(target=flusher)
    th2 = threading.Thread(target=writer)
    th1.start()
    th2.start()
    th1.join(timeout=5.0)
    th2.join(timeout=5.0)
    assert not errors
    assert mid_ticks["n"] == 30
    rec.close()
    ctl = np.load(tmp_path / "robot-live-during" / "control_trace.npz")
    # 10 before + 30 during/after >= 40 (flush may have written partial then append)
    assert ctl["monotonic_s"].shape[0] == 40
    assert rec.drops == 0
    assert rec.total_ticks == 40


# 用例: record_tick 热路径不做磁盘 I/O（目录在 flush 前可以不存在完整 npz）
def test_recorder_hot_path_no_npz_until_flush(tmp_path: Path) -> None:
    out = tmp_path / "robot-live-hot"
    rec = TraceRecorder(out, capacity_ticks=32, horizon=30)
    _tick(rec, 1.0)
    assert not (out / "control_trace.npz").exists()
    rec.flush()
    assert (out / "control_trace.npz").exists()


# 用例: report.json 含 buffer_empty / starved / drops 计数
def test_recorder_report_json_counters(tmp_path: Path) -> None:
    rec = TraceRecorder(tmp_path / "robot-live-rep", capacity_ticks=16, horizon=30)
    rec.record_tick(
        monotonic_s=0.0,
        measured24=_meas(),
        reference21=_cmd(),
        command21=_cmd(),
        progress=0.0,
        plan_id=0,
        stage=STAGE_BUFFER_EMPTY_HOLD,
    )
    rec.bump_counter("buffer_empty_cycles", 3)
    rec.bump_counter("starved_events", 2)
    tiny = TraceRecorder(
        tmp_path / "robot-live-rep2",
        capacity_ticks=1,
        horizon=30,
        max_sealed_segments=0,
    )
    _tick(tiny, 0.0)
    _tick(tiny, 0.1)
    tiny.flush()
    report = json.loads((tmp_path / "robot-live-rep2" / "robot_bridge_report.json").read_text())
    assert report["trace_drops"] == 1
    assert report["control_trace_samples"] == 1

    rec.flush()
    report2 = json.loads((tmp_path / "robot-live-rep" / "robot_bridge_report.json").read_text())
    assert report2["buffer_empty_cycles"] == 3
    assert report2["starved_events"] == 2
    assert report2.get("double_buffer") is True


# 用例: 多线程 record_tick + record_chunk 不丢一致性（无死锁）；flush 后两者都有数据
def test_recorder_concurrent_tick_and_chunk(tmp_path: Path) -> None:
    rec = TraceRecorder(tmp_path / "robot-live-conc", capacity_ticks=500, horizon=30)
    stop = threading.Event()
    errors: list[BaseException] = []

    def ticks() -> None:
        try:
            i = 0
            while not stop.is_set() and i < 200:
                _tick(rec, i * 0.01)
                i += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def chunks() -> None:
        try:
            for j in range(5):
                rec.record_chunk(
                    actions=_actions(base=float(j)),
                    actions_raw=_actions(base=float(j)),
                    requested_monotonic_s=float(j),
                    ready_monotonic_s=float(j) + 0.05,
                    observation_monotonic_s=float(j),
                    accepted_monotonic_s=float(j) + 0.05,
                    d_hat=2,
                    d_actual=1.0,
                    plan_id=1,
                    rtc_applied=False,
                    rtc_d=2,
                )
                time.sleep(0.001)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    th1 = threading.Thread(target=ticks)
    th2 = threading.Thread(target=chunks)
    th1.start()
    th2.start()
    th2.join(timeout=2.0)
    stop.set()
    th1.join(timeout=2.0)
    assert not errors
    out = rec.close()
    ctl = np.load(out / "control_trace.npz")
    pred = np.load(out / "prediction_trace.npz")
    assert ctl["monotonic_s"].shape[0] >= 50
    assert pred["actions"].shape[0] == 5


# 用例: 空 recorder flush 仍写出空数组文件 + report（不崩）
def test_recorder_flush_empty(tmp_path: Path) -> None:
    rec = TraceRecorder(tmp_path / "robot-live-empty", capacity_ticks=8, horizon=30)
    out = rec.flush()
    ctl = np.load(out / "control_trace.npz")
    assert ctl["monotonic_s"].shape == (0,)
    pred = np.load(out / "prediction_trace.npz")
    assert pred["actions"].shape == (0, 30, 24)
    assert (out / "robot_bridge_report.json").is_file()


# 用例: stage 枚举与老栈一致（extract_jitter 依赖名字）
def test_stage_names_match_legacy() -> None:
    assert list(STAGE_NAMES) == ["step", "initial_hold", "buffer_empty_hold"]
    assert STAGE_STEP == 0
    assert STAGE_BUFFER_EMPTY_HOLD == 2


# 用例: make_run_dir 创建 robot-live-<id> 目录
def test_make_run_dir(tmp_path: Path) -> None:
    d = make_run_dir(tmp_path)
    assert d.is_dir()
    assert d.name.startswith("robot-live-")
    assert d.parent == tmp_path


def test_make_run_dir_with_tag(tmp_path: Path) -> None:
    from fastervla.deploy.client.recorder import sanitize_run_label

    d = make_run_dir(tmp_path, tag="R2 lead2")
    assert d.name.endswith("-R2_lead2")
    assert sanitize_run_label("R2 lead2") == "R2_lead2"


def test_make_run_dir_with_run_id(tmp_path: Path) -> None:
    d = make_run_dir(tmp_path, run_id="20261125-R0-baseline")
    assert d.name == "robot-live-20261125-R0-baseline"
    with pytest.raises(FileExistsError):
        make_run_dir(tmp_path, run_id="20261125-R0-baseline")


# 用例: PeriodicFlusher 周期性 flush，期间仍可 record；close 后数据齐全
def test_periodic_flusher(tmp_path: Path) -> None:
    run = make_run_dir(tmp_path)
    rec = TraceRecorder(run, capacity_ticks=256, horizon=30)
    flusher = PeriodicFlusher(rec, every_s=0.05)
    flusher.start()
    for i in range(20):
        _tick(rec, i)
        time.sleep(0.01)
    flusher.stop()
    assert flusher.flush_count >= 1
    rec.close()
    ctl = np.load(run / "control_trace.npz")
    assert ctl["monotonic_s"].shape[0] == 20
    assert (run / "robot_bridge_report.json").is_file()


# 用例: 富字段 chunk — previous_command_difference / request_to_ready_ms / report infer_records
def test_recorder_enriched_chunk_and_report(tmp_path: Path) -> None:
    run = make_run_dir(tmp_path)
    rec = TraceRecorder(run, capacity_ticks=32, horizon=30)
    rec.set_experiment(tag="unit", kappa=1.0)
    acts = _actions(base=2.0)
    rec.record_chunk(
        actions=acts,
        actions_raw=acts,
        requested_monotonic_s=1.0,
        ready_monotonic_s=1.08,
        observation_monotonic_s=0.99,
        accepted_monotonic_s=1.09,
        d_hat=3,
        d_actual=2.5,
        plan_id=2,
        rtc_applied=True,
        rtc_d=3,
        observation_age_ms=12.0,
        image_age_ms=15.0,
        previous_command_difference=0.042,
        request_to_ready_ms=80.0,
        client_prep_ms=5.0,
        infer_call_ms=70.0,
        overlap_gap=0.01,
        rtc_guided_gap=0.01,
        merge_accepted=True,
        step_dt=np.full(30, 1.0 / 30.0),
        frame_times_s=np.linspace(0, 1, 31),
    )
    rec.note_infer_record({"packet_id": 1, "accepted": True, "request_to_ready_ms": 80.0})
    rec.note_tracking_error(0.02)
    rec.note_tracking_error(0.15)  # warning
    rec.close()
    pred = np.load(run / "prediction_trace.npz", allow_pickle=True)
    assert float(pred["previous_command_difference"][0]) == pytest.approx(0.042)
    assert float(pred["request_to_ready_ms"][0]) == pytest.approx(80.0)
    assert float(pred["observation_age_ms"][0]) == pytest.approx(12.0)
    assert pred["frame_times_s"].shape == (1, 31)
    report = json.loads((run / "robot_bridge_report.json").read_text())
    assert report["experiment"]["tag"] == "unit"
    assert len(report["inference_records"]) == 1
    assert report["tracking_warning_cycles"] >= 1
    assert report["track_err_p95_rad"] is not None


# 用例: InferenceLoop 接线 recorder 后 flush 出可喂 extract_jitter 的三件套
def test_inference_loop_writes_robot_live_traces(tmp_path: Path) -> None:
    from fastervla.core.types import Observation
    from fastervla.deploy.backends.mock import MockBackend
    from fastervla.deploy.client.inference_loop import InferenceLoop, InferenceLoopConfig
    from fastervla.deploy.client.io.mock_sim import MockRobotPlatform
    from fastervla.deploy.client.plan_buffer import PlanBuffer
    from fastervla.deploy.postprocess.retime import UniformRetimer

    plat = MockRobotPlatform(
        state_dim=24,
        command_dim=21,
        camera_names=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        plant_tau_s=0.057,
        plant_delay_s=0.036,
        seed=0,
    )
    backend = MockBackend(horizon=30, action_dim=24, seed=0, mode="sine")
    backend.warmup()
    buf = PlanBuffer()
    seed = backend.sample(
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

    out = tmp_path / "robot-live-loop"
    rec = TraceRecorder(out, capacity_ticks=500, horizon=30)
    loop = InferenceLoop(
        platform=plat,
        backend=backend,
        plan_buffer=buf,
        retimer=UniformRetimer(policy_fps=30.0, kappa=1.0),
        config=InferenceLoopConfig(
            control_hz=100.0,
            duration_s=0.35,
            cold_start_latency_ms=20.0,
            latency_margin_ms=5.0,
        ),
        recorder=rec,
    )
    summary = loop.run()
    assert (out / "control_trace.npz").is_file()
    assert (out / "prediction_trace.npz").is_file()
    assert (out / "robot_bridge_report.json").is_file()
    ctl = np.load(out / "control_trace.npz")
    assert ctl["monotonic_s"].shape[0] == summary.control_steps
    assert summary.extras.get("trace_dir") == str(out)
    pred = np.load(out / "prediction_trace.npz")
    assert pred["actions"].ndim == 3
    assert pred["actions"].shape[1:] == (30, 24)
    report = json.loads((out / "robot_bridge_report.json").read_text())
    assert "inference_records" in report
    assert report["experiment"].get("kappa") == 1.0
    # LeadTracker on → reference and command may differ on arm dims.
    if ctl["monotonic_s"].shape[0] > 5:
        ref = ctl["reference21"]
        cmd = ctl["command21"]
        # At least tracker_status recorded as lead (1) on some step ticks.
        assert np.any(ctl["tracker_status"] == 1) or np.allclose(ref, cmd)


# 用例: aux 列（实测速度/时间戳/耗时/饱和）写盘；缺省为 NaN；追加到旧 trace 时补 NaN
def test_recorder_aux_tick_columns(tmp_path: Path) -> None:
    out_dir = tmp_path / "robot-live-aux"
    rec = TraceRecorder(out_dir, capacity_ticks=16, horizon=30)
    _tick(rec, 0.0)
    rec.flush()
    vel = np.arange(24, dtype=np.float64)
    _tick(
        rec,
        1.0,
        aux={
            "measured_velocity24": vel,
            "state_stamp_ms": 123.0,
            "send_ms": 0.7,
            "sat_excess_rad": 0.01,
        },
    )
    rec.close()
    ctl = np.load(out_dir / "control_trace.npz", allow_pickle=True)
    assert ctl["measured_velocity24"].shape == (2, 24)
    assert np.all(np.isnan(ctl["measured_velocity24"][0]))
    np.testing.assert_allclose(ctl["measured_velocity24"][1], vel)
    np.testing.assert_allclose(ctl["arm_velocity14"][1], vel[8:22])
    assert ctl["state_stamp_ms"][1] == 123.0
    assert np.isnan(ctl["state_read_ms"][1])
    assert ctl["sat_excess_rad"].shape == (2,)
