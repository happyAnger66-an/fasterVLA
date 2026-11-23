# fasterVLA · Trace / Metric 设计（F3 前）

> 状态：v0.2 · 2026-11-23（P0 Recorder 已实现）。  
> 配套：[deploy-g2-plan.md](deploy-g2-plan.md) §4.7 · [design.md](design.md)（速度轨 / 质量轨）。  
> 对照只读：`g2_tools/h7_deploy/scripts/robot_bridge.py`（ControlTrace / prediction_trace / report）、`tools/jitter/`（hf3 / hf7）。  
> Canvas 摘要：会话内 `fastervla-trace-observability.canvas.tsx`（与本文同步意图）。

## 0. 结论

- **P0 已落地**：`src/fastervla/deploy/client/recorder.py`；`InferenceLoop(recorder=...)` 接线；测试 `tests/deploy/test_recorder.py`。
- 落盘三件套：`control_trace.npz` / `prediction_trace.npz` / `robot_bridge_report.json` → `results/robot-live-<id>/`。
- **A/B 双缓冲**：`flush()` 封存当前 bank 写盘时，另一侧继续 `record_*`；多次 flush **追加合并**进同一 npz。`close()` = 最终 flush 并拒绝再写。
- 热路径只拷贝；写盘在 `flush` / 后台 writer。仅当 **两侧都满且 sealed 队列饱和**（`max_sealed_segments`）才 `drops++`。
- 仍缺：`safety.py` 降 κ。
- **离线分析已本仓自包含**：`deploy/analysis/`（hf3/hf7、exec、tracking、r2r）；`fastervla-deploy analyze`。口径对齐原 `tools/jitter`，但**运行时不依赖**仓外路径。

## 1. 双路径架构

| 路径 | 热路径做什么 | 旁路 / 事后做什么 | 产物 / 用途 |
|---|---|---|---|
| **离线** | `record_tick` / `record_chunk`：短临界区拷贝进 ring / 预分配数组 | 后台 writer 批量 flush；分析脚本读盘 | `control_trace.npz` · `prediction_trace.npz` · `report.json` |
| **在线** | 只 `LatencyWindow.record` 或写固定 ring；读**上次缓存**的 p95 | 推理线程 / 旁路线程算分位数；stdout 节流告警 | r2r P95 → \(\hat d\) / L2 预算；超预算 → safety 降 κ |

### 1.1 性能约束（硬）

100 Hz 控制拍内 **禁止**：

- 全量 `json.dumps` trace
- `np.savez` / 同步磁盘 I/O
- 对整窗做 `percentile` 排序

**允许**：预分配 float 行写入；`Queue.put_nowait`（满则丢并 `drops++`）；读原子拷贝的「上次 p95」。

## 2. 老栈落盘与已有分析（对齐目标）

| 产物 / 工具 | 粒度 | 关键字段 | 消费方 |
|---|---|---|---|
| `control_trace.npz` | ~100 Hz 拍 | `monotonic_s`, `measured_state24`, `reference21`, `command21`, `stage`(`step`/`initial_hold`/`buffer_empty_hold`), `action_frame` | `analyze_tracking` / `analyze_jitter` / `extract_jitter`(exec) |
| `prediction_trace.npz` | accepted chunk | `actions` / `actions_raw`, request/ready/obs mono, `rtc_*`, `overlap_*`, `frame_times_s`, `handoff_at` | `extract_jitter`(chunk hf3/hf7) / `replay_timeline` |
| `robot_bridge_report.json` | run + 每包 | `timing.*_ms`, `buffer_empty_cycles`, `tracking_warning_*`, 交接 Δa, `servo_timing` | 全部离线脚本 + 人工 |
| **`tools/jitter`（已有，勿重做）** | 离线 per-run / 多 run | `hf3_mrad` · `hf7_mrad` · hf3/速度 · 换向率 · `handoff_jump` · cmd/meas hf3 | `build_report.py` · 放球日报「抖动 hf3/hf7」 |

### 2.1 hf3 / hf7 口径（复用）

| 指标 | 含义 | 采样 |
|---|---|---|
| **hf3** | ≥3 Hz 高频带 RMS（mrad）；chunk 侧约 3–7.5 Hz；exec 侧约 3–8 Hz | `prediction_trace.actions(_raw)` / `control_trace` command·measured |
| **hf7** | ≥7.5 Hz 带（chunk，30 Hz 奈奎斯特内）；更接近锯齿 / 量化噪声 | 同上 |
| **hf3/速度** | hf ÷ 平均关节角速度，公平对比 | 离线派生 |
| **换向率** | 速度符号翻转 / 关节·秒 | 离线派生 |

实现：`r-infra/tools/jitter/extract_jitter.py` + `build_report.py`。fasterVLA 只保证落盘字段可喂入。

## 3. 指标对照：老栈 → 本仓

| 指标 | 老栈记录 | 本仓现状 | 离线必需 | 在线用途 | 缺口 |
|---|---|---|---|---|---|
| r2r 延迟 | `request_to_ready_ms` + 分解 timing | `InferenceLoop` deque P95（内存）；shadow `infer_ms` | 是 | \(\hat d\) / QP 预算 | 部分 |
| 推理延迟 (RTC \(d\)) | `rtc_d` / `inference_delay` 帧数 | \(d̂\) 有计算；不落盘 | 是 | 开推前缀 | 部分 |
| 相机 / obs 龄 | `observation_age_at_capture_ms`, `obs_warn` | 无专用字段；`image_ts≈state_ts` | 是 | 告警节流 | **缺失** |
| 抖动 hf3 / hf7 | `tools/jitter` + 日报 | 脚本已有；缺兼容 trace → 跑不起来 | 是（复用） | 否（事后） | 部分 |
| 交接跳变 Δa | `previous_command_difference`, overlap/guided_gap | `last_merge` 有 `d_actual`；无 Δa | 是 | 可选 stdout | **缺失** |
| 空缓冲 / step empty | `stage=buffer_empty_hold` + `buffer_empty_cycles` | `empty_buffer_ticks`；`early_idle_s=0` | 是 | summary | 部分 |
| starved (\(d̂\)>剩余) | 无字面；语义并入 empty | `starved_events++`；不降 κ | 是 | safety→κ | 部分 |
| 跟踪误差 \|r−meas\| | 每拍 trace + `tracking_warning` | 无 measured/reference 同拍记录 | 是 | 节流 warning | **缺失** |
| servo 拍内耗时 | `servo_timing_ms` | 无 | 建议 | 过载检测 | **缺失** |
| retime / velocity hits | `arm_timeline` meta | `retime_status=uniform`；F3 未接 | 是（F3） | 否 | 部分 |

## 4. 本仓已有碎片

| 组件 | 路径 | 状态 |
|---|---|---|
| **`TraceRecorder`** | `deploy/client/recorder.py` | **P0 已实现**；`record_tick` / `record_chunk` / `flush` / drops |
| `LatencyWindow` | `core/clock.py` + recorder 内 `r2r_ms` 等 | recorder 已挂窗；InferenceLoop 仍自建 deque 估 \(\hat d\) |
| `InferenceLoop` 接线 | `inference_loop.py` | 可选 `recorder=`；finally `flush`；summary.extras[`trace_dir`] |
| CLI `shadow` / `run` / `analyze` | `deploy/cli.py` | 录制 + 离线汇总 |
| `deploy/analysis/` | `spectrum` / `jitter` / `tracking` / `summarize` | **已落地**；本仓闭环 |
| **仍缺** | `safety.py` | P2 |

### 用法

```python
from fastervla.deploy.client.recorder import TraceRecorder
from fastervla.deploy.analysis import analyze_run

rec = TraceRecorder("results/robot-live-demo", capacity_ticks=int(duration_s * 100) + 100)
loop = InferenceLoop(..., recorder=rec)
summary = loop.run()  # 内部 close
# → results/robot-live-demo/{control_trace,prediction_trace,robot_bridge_report}

stats = analyze_run("results/robot-live-demo")
print(stats["headline"])  # hf3 / hf7 / track_p95 / r2r_p95 / empty_frac
```

```bash
fastervla-deploy analyze --run results/robot-live-<id> --out summary.json
fastervla-deploy analyze --results-dir results --out all.json
```

## 5. 与 deploy-g2-plan §4.7 对照

| 层级 | 规划字段 | 实现 |
|---|---|---|
| 每拍 | measured/reference/command/progress/plan_id/stage/tracker_status/servo_tick_ms | **已记**；Lead 开启时 reference≠command |
| 每 chunk | r2r 分解、d̂/d、rtc、overlap/guided_gap、prev_cmd_Δ、frame_times、obs/image age | **已记** |
| 汇总 | report：empty/starved/track_p95/r2r_p95/infer_records/experiment | **已记**；hf3 仍用 `tools/jitter` 事后算 |
| L0 safety | 在线 r2r P50/P99 → 降 κ | 仍缺 `safety.py`（窗已有，未驱动 κ） |

## 6. 目标 schema（离线最小完备集）

产物目录：`results/robot-live-<id>/`（与老栈命名对齐，便于 `extract_jitter`）。

### 6.1 `control_trace`（每拍）

| 字段 | 类型 / 形状 | 说明 |
|---|---|---|
| `monotonic_s` | float64 `[T]` | 控制拍墙钟（monotonic） |
| `measured_state24` | float64 `[T,24]` | 实测策略空间状态 |
| `reference21` | float64 `[T,21]` | tracker 前参考（L3 输入） |
| `command21` | float64 `[T,21]` | 实际下发 |
| `progress` | float64 `[T]` | 小数模型步 |
| `plan_id` | int32 `[T]` | 当前计划世代 |
| `stage` | int8 `[T]` | `0=step`, `1=hold`, `2=buffer_empty`（与老栈语义对齐） |
| `tracker_status` | int8 `[T]` | 预留：passthrough / lead / mpc / fallback |
| `contact` | int8 `[T]` | F4；先填 0 |
| `servo_tick_ms` | float32 `[T]` | 可选：本拍墙钟耗时 |

### 6.2 `prediction_trace` / chunk 记录（每次 accept）

| 字段 | 说明 |
|---|---|
| `actions` / `actions_raw` | `(N,H,24)` 滤波后 / 模型原始（无滤波则二者同） |
| `requested_monotonic_s` / `ready_monotonic_s` | → **r2r_ms** |
| `observation_monotonic_s` / `observation_age_ms` | 相机 / obs 龄 |
| `d_hat` / `d_actual` / `anchor_step` / `plan_id` | L1 对齐 |
| `rtc_applied`, `rtc_d`, `rtc_s`, `rtc_guided_gap` | RTC |
| `step_dt`, `retime_status`, velocity hits（F3） | L2 |
| `handoff_at_monotonic_s`, `overlap_gap` / `previous_command_difference` | 交接 |
| `frame_times_s` | 重定时节点（可选，喂 replay_timeline） |

### 6.3 `report.json`（run 级）

- 计数：`buffer_empty_cycles`, `starved_events`, `tracking_warning_cycles`, `trace_drops`
- `infer_records[]`：每包 accepted/rejected + 完整 timing 分解
- `servo_p95_ms`、实验身份（tag / config / κ / hostname）

## 7. 在线路径

统一 `LatencyWindow`（或共享 ring）：

| 窗 | 写入方 | 读取方 |
|---|---|---|
| `r2r_ms` | 推理线程（ready 时） | \(\hat d\)、L2 buffer 下界、safety |
| `image_age_ms` | 开推抓图后 | 节流 `obs_warn` |
| `servo_tick_ms` | 控制环末尾 | 过载检测 |

- 控制环**只读缓存 p95**，不在拍内重算。
- stdout JSON warning：**≥1 s 节流**（对标老栈 `tracking_warning`）。
- `starved` / P99 超预算 → 回调改 κ（`safety.py`，可后于 Recorder）。

## 8. 模块与接线

```text
ControlLoop / InferenceLoop
        │ record_tick / record_chunk（拷贝）
        ▼
  TraceRecorder（ring + drops）
        │ put_nowait
        ▼
  WriterThread → results/robot-live-<id>/{control_trace,prediction_trace,report}.…
        │
        ▼ 事后
  tools/jitter/extract_jitter.py · h7 analyze_tracking / analyze_jitter
```

- 实现位置：`src/fastervla/deploy/client/recorder.py`
- 测试：`tests/deploy/test_recorder.py`（先测后码）
- 可选薄封装：`deploy/analysis/` 调老栈脚本或复述同口径汇总；**不 fork** `tools/jitter` 指标定义。

## 9. 落地优先级（F3 前）

| P | 项 | 理由 |
|---|---|---|
| **P0** | 每拍 measured / reference / command / progress / stage | 否则无法算跟踪、hf3 执行侧、空缓冲时段 |
| **P0** | 每 chunk r2r 分解 + \(d̂\)/\(d\) + rtc | L1/L2 验收与老栈 A/B 核心 |
| **P0** | 异步 writer + `trace_drops` | 保证异步不影响控制性能 |
| P1 | 交接 Δa / guided_gap 落盘 | 换段质量；merge 时即可算 |
| P1 | obs/camera age + servo_tick_ms | 延迟预算与过载 |
| P2 | 接 `tools/jitter` + `analyze_*` | hf3/hf7 已有；字段对齐即可 |
| P2 | safety 在线降 κ | L0；可在 Recorder 之后 |

### 建议顺序

1. **TraceRecorder 契约**（先测后码）：`record_tick` / `record_chunk` / `flush` / drops。
2. 控制环 + 推理环接线；shadow / execute 统一写 `robot-live-<id>/`。
3. 用一份 mock 或真机短跑产物跑通 `extract_jitter.py`。
4. 再接 safety / analysis 汇总。

## 10. 非目标

- 不改 `g2_tools/h7_deploy/**`。
- 不在本仓重定义 hf3/hf7 公式（以 `tools/jitter` 为准）。
- 本期不上 Prometheus / 外部 metrics 服务；在线 = 内存滑窗 + 节流 stdout。
