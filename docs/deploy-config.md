# Deploy 配置说明（加速比 κ 与约束）

> 状态：2026-11-24。  
> **真机入口**：`fastervla-deploy run --config configs/deploy/<recipe>.yaml`（CLI 显式 flag 覆盖配方）。  
> 下文是与代码对齐的配方格式；**仅 CLI 已接线字段**可写入 YAML（见 `deploy/recipe.py`）。QP 细参等未接线键会被拒绝。

相关：[`deploy-g2-plan.md`](deploy-g2-plan.md) §4.3、[`retime-qp-flow.html`](retime-qp-flow.html)（TimeAxisQP 公式）。

---

## 1. 怎么配「加速比」

| 你想做什么 | 配什么 | 说明 |
|---|---|---|
| 整体提速到 1.2× | `kappa: 1.2` | 相对 `policy_fps`（默认 30）的目标倍速 |
| 禁止超过 2.0× | `kappa_max: 2.0` | 人在回路 / 将来 safety 降速的天花板（CLI 默认） |
| 降速地板 | `kappa_min: 1.0` | 超预算时不应低于此（safety 未接线前仅校验） |
| 匀速时间轴（F2） | `retimer: uniform` | `step_dt ≡ 1/(κ·fps)`，无 v_max QP |
| 物理盒内重定时（F3） | `retimer: qp` | κ 是软目标；大位移段会被 `v_max` / `dt` 盒拉慢 |

**硬校验（启动即失败）**

1. `kappa_min ≤ kappa ≤ kappa_max`
2. 若用 TimeAxisQP：`1/(policy_fps·κ)` 必须落在 `[dt_min_s, dt_max_s]`  
   - 例：`fps=30, κ=1.2` → `dt_ref≈27.8ms`，需 `dt_min≤27.8≤dt_max`  
   - `κ=2, dt_min=20ms` → `dt_ref≈16.7ms < 20ms` → **拒绝**

**配方速查（推荐）**

```bash
# h7 正常倍速 soft（4+4 + ZP + pchip）
fastervla-deploy run --config configs/deploy/h7_soft_1x.yaml

# h7 1.8× soft；CLI 覆盖 mode / duration
fastervla-deploy run --config configs/deploy/h7_soft_1p8.yaml --mode execute --duration-s 30
```

**CLI 速查（无配方时）**

```bash
# 1.0× Uniform（默认）
python -m fastervla.deploy.cli run --mode shadow --kappa 1.0 --retimer uniform

# 1.2× TimeAxisQP
python -m fastervla.deploy.cli run --mode shadow --kappa 1.2 \
  --retimer qp

# 可选：打开老栈式 inference-buffer 墙钟地板
python -m fastervla.deploy.cli run --mode shadow --kappa 1.4 --retimer qp \
  --qp-buffer-window
```

| CLI | 对应字段 |
|---|---|
| `--kappa` | `kappa` |
| `--kappa-max` / `--kappa-min` | `kappa_max` / `kappa_min` |
| `--retimer {uniform,qp}` | 选 `UniformRetimer` 或 `TimeAxisQP` |
| `--qp-buffer-window` | `time_axis_qp.buffer_window_enabled` |
| `--t-infer-min-s` / `--t-switch-min-s` | L1 墙钟节流 |
| `--prefix-frames` / `--buffer-frames` | L1 **4+4 合块节流**（`execution_prefix_frames` / `execution_buffer_frames`；默认 4/4；`0` 关 prefix 门） |
| `--arm-chunk-filter` | 路径平滑：`none` / `zero_phase_dct` / `zero_phase_dct_cross`（默认 `none`；soft 用 `zero_phase_dct`） |
| `--arm-filter-cutoff-hz` | ZP 截止频率（默认 `3`） |
| `--arm-interp` | `PlanBuffer.reference`：`linear` / `pchip`（默认 `linear`；soft 用 `pchip`） |

软栈对齐路径平滑示例：

```bash
python -m fastervla.deploy.cli run --mode shadow \
  --arm-chunk-filter zero_phase_dct --arm-filter-cutoff-hz 3 --arm-interp pchip
```

其余 QP 约束（`dt_*`、`v_max_*`、`lambda_*`）**目前只在 `TimeAxisQPConfig` 默认值 / 代码构造处**，CLI 尚未全部暴露；改示例 YAML 或直接在构造 `TimeAxisQPConfig(...)` 时传入。

> **注意**：`retimer.prefix_frames`（QP buffer 窗）与 CLI `--prefix-frames`（合块门控）是两套字段；合块节流走 `InferenceLoopConfig.execution_*`。

---

## 2. 推荐 YAML 结构

根是一个 mapping；未知键应拒绝（与 `core/config.from_mapping` 约定一致）。分段如下：

```yaml
# configs/deploy/g2_badminton651_example.yaml
meta:
  name: g2_badminton651_example
  robot_aid: G2A0004C205445

run:
  mode: shadow          # shadow | execute
  duration_s: 60.0
  results_dir: results
  flush_every_s: 30.0
  no_trace: false

policy:
  host: 10.42.1.201
  port: 8013
  prompt: "Put the shuttlecock into the shuttlecock tube."
  policy_fps: 30.0
  horizon: 30
  action_dim: 24

speed:
  kappa: 1.2
  kappa_max: 2.0
  kappa_min: 1.0

retimer:
  type: qp              # uniform | qp
  # --- 仅 type=qp 时生效（TimeAxisQPConfig）---
  dt_min_s: 0.015
  dt_max_s: 0.08
  lambda_time: 0.1
  lambda_acc: 1.0
  v_max_rad_s: null     # null | 标量 | 长度 14 的列表
  v_max_from_lag: false
  lag_tau_s: 0.057
  lag_gain: 0.25
  lag_max_lead_rad: 0.06
  buffer_window_enabled: false
  prefix_frames: 4
  buffer_frames: 4
  r2r_default_ms: 100.0
  buffer_margin_ms: 20.0
  r2r_window: 30
  solver: active_set    # active_set | osqp | lbfgsb
  calibrate_iters: 8

inference:
  control_hz: 100.0
  rtc_mode: INPAINT
  execution_prefix_frames: 4   # 合块门：接受后至少推进这么多模型帧再重规划；0=关
  execution_buffer_frames: 4   # remaining_steps ≤ 此值时提前放行（防空缓冲）
  arm_chunk_filter: none       # none | zero_phase_dct | zero_phase_dct_cross
  arm_chunk_filter_cutoff_hz: 3.0
  arm_interpolation: linear    # linear | pchip
  t_infer_min_s: 0.0
  t_switch_min_s: 0.0
  latency_margin_ms: 10.0
  cold_start_latency_ms: 100.0
  latency_window: 32
  use_lead_tracker: true
  track_warn_threshold_rad: 0.10

lead_tracker:
  tau_s: 0.057
  gain: 0.25
  max_lead_rad: 0.06
  lead_smooth_tau_s: 0.040
  hold_during_blend: true

robot:
  sdk_root: /home/agi/app/gdk
  # 其余 G2 硬约束见 deploy-g2-plan §2；多数在代码里写死校验
```

---

## 3. 字段详解

### 3.1 `speed` — 加速比

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `kappa` | float | `1.0` | 本跑目标倍速。Uniform：每段 `dt=1/(κ·fps)`；QP：目标速率 `s_ref=fps·κ` |
| `kappa_max` | float | `2.0` | 允许上限，不得越过（CLI 默认） |
| `kappa_min` | float | `1.0` | 降速地板 |

建议扫参顺序：`1.0`（持平）→ `1.2` → `1.4` → `1.5` → `1.6+`。真机 `kappa_max` 默认 `2.0`；κ≥1.5 时建议加大 `--prefix-frames`。

### 3.2 `retimer` — 时间轴与物理约束

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `type` | string | `uniform` | `uniform`：匀速；`qp`：TimeAxisQP |
| `dt_min_s` | float | `0.015` | 单段最短时长（→ 最大速率 `s_max=1/dt_min`） |
| `dt_max_s` | float | `0.08` | 单段最长时长（→ 最小速率 `s_min=1/dt_max`） |
| `lambda_time` | float | `0.1` | QP：贴近 `target≈s_ref` 的权重（须 >0） |
| `lambda_acc` | float | `1.0` | QP：相邻段速度平滑权重 |
| `v_max_rad_s` | null / float / [14] | `null` | 臂关节角速度上限（rad/s）。`null`=不额外限；标量广播到 14 维 |
| `v_max_from_lag` | bool | `false` | `true` 时用 soft-lead 导出上限 `max_lead/(gain·τ)` 与显式 `v_max` 取更紧 |
| `lag_tau_s` | float | `0.057` | 仅 `v_max_from_lag` 时用 |
| `lag_gain` | float | `0.25` | 同上 |
| `lag_max_lead_rad` | float | `0.06` | 同上 |
| `buffer_window_enabled` | bool | `false` | `true`：prefix 后 buffer 窗墙钟 ≥ r2r 预算（老栈 hack；F3 默认关） |
| `prefix_frames` | int | `4` | buffer 窗起点（模型步） |
| `buffer_frames` | int | `4` | buffer 窗长度 |
| `r2r_default_ms` | float | `100` | 样本不足时的 r2r 默认 |
| `buffer_margin_ms` | float | `20` | 加在 r2r p95 上的余量 |
| `r2r_window` | int | `30` | 计算 p95 的样本窗长 |
| `solver` | string | `active_set` | `active_set` / `osqp` / `lbfgsb` |
| `calibrate_iters` | int | `8` | 外层校准 `target` 的最大迭代 |

**约束怎么影响加速**

```text
想加快（↑κ）
  → s_ref ↑ → 名义 dt ↓
  → 仍受 dt_min 卡住（κ 过大启动失败）
  → 大 |dp| 段还受 v_max：dt ≥ |dp|/v_max（QP 自动拉长该段）
  → 瞬时倍速 ≠ 常数 κ；总时长通常仍短于 1.0×
```

调参经验：

- 抖 / 啸叫：略降 `kappa`，或收紧 `v_max_rad_s`，或增大 `lambda_acc`
- 空缓冲 / 等推理：优先 L1（`latency_margin_ms`、`t_infer_min_s`），不要只靠把 `dt_max` 拉大
- 与老栈 A/B：打开 `buffer_window_enabled: true`

### 3.3 `inference` — L1 环（`InferenceLoopConfig`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `control_hz` | `100` | 控制环频率 |
| `policy_fps` | `30` | 与 `speed`/`retimer` 共用的策略帧率 |
| `rtc_mode` | `INPAINT` | 开推 RTC 模式 |
| `execution_prefix_frames` | `4` | 合块门：接受后至少推进这么多模型帧再重规划；`0` 关闭 |
| `execution_buffer_frames` | `4` | `remaining_steps ≤` 此值时提前放行（防空缓冲） |
| `arm_chunk_filter` | `none` | 推理后、Retimer 前对臂维做 ZP DCT |
| `arm_chunk_filter_cutoff_hz` | `3` | ZP 截止（Hz） |
| `arm_interpolation` | `linear` | `PlanBuffer.reference` 插值；`pchip` 对齐 soft |
| `t_infer_min_s` | `0` | 两次推理之间最小墙钟间隔 |
| `t_switch_min_s` | `0` | 两次换段（merge）最小间隔 |
| `latency_margin_ms` | `10` | 估 \(\hat d\) 时叠在 r2r P95 上的裕量 |
| `cold_start_latency_ms` | `100` | 尚无 r2r 样本时的延迟假设 |
| `latency_window` | `32` | r2r / 延迟滑窗长度 |
| `use_lead_tracker` | `true` | 是否启用 soft lead |
| `track_warn_threshold_rad` | `0.10` | 跟踪误差告警阈值（与方案 `max_tracking_error_rad` 对齐） |
| `horizon` / `action_dim` | `30` / `24` | chunk 形状 |
| `duration_s` | 由 CLI | 本跑时长 |

### 3.4 `lead_tracker` — L3 基线（`LeadTrackerConfig`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `tau_s` | `0.057` | 超前时间常数（≈57ms） |
| `gain` | `0.25` | lead 增益 |
| `max_lead_rad` | `0.06` | 单关节 lead 限幅 |
| `lead_smooth_tau_s` | `0.040` | lead EMA 平滑 |
| `hold_during_blend` | `true` | 交接 blend 期间 hold lead |

与 `retimer.v_max_from_lag` 共用同一套 lag 直觉：`v_cap ≈ max_lead/(gain·τ)`。

### 3.5 `policy` / `run` / `robot`

| 段 | 要点 |
|---|---|
| `policy.host/port` | 5090 openpi websocket（默认 `10.42.1.201:8013`） |
| `policy.prompt` | 与 server metadata 校验一致 |
| `run.mode` | 默认 `shadow`；`execute` 需 TTY |
| `run.flush_every_s` | Trace 周期落盘；`0` 只在结束写 |
| `robot.sdk_root` | GDK 根目录 |

G2 硬事实（`robot_aid`、21 维命令、健康字段等）见 [`deploy-g2-plan.md`](deploy-g2-plan.md) §2，多数在 `agibot_g2` / `gdk_real` 内校验，不建议用 YAML 关掉。

---

## 4. 与代码 dataclass 的对应关系

| YAML 段 | 代码类型 | 路径 |
|---|---|---|
| `speed` + 部分 `inference` | `InferenceLoopConfig` | `deploy/client/inference_loop.py` |
| `retimer`（uniform） | `UniformRetimer` | `deploy/postprocess/retime.py` |
| `retimer`（qp） | `TimeAxisQPConfig` | 同上 |
| `lead_tracker` | `LeadTrackerConfig` | `deploy/client/tracker/lead.py` |
| （加载工具） | `from_mapping` / `load_dataclass_yaml` | `core/config.py` |

未知键：`from_mapping` 会 `KeyError`。新增字段先改 dataclass，再改本文与示例 YAML。

---

## 5. 最小可跑配方

**持平基线（1.0× Uniform）**

```yaml
speed: { kappa: 1.0, kappa_max: 2.0, kappa_min: 1.0 }
retimer: { type: uniform }
```

**提速 + QP（1.2×）**

```yaml
speed: { kappa: 1.2, kappa_max: 2.0, kappa_min: 1.0 }
retimer:
  type: qp
  dt_min_s: 0.015
  dt_max_s: 0.08
  v_max_rad_s: 2.0          # 可按关节再拆成 14 维
  v_max_from_lag: false
  buffer_window_enabled: false
```

**更猛（1.4×）仍卡在物理盒内**

```yaml
speed: { kappa: 1.4, kappa_max: 2.0, kappa_min: 1.0 }
retimer:
  type: qp
  # 确认 1/(30*1.4)≈23.8ms ∈ [dt_min, dt_max]
  dt_min_s: 0.015
  dt_max_s: 0.08
  lambda_acc: 1.0
  v_max_rad_s: 2.0
```

---

## 6. 尚未接线 / 不要指望的配置

| 项 | 状态 |
|---|---|
| `fastervla-deploy run --config xxx.yaml` | **已接线**（`deploy/recipe.py` → argparse defaults；仅 CLI 字段） |
| TimeAxisQP 细参（`dt_*` / `v_max_*` / `lambda_*`）进 YAML | 未接线；写进配方会 KeyError |
| `lead_tracker` 段 | **已接线**（→ CLI `--lead-*` → `InferenceLoopConfig` → `LeadTrackerConfig`） |
| `retimer.dwell_kappa_max` | **已接线**（远离接触的近静止帧钉到更高倍速；`fastervla-deploy report` 汇总球周期） |
| `safety` 在线按 r2r P99 自动降 `kappa` | 未实现（方案有，`safety.py` 缺） |
| Contact（F5） | 未实现 |
| acados 系统库 | `pip install 'fastervla[mpc]'` 只装 Python 包；还需本机编译/安装 [acados](https://docs.acados.org) 并设置 `ACADOS_SOURCE_DIR`。未装齐时 `AcadosMpcTracker` 自动降级 Lead |
| 训练配方 `configs/training/*.yaml` | 另一套，见训练 README / recipe |

---

## 7. 离线指标（`analyze`）

`fastervla-deploy analyze` 对 `robot-live-*` 出四类统计（**不含**易误解的 `prog_jump`）：

| 类 | 代表字段 |
|---|---|
| 加速 | `progress_wall_fps`（按 `plan_id` 段累加净模型步）、`cmd_spd_p95`、`ee_speed_p50/p95`（g2_tools URDF FK）、`step_dt_*` |
| 摆动/犹豫 | `merge_hz`、`sw_ms_p50`、`ref/cmd_jump_sw_p95`、`cmd_hf3`、`rev_per_s`、`handoff_jump_p95` |
| 稳定 | `empty` / `starved` / `empty_frac`、`rem_min`、`r2r_p95`、`infer_duty`、`servo_tick_p95` |
| 准确 | `ref_track_p95`（高 κ 主看）、`cmd_track_p95`、`lead_p95`、`frac_*_gt_100mrad` |

```bash
# 单 run：打印 headline；完整 JSON 含 speed/sway/stability/accuracy + wide 行
python -m fastervla.deploy.cli analyze --run results/robot-live-... --out /tmp/sum.json

# 目录扫描成宽表（日报一行一 run）
python -m fastervla.deploy.cli analyze --results-dir results --wide --out /tmp/wide.json

# EE FK：默认搜 g2_tools URDF；可显式指定
python -m fastervla.deploy.cli analyze --run ... --urdf /path/to/G2_t2_crsB.urdf --ee-stride 5
```
