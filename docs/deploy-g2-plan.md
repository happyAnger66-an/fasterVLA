# fasterVLA deploy · G2 落地方案（交接给实现 chat）

> 状态：v0.1 · 2026-11-23。总体架构以 [design.md](design.md) 为准；本文补「在智元 G2（huixi）上能跑」所需的硬约束、分期和验收。
>
> 参考文章：[真机部署加速方案](http://www.xiaoanger.com/topics/robot/deploy/execution/openpi-exec-plan/)（L0–L5）。
>
> 老栈（只读参考，**不要改**）：`r-infra/g2_tools/h7_deploy/`，真机部署在 `agi@g2s:/data/project_r/agibotg2_pi0.5_deploy/`。真机 A/B 仍由老栈继续跑，本仓库成熟前两边并存。

## 0. 目标与边界

- 在 `src/fastervla/deploy/` 实现文章 L1–L4（+L0 计时）分层执行栈，支持 G2 badminton651 现有推理与控制。
- **推理节点不动**：继续用 5090 上已有 openpi websocket 服务（RTC 包，端口 8013，host `10.42.1.201`）。本仓库只写 client 侧 backend。
- **首个可交付**：在 G2 上 1.0× 与老栈 soft 配置持平，然后 1.2 → 1.4 → 1.5 逐级对标老栈 QP 结果。
- 不 vendor openpi / 老栈；算法按新接口重写，可逐段对照老栈单测行为。

## 1. 与文章分层的映射，以及相对老栈要修的问题

| 层 | 文章要求 | 老栈现状 | 本仓库要做的 |
|---|---|---|---|
| L0 | 统一时钟；图像 / 状态 / 执行延迟 P50/P99；超预算降速 | 有 trace，无在线预算降速 | `core/clock.py` + `deploy/client/safety.py`：在线 r2r P50/P99；超预算 → 降 κ |
| L1 | 推理线程串行；开推时按延迟估 \(\hat d\) 冻前缀；`T_switch` 换段时用实际 \(d\) 对齐 | 固定 `prefix+buffer` 帧（4+4 / 5+5）；\(d\) 由预定交接点反推；早到也等交接点 | `PlanBuffer`（模型步计数）+ `InferenceLoop`：\(\hat d\) = 沿当前时间轴从「现在」累加到 `now + L̂_p95` 覆盖的模型步；结果到达后按实际 progress 接入；`T_infer_min` / `T_switch` 独立可配 |
| L2 | 路径不变，每段 \(s_i=1/dt_i\) 的时间轴 QP | `arm_timeline.py` 盒约束 QP（active-set），含 prefix lock 与 buffer 墙钟下界 | `postprocess/retime.py`：`UniformRetimer` + `TimeAxisQP`（算法移植老栈 `arm_timeline.py`，可选 OSQP 后端）；buffer 下界改为由 L1 的 \(\hat d\) 驱动 |
| L3 | 端侧跟踪：滞后回放估计 \(\hat q\) + MPC，硬约束 \(\Delta y\) / \(\Delta^2 y\) | 开环 soft lead：\(u=r+\tau\dot r\)，gain / 限幅 / EMA；交接 blend 时可 hold | `tracker/`：先 `LeadTracker`（移植 soft lead 作基线）→ `LagStateEstimator` → `AcadosMpcTracker`（可选依赖，失败降级） |
| L4 | 新息 EMA 判接触 → 降 \(v_{\max}\)、钉参考、冻进度 | 无 | `client/contact.py`：同 design §5.2；先离线回放调阈值，再真机 |

**相对老栈最想修掉的行为**：加速后「固定帧配额 + 非常数倍速」导致的等推理（`buffer_empty_hold`）/ 等执行（结果早到闲置）与交接犹豫。验收要专门看这两项。

## 2. G2 硬事实（必须写进 `EmbodimentSpec` / IO，不要靠猜）

全部来自老栈 `configs/common/robot_schema.json`、`scripts/robot_control.py`、`configs/badminton651/huixi_g2_async_rtc_smooth_zp_30hz_lag1a_soft_tl.json`。

| 项 | 值 |
|---|---|
| 机器人 | `robot_aid = G2A0004C205445`（不符即拒绝启动） |
| 策略状态 / 动作 | 24 维，名字顺序见 `robot_schema.json` 的 `state_names` / `action_names`：`[0:5]` 腰、`[5:8]` 头、`[8:15]` 左臂、`[15:22]` 右臂、`[22:24]` 左右夹爪 |
| 下发命令 | 21 维：`COMMAND_INDICES = [0..4, 8..23]`；腰 `[0:8]` 用起始 baseline **保持**，头不下发；夹爪原生名 `idx31_gripper_l_inner_joint1` / `idx71_gripper_r_inner_joint1` |
| 进 L2/L3 的维度 | 只有双臂 14 维（策略下标 8..21）；夹爪走 lookahead 旁路 |
| chunk | `H = 30`，`policy_fps = 30`，动作语义 `absolute_position_rad` |
| 相机 | `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`；GDK `Camera.get_latest_image` JPEG，`stretch_640x480`，`image_transport = jpeg` |
| SDK | `agibot_gdk`；`sdk_root=/home/agi/app/gdk`；`LOCATOR_IP=10.42.1.101`、`AORTA_DISCOVERY_URI=http://10.42.1.101:2379`，网卡 `enp1s0` |
| 伺服 | 控制环 100 Hz；`JointServoControlReq.control_period = 0.02`；`joint_servo_control` 超 50 ms 或返回非 0 → 立即失败，**绝不重试** |
| 读状态 | `get_whole_body_status` 健康字段全 0；`control_mode ∈ {1,3}`、`mode ∈ {0,1}` 且运行中不得变化；22 关节名集合精确匹配；夹爪 `end_state` 身份 / enable / err 校验；关节时间戳单调，停滞 150 ms 失败；单次读取超 50 ms 失败 |
| 限位 | `robot.get_joint_limits()` 逐关节 clip + 违规报错；`arm_clip_tolerance 0.02`、`gripper_clip_tolerance 0.01` |
| 安全阈值 | `max_tracking_error_rad 0.1`；原厂臂误差急停 0.3 rad（不要逼近）；`max_observation_age 0.2 s`；`max_prediction_age 0.8 s` |
| 物理执行 | 必须交互式 TTY 启动；`--mode shadow` 默认，`execute` 显式开启 |
| 推理请求 | openpi websocket `infer(obs)`：`state`(24 float32) + `images`(3 路 JPEG) + 可选 `rtc={prev_chunk_left_over (30,24), inference_delay d, execution_horizon H-s}`；返回 `actions (30,24)`、`policy_timing`、`rtc_applied` 等。启动时校验 server metadata（checkpoint 89999、`badminton651` norm、prompt、`num_steps 10`、`rtc: true`） |
| 滞后辨识 | 臂关节总滞后约 90 ms（纯延迟约 36 ms + τ 约 57 ms）；老栈 soft：`tau_ms 57, gain 0.25, max_lead 0.06, lead_smooth_tau 40 ms` |
| ZP 滤波 | 老栈在 chunk 上做零相位 DCT 低通 3 Hz（`arm_chunk_filter`）；本仓库作为可选 `postprocess` 一步 |

## 3. 目录与模块（在 design §4 基础上收窄到本期）

```text
src/fastervla/
  core/
    types.py          # Observation, Chunk, TimedChunk, Prefix, RobotState, RobotCommand
    clock.py          # monotonic 统一、延迟统计（滑窗 P50/P95/P99）
    config.py         # dataclass + 未知键报错
    protocol.py       # 与 openpi websocket 的请求/响应映射（含 rtc 字段）
  deploy/
    backends/
      openpi_ws.py    # OpenPiWebsocketBackend：连 5090:8013；metadata 校验；RTC INPAINT
      mock.py         # 可复现的假策略（正弦/录制回放）
    postprocess/
      chunk_filter.py # ZP 3 Hz（可选）
      retime.py       # UniformRetimer / TimeAxisQP
      pipeline.py
    client/
      io/
        interfaces.py # StateSource / CommandSink / ImageSource / RobotPlatform
        layout.py     # 24 ↔ 21 映射、腰 baseline 保持
        mock_sim.py   # 一阶滞后被控对象（τ、纯延迟可配）
        agibot_g2.py  # GDK 实现（§2 全部检查）
      plan_buffer.py
      loops.py        # InferenceLoop（线程）+ ControlLoop（100 Hz）
      estimator.py    # LagStateEstimator
      tracker/        # passthrough / lead / rate_limited / acados_mpc
      contact.py
      safety.py
      recorder.py
    analysis/         # 跟踪误差、空缓冲、早到闲置、交接 jerk
    cli.py            # fastervla-deploy run / shadow / replay
configs/
  embodiment/agibot_g2.yaml
  deploy/g2_badminton651_*.yaml
tests/deploy/ , tests/core/
```

依赖：`deploy` 不依赖 JAX；extras `deploy-client`（`websockets`/`openpi-client`、`scipy`）、`mpc`（acados，可选）、`osqp`（可选）。

## 4. 关键设计（实现时以此为准）

### 4.1 PlanBuffer（模型步是唯一尺子）

- 存当前计划：`waypoints[30,24]` + `step_dt[30]`（来自 Retimer）+ `origin_monotonic`。
- `progress`（小数模型步）由 `advance(dt_real, alpha)` 推进：在 `step_dt` 上累加；接触时 `alpha=0`。
- `prefix(d_hat)`：返回从当前 progress 起未来 `d_hat` 个模型步（绝对量）+ `anchor_step`，供 RTC。
- `merge(timed, anchor_step)`：用到达时的实际 progress 算 \(d\)，从新 chunk 的 `d` 处接上；比当前计划旧的结果丢弃；记录 `d_hat - d`。
- `reference(t_now, horizon_s)`：pchip 在真实时间上插值给 tracker；交接用速度匹配过渡（移植老栈 `robot_velocity_blend.py` 的 `VelocityMatchedHandoff`，时长可配）。
- 一把锁；推理线程只调 `prefix/merge`，控制环只调 `advance/reference`。

### 4.2 InferenceLoop（L1）

```text
loop:
  obs = 最新观测（GDK 三相机 + 24 维状态；状态按 DelayCompensator 对齐到图像时间）
  L̂ = 近期 request→ready 的 P95（冷启动 100 ms）+ margin
  d̂ = PlanBuffer 沿当前 step_dt 从 progress 累加到墙钟 L̂ 覆盖的模型步数（上取整）
  prefix = PlanBuffer.prefix(d̂)
  A = backend.infer(obs, rtc={prev_chunk_left_over: 旧计划从观测时刻起的剩余, inference_delay: d̂, execution_horizon})
  timed = pipeline(A)          # ZP → Retimer（prefix 段时长锁定为旧计划）
  PlanBuffer.merge(timed, anchor)
  等到 T_infer_min
```

- 不再有固定 `prefix+buffer` 帧配额；替代旋钮：`T_infer_min`、`T_switch_min`（距上次换段的最小墙钟间隔）、`latency_margin_ms`。
- 若 `d̂` 超过剩余计划长度 → 记 `starved` 事件并按 safety 策略降 κ。

### 4.3 κ 与 Retimer（L2）怎么配合

**κ 可配置（YAML / CLI，不要写死）** — 字段全集与示例见 **[deploy-config.md](deploy-config.md)**、[`configs/deploy/g2_badminton651_example.yaml`](../configs/deploy/g2_badminton651_example.yaml)。

| 键 | 含义 | 测试建议 |
|---|---|---|
| `kappa` / `kappa_target` | 本跑目标倍速（相对 `policy_fps`） | F0–F2 默认 `1.0`；F3 起扫 `1.2` / `1.4`；F5 试 `1.5` |
| `kappa_max` | 允许上限；safety / 人在回路不得越过 | 真机先钉死，例如本期实验 `kappa_max: 1.5` |
| `kappa_min` | 降速地板（超预算 / starved 时） | 如 `1.0`，避免降到不可用 |

在线「P99 超预算自动降速」= 把当前生效 κ 往下调（≥ `kappa_min`），不是改路点。

**与时间重规划的分工（文章 L2 / L5）**

```text
κ（全局意图：整体更快）
  → SpeedAdapter：名义 s_ref ≈ κ · policy_fps
      （或每步参考时长 Δt_i^ref ≈ Δt_demo / κ）
  → Retimer / TimeAxisQP：路径 waypoints 不变，只解每段 dt_i（或 s_i=1/dt_i）
      贴近 s_ref，且满足 dt 盒约束 + 逐关节 v_max（+ 可选 acc）
  → 输出非均匀 step_dt[]
  → L1 估 d̂ / 实际 d：必须沿 step_dt 累加墙钟，不能再假设固定 1/policy_fps
```

- **κ 单独匀速缩放**：大位移段更容易顶满 `v_max`、抖或越限 → 所以要有 L2。
- **L2 单独无 κ**：只能在示教节拍附近「抹平」段间快慢，缺少「整体提速」旋钮。
- **二者一起**：κ 给「想多快」的软目标；QP 在物理盒内尽量靠近，大位移段往往相对放慢、小位移段加快，总时长仍相对示教缩短。重定时后**沿轨迹的瞬时倍速不再等于常数 κ**。
- Uniform 回退（F2）：`step_dt_i = 1/(κ·policy_fps)`，无 QP；F3 再上 `TimeAxisQP`。

### 4.4 Retimer 实现要点

- `TimeAxisQP`：目标与约束同老栈 `arm_timeline.py`（`s_ref=κ·policy_fps`、`λ_time=0.1`、`λ_acc=1.0`、`dt∈[15,80] ms`、per-joint `v_max`、prefix 时长锁定、calibrated rates）。默认 active-set 盒 QP；`osqp` 装了可切换，二者在随机盒问题上结果需一致（单测）。
- 移除老栈里「buffer 窗墙钟 ≥ r2r」这一 hack 的必要性由 4.2 的 \(\hat d\) 承担；保留为可选开关，便于 A/B。
- 失败退 `UniformRetimer` 并在 recorder 里带 `retime_status`。

### 4.5 Tracker / Estimator（L3）

- `PassthroughTracker`：参考直接下发（对应老栈无补偿）。
- `LeadTracker`：老栈 soft lead 行为逐项对齐（`tau/gain/max_lead/lead_smooth_tau/hold_during_blend`），作为 1.0× 持平基线。
- `LagStateEstimator`：design §5.2；锚点取约 50 ms 前测量，用 \(x_{k+1}=a x_k+(1-a)y_k\) 回放已下发命令。
- `AcadosMpcTracker`：design §5.2 / 文章 2.3；默认 **N=8**、14 维臂、cython 封装；超时 / 失败降级到 `LeadTracker` 并计数。g2s shadow 实测约 **99 Hz**、超时率 < 1%。安装见 `scripts/install_acados_g2s.sh`。流程见 **[acados-mpc-tracker.html](acados-mpc-tracker.html)**。

### 4.6 Contact（L4）

- 新息 EMA、`threshold 0.15`、`hold_min 8` 拍、`v_scale 0.4`；触发时冻进度、钉参考、降 `v_max`。
- 先用老栈录的 `control_trace.npz` 离线回放统计夹球段新息分布再定阈值。

### 4.7 Recorder（与老栈分析可比）

详细设计见 **[trace-design.md](trace-design.md)**。**录制 + 离线分析已本仓闭环**：`TraceRecorder` → `robot-live-*`；`deploy/analysis/` 算 hf3/hf7/跟踪/r2r（不依赖 `tools/jitter` / h7）；`fastervla-deploy analyze`。

摘要：每拍 measured/reference/command/…；每 chunk r2r 分解、d̂/d、交接 Δa。热路径只拷贝；`flush`/`PeriodicFlusher`/`close` 写盘。仍缺 safety 降 κ。

## 5. 分期（每期先测试后实现，按 AGENTS.md）

| 期 | 内容 | 退出条件 |
|---|---|---|
| F0 | `core` 类型 / 配置 / 时钟；`io` 接口 + `mock_sim`；`MockBackend`；同步 1× 控制环 | CPU 上 mock 链路跑通；IO / layout 契约单测 |
| F1 | `agibot_g2.py`（读 / 发 / 图像）+ `OpenPiWebsocketBackend`；`shadow` 模式 | 在 g2s 上 shadow 跑 5 min：状态读数、图像、推理返回与老栈同一时刻对比一致；零下发 |
| F2 | L1：`PlanBuffer` + `InferenceLoop` + RTC INPAINT + `LeadTracker`，κ=1.0，Uniform 时间轴 | 真机 execute 1.0×：|r−meas| p95、空缓冲、成功率与老栈 soft（`qHmSg38e`）同档 |
| F3 | L2：`TimeAxisQP`，κ=1.2 / 1.4（**单测已完成**；`run --retimer qp`） | 对标老栈 `vSLxMWJg`（1.2）、`tl_qp_1p4_5_5`；空缓冲与早到闲置均不高于老栈 |
| F4 | L3：`LagStateEstimator` + `AcadosMpcTracker`（**单测已完成**；acados SQP-RTI 已接线，缺依赖时降级 Lead） | 1.4× 下跟踪 p95 优于 `LeadTracker`；求解超时率 < 1% |
| F5 | L4：`ContactMonitor`；尝试 κ=1.5 | 抓取成功率不低于 F3；交接 jerk 不恶化 |

## 6. 测试要点（tests/deploy）

- `PlanBuffer`：非均匀 `step_dt` 下 progress 推进；`prefix` 按模型步；晚到结果丢弃；\(\hat d\) 与实际 \(d\) 在 mock 延迟下误差 ≤ 1 步。
- `InferenceLoop`（mock 延迟 + mock_sim）：κ 1.0 / 1.4、延迟 60 / 120 ms 下无空缓冲；结果早到时也不出现停顿。
- `TimeAxisQP`：路点不变；约束满足；active-set 与 L-BFGS-B（及 OSQP，若装）一致；用老栈 `tests/test_arm_timeline.py` 的用例做行为对照。
- `LeadTracker`：与老栈 `robot_async_buffer._apply_lag_compensation` 在同一输入序列上逐拍一致（可把老栈函数输出存成 fixture npz）。
- `agibot_g2`：用假 SDK 对象覆盖 §2 每条拒绝路径（健康字段、模式变化、名字不匹配、时间戳回退、超时不重试、限位）；用例文件名带 `g2`（如 `test_g2_layout.py` / 后续 `test_g2_agibot.py`），其它本体另开 `test_<embodiment>_*.py`。
- 端到端：mock 后端 + mock 平台 + RTC + 重定时，跑 60 s 无越界、无停顿。

## 7. 真机运行约定

- 部署目录：`agi@g2s:/data/project_r/fastervla/`（与老栈目录分开；共用 `/home/agi/app/gdk` 与 5090 推理服务）。
- 同一时刻只允许一个栈连机器人；入口脚本启动前检查老栈进程不存在。
- 默认 `shadow`；`execute` 需显式 `--mode execute` + 交互 TTY。
- 运行产物：`results/robot-live-<id>/`，字段见 [trace-design.md](trace-design.md) §6（及本文 4.7）；跑完给出 run id，真机分析由人工 / `tools/jitter` / 老栈 chat 对照。

## 8. 给实现 chat 的起步清单

1. 读 `docs/design.md`、本文、`AGENTS.md`；把 AGENTS 的「deploy 未开工」边界改为按本文分期开放 `deploy/`、`core/`、`tests/deploy/`、`configs/deploy|embodiment/`。
2. F0：先提交测试（`tests/core`、`tests/deploy` 的 F0 部分）等 review，再实现。
3. 读老栈对应文件只为对齐行为：`robot_control.py`（G2 IO / 安全）、`robot_gdk_capture.py`（相机）、`robot_bridge.py`（RTC payload、推理线程）、`robot_async_buffer.py` + `robot_velocity_blend.py`（参考生成、soft lead、交接）、`arm_timeline.py`（L2）。
4. 不改 `g2_tools/h7_deploy/`，不改 5090 推理服务。
