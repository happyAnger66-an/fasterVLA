# AGENTS.md — fasterVLA

给编码 Agent 的仓库约定。人类向用法见 [README.md](README.md)；架构见 [docs/design.md](docs/design.md)；G2 部署分期见 [docs/deploy-g2-plan.md](docs/deploy-g2-plan.md)；**κ / 约束配置**见 [docs/deploy-config.md](docs/deploy-config.md)；Trace / Metric 见 [docs/trace-design.md](docs/trace-design.md)。

## 边界

- **训练（可写）**：`src/fastervla/training/**`、`configs/training/**`、`tests/training/**`、本文件与 README。
- **部署（按 `deploy-g2-plan` 分期开放，先测后码）**：
  - **F0（已完成）**：`core/**`、`deploy/backends/mock.py`、`deploy/client/io/{interfaces,layout,mock_sim}.py`、`deploy/client/loops.py`（同步 1×）、对应 tests。
  - **F1（已完成）**：`agibot_g2` / `gdk_real` / `openpi_ws` / `runtime` / `deploy.cli shadow`；g2s 真机 shadow 5 min 已跑通（右夹爪健康暂跳过）。
  - **F2（已完成单测）**：`plan_buffer.py`、`inference_loop.py`、`postprocess/retime.UniformRetimer`、`tracker/lead.py`；对应 tests 全绿。真机 execute 1.0× 对标仍待 g2s。
  - **F3（已完成单测）**：`postprocess/retime.TimeAxisQP`（+ `box_qp` / `rate_hessian`）；`tests/deploy/test_time_axis_qp.py` 全绿；`run --retimer qp` 可选用。真机 κ=1.2/1.4 对标仍待 g2s。勿提前写 MPC / Contact（F4+）。
  - **Trace + 离线分析（本仓闭环）**：`recorder.py`；`deploy/analysis/`（hf3/hf7、跟踪、r2r，**不依赖** `tools/jitter` / h7_deploy）；`fastervla-deploy shadow|run|analyze`。见 [docs/trace-design.md](docs/trace-design.md)。
  - **F4（下期·先测后码）**：`LagStateEstimator` + `AcadosMpcTracker`；打开范围以 `deploy-g2-plan` 为准。未到该期前不要实现。
- **禁止改动**：`r-infra/g2_tools/h7_deploy/**`；5090 上已有 openpi websocket 推理服务（只写本仓 client / mock backend）。
- **不 vendor openpi / 老栈**：训练走 `openpi_jax_backend`；部署算法按新接口重写，可对照老栈行为单测。

## 训练数据流（必须保持）

```text
TrainRecipe YAML
  → load_train_recipe / build_dataset_plan
  → delta_timestamps 长度 = window（名义 k/fps）
  → create_dataset_from_plan 或 FakeDataset
  → VelocitySkipActions（H 次最远合法跳，Δt=1/H）
  → openpi transform + train_step
```

- `skip_frames: false` → 连续 `window[:H]`，与标准 pi05 一致。
- `repo_id: fake` 且 `window > horizon` → **必须**在进 openpi 前 `validate_dataset_plan_for_openpi_fake` 失败。
- 真 LeRobot：`window > horizon`，`limits` 维数 = action 维。

## 开发约定

1. **先测试后编码**（新功能 / 每期 F*）：只提交 `tests/` → 等人 review → 再写实现。
2. **修 bug** 必须带能复现原失败路径的回归测试。
3. 对话可用中文；**提交信息 / PR / 面向上游的说明用英文**（若无上游要求则与现有 commit 风格一致）。
4. 一刀一功能一 commit；不要把无关改动捆进同一笔。
5. 用户可见 CLI / 配方字段变更时，同步改 README（本文件只写 Agent 约束）。

## 常用命令

```bash
# 训练单测（无 GPU）
PYTHONPATH=src python -m pytest tests/training -m "not openpi"

# F0–F3 部署 / core 单测（应全绿；osqp 未装时相关用例 skip）
PYTHONPATH=src python -m pytest tests/core tests/deploy -q

# dry-run
PYTHONPATH=src python -m fastervla.training.cli train \
  --recipe configs/training/pi05_velocity_skip_debug.yaml \
  --checkpoint-dir /tmp/ckpt --dry-run

# debug 真训（需 openpi venv）
JAX_PLATFORMS=cpu PYTHONPATH=src \
  /path/to/openpi/.venv/bin/python -m fastervla.training.cli train \
  --recipe configs/training/pi05_velocity_skip_debug.yaml \
  --checkpoint-dir /tmp/ckpt
```

入口：`fastervla.training.cli:main`（console script：`fastervla-train`）。

## 关键文件

| 路径 | 角色 |
|------|------|
| `training/resample/velocity_skip.py` | 跳帧核心 |
| `training/data/limits.py` | YAML `v_max` |
| `training/policy/openpi_bridge.py` | `VelocitySkipConfig` / Actions / 单步协议 |
| `training/policy/dataset_plan.py` | Recipe → 拉窗计划 |
| `training/policy/create_dataset.py` | Plan → Dataset（可注入 factory） |
| `training/policy/openpi_jax_backend.py` | openpi debug 训 N 步 |
| `training/cli.py` | 训练 CLI |
| `configs/training/*.yaml` | 训练示例配方 |
| `docs/deploy-g2-plan.md` | G2 部署硬约束与 F0–F5 |
| `core/` · `deploy/` | 见 design §4–5；按分期开放 |

## 不要做

- 不要在默认 CI 环境硬依赖 openpi/JAX；烟测标 `@pytest.mark.openpi`。
- 不要把 FakeDataset 当成能提供 `window > H` 的数据源。
- 不要复制一份 openpi 前向/优化器到本仓库。
- 不要改 `g2_tools/h7_deploy` 或 5090 推理服务来「省事对齐」。
- 不要跳过当期测试 review 直接铺 F1+ 实现。
