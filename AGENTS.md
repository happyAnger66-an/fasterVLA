# AGENTS.md — fasterVLA

给编码 Agent 的仓库约定。人类向用法见 [README.md](README.md)；架构见 [docs/design.md](docs/design.md)。

## 边界

- **本期可写**：`src/fastervla/training/**`、`configs/training/**`、`tests/training/**`、本文件与 README。
- **部署 `deploy/`**：未开工前不要大面积铺代码；接口以 design 为准。
- **不 vendor openpi**：通过 `openpi_jax_backend` 调本机 openpi；梯度与 train_step 以 openpi 为准。

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

1. **先测试后编码**（新功能）：只提交 `tests/` → 等人 review → 再写实现。
2. **修 bug** 必须带能复现原失败路径的回归测试。
3. 对话可用中文；**提交信息 / PR / 面向上游的说明用英文**（若无上游要求则与现有 commit 风格一致）。
4. 一刀一功能一 commit；不要把无关改动捆进同一笔。
5. 用户可见 CLI / 配方字段变更时，同步改 README（本文件只写 Agent 约束）。

## 常用命令

```bash
# 单测（无 GPU）
PYTHONPATH=src python -m pytest tests/training -m "not openpi"

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
| `training/cli.py` | CLI |
| `configs/training/*.yaml` | 示例配方 |

## 不要做

- 不要在默认 CI 环境硬依赖 openpi/JAX；烟测标 `@pytest.mark.openpi`。
- 不要把 FakeDataset 当成能提供 `window > H` 的数据源。
- 不要复制一份 openpi 前向/优化器到本仓库。
