# fasterVLA

比示教更快的 VLA：**训练侧**按关节速度约束跳帧组 action chunk；**部署侧**走 Realtime-VLA V2 执行栈（后续）。

设计见 [docs/design.md](docs/design.md)。Agent 约定见 [AGENTS.md](AGENTS.md)。

## 现状（训练）

已打通：

`recipe YAML` → `delta_timestamps` 拉窗 → `VelocitySkipActions` → openpi JAX debug 训 N 步

| 模块 | 路径 |
|------|------|
| 跳帧算法 | `src/fastervla/training/resample/velocity_skip.py` |
| `v_max` YAML | `src/fastervla/training/data/limits.py` |
| 数据钩子 | `src/fastervla/training/policy/openpi_bridge.py` |
| Dataset 组装 | `src/fastervla/training/policy/dataset_assembly.py` / `create_dataset.py` |
| 配方 / CLI | `train_recipe.py` · `cli.py` · `openpi_jax_backend.py` |

## 安装

```bash
cd fasterVLA
pip install -e ".[dev]"
# 真训还需要本机 openpi（含 JAX），例如：
#   /path/to/openpi/.venv/bin/python ...
```

## 用法

### 1. Dry-run（不依赖 openpi）

```bash
PYTHONPATH=src python -m fastervla.training.cli train \
  --recipe configs/training/pi05_velocity_skip_debug.yaml \
  --checkpoint-dir /tmp/fastervla_ckpt \
  --dry-run
```

或安装后：

```bash
fastervla-train train \
  --recipe configs/training/pi05_velocity_skip_debug.yaml \
  --checkpoint-dir /tmp/fastervla_ckpt \
  --dry-run
```

### 2. Debug 真训（openpi FakeDataset，`window == horizon`）

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src \
  /path/to/openpi/.venv/bin/python -m fastervla.training.cli train \
  --recipe configs/training/pi05_velocity_skip_debug.yaml \
  --checkpoint-dir /tmp/fastervla_ckpt
```

示例配方：`configs/training/pi05_velocity_skip_debug.yaml`（`repo_id: fake`，`horizon=window=50`）。

### 3. 真实 LeRobot + 长窗跳帧

1. 复制并改 `configs/training/pi05_velocity_skip_lerobot.yaml`：
   - `repo_id` → 本地 / Hub 数据集
   - `window` > `horizon`（否则跳帧没有搜索空间）
   - `limits_path` 关节维与 action 维一致
2. 在装有 `lerobot` + openpi 的环境中去掉 `--dry-run` 启动训练。

`repo_id: fake` 且 `window > horizon` 会被拒绝（FakeDataset 拉不出长窗）。

### 配方字段

| 字段 | 含义 |
|------|------|
| `openpi_config` | openpi 配置名（如 `debug_pi05`） |
| `skip_frames` | 是否速度跳帧 |
| `horizon` / `window` | 输出 H 步 / 名义拉窗长度 |
| `fps` | 名义时间戳步长 `1/fps` |
| `limits_path` | 关节 `v_max` YAML |
| `repo_id` | `fake` 或 LeRobot id |
| `num_steps` | debug 跑几步 |

速度：\(\Delta t = 1/H\)；连续 H 次最远合法跳。详情见 design §6。

## 测试

```bash
# 默认环境（跳过需 openpi 的用例）
PYTHONPATH=src python -m pytest tests/training -m "not openpi"

# openpi 烟测
JAX_PLATFORMS=cpu PYTHONPATH=src \
  /path/to/openpi/.venv/bin/python -m pytest tests/training -m openpi
```

## 布局

```text
configs/training/     # recipe + joint_velocity_limits
src/fastervla/
  training/           # 数据、跳帧、openpi 桥、CLI
  # deploy/           # 后续
docs/design.md
tests/training/
```
