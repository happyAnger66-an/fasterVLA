# fasterVLA · h7 部署

目录：`/data/project_r/fastervla`（与老栈 `agibotg2_pi0.5_deploy` **并列**，互不修改）。

## 共用 / 不共用

| 项 | 说明 |
|---|---|
| 推理服务 | **共用** h7 上现成 apex websocket `:8013`（不重启、不改 serve） |
| GDK | **共用** `/home/agi/app/gdk`（只读环境变量） |
| Python venv | **本树独立** `.venv`，不碰老栈 `.venv` |
| openpi-client | 首次 bootstrap **复制**到本树 `vendor/openpi-client`（只读源） |

## 一次安装

```bash
cd /data/project_r/fastervla
bash scripts/h7/bootstrap_venv.sh
source scripts/h7/env.sh
```

## 运行（配方 YAML）

```bash
# 先确认推理在听
ss -lntp | grep 8013
source scripts/h7/env.sh

# 默认工作点（2026-11-25）：κ=1.4 Lead2 + 控制护栏
fastervla-deploy run --config configs/deploy/h7_k1p4_qp_phase_lead2_ctrl.yaml \
  --mode execute --duration-s 180 --tag M4b_ctrl

# 正常倍速 soft shadow：κ=1 + 4+4 + ZP@3 + pchip
fastervla-deploy run --config configs/deploy/h7_soft_1x.yaml

# execute 示例（需 TTY；可用 CONFIG= 覆盖）
CONFIG=configs/deploy/h7_k1p4_qp_phase_lead2_ctrl.yaml bash scripts/h7/run_execute_example.sh
```


裸连通性冒烟（无 RTC）仍可用 `bash scripts/h7/run_shadow_example.sh`。

## κ=1.1 扫参（retimer × p/b）

对照基线 `docs/baseline-h7-soft-1x.json`。建议顺序（每场 execute ~60s）：

```bash
source scripts/h7/env.sh
# 1) uniform 4+4（最接近基线）
fastervla-deploy run --config configs/deploy/h7_k1p1_uniform_4x4.yaml --mode execute
# 2) uniform 8+8（少合块）
fastervla-deploy run --config configs/deploy/h7_k1p1_uniform_8x8.yaml --mode execute
# 3) QP 4+4
fastervla-deploy run --config configs/deploy/h7_k1p1_qp_4x4.yaml --mode execute
# 4) QP 8+8
fastervla-deploy run --config configs/deploy/h7_k1p1_qp_8x8.yaml --mode execute
```

跑完后：`fastervla-deploy analyze --run results/robot-live-<id>`，看 progress_wall / 合块Hz / 跟踪 / 交接跳 / empty。

h7 默认机参（可用环境变量 / CLI 覆盖）：

- `host=127.0.0.1` `port=8013`
- `robot_aid=G2A0205C800352`
- `sdk_source_ip=192.168.112.212`
- `discovery_uri=http://192.168.112.212:2379`

## 从开发机同步代码

```bash
rsync -az --delete \
  --exclude '.venv' --exclude 'results' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude 'vendor' \
  ./ agi@h7:/data/project_r/fastervla/
```

同步后若只改了 Python，一般不用重跑 bootstrap；依赖变更时再 `bash scripts/h7/bootstrap_venv.sh`。
