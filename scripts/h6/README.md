# fasterVLA · h6 部署

**全新目录**（不改动老栈）：

| 项 | 路径 |
|---|---|
| fasterVLA | `/data/project_r/fastervla-h6` |
| Apex 推理 | `/data/project_r/pi05-apex-h6` |
| Docker 容器 | `pi05-apex-h6` |
| 推理端口 | `:8013`（RTC） |

老栈 `/data/project_r/agibotg2_pi0.5_deploy` 只读引用（openpi-client 可拷贝进本树）。

## 一次安装

```bash
cd /data/project_r/fastervla-h6
bash scripts/h6/bootstrap_venv.sh
source scripts/h6/env.sh
```

## Acados MPC（可从 h7 拷贝）

h7 / h6 同为 aarch64，可走 `192.168.5.6` → `192.168.5.5` 快路径拷贝 C 库，再在本机 venv 装 Python 包：

```bash
# 在 h6 上执行
mkdir -p /home/agi/deps
rsync -a agi@192.168.5.6:/home/agi/deps/acados /home/agi/deps/
rsync -a agi@192.168.5.6:/home/agi/deps/acados_env.sh /home/agi/deps/
cd /data/project_r/fastervla-h6 && source scripts/h6/env.sh && source /home/agi/deps/acados_env.sh
pip install -q "casadi>=3.6" "matplotlib>=3.9"
pip install -q -e /home/agi/deps/acados/interfaces/acados_template
# dry-run：编译 OCP + 解一步（应打印 MPC_READY / ok）
python -c "from fastervla.deploy.client.tracker.arm_mpc_acados import acados_importable; print(acados_importable())"
```

`scripts/h6/env.sh` 会自动 `source /home/agi/deps/acados_env.sh`（若存在）。

## 起推理（host）

```bash
CONTAINER=pi05-apex-h6 \
HOST_SERVE=/data/project_r/pi05-apex-h6/serve \
HOST_MODEL_ROOT=/data/project_r/pi05-apex-h6/model \
HOST_LOG=/data/project_r/pi05-apex-h6/logs/serve_apex_fast60_nvfp4_ep0_rtc_ws.log \
MODEL=/work/model/fast60_nvfp4_ep0 \
PORT=8013 RTC=1 IMAGE_DECODE=gpu \
  bash /data/project_r/pi05-apex-h6/serve/run_apexinfer_h7.sh
```

## 跑配方（h6 专用，机参已写进 yaml）

```bash
source scripts/h6/env.sh

# 默认工作点（Lead2 护栏 @1.4）
fastervla-deploy run --config configs/deploy/h6_k1p4_qp_phase_lead2_ctrl.yaml \
  --mode execute --duration-s 180 --tag M4b_re

# prefix=0
fastervla-deploy run --config configs/deploy/h6_k1p4_qp_phase_lead2_ctrl_0x4.yaml \
  --mode execute --duration-s 180 --tag M5_p0

# MPC 真实预览（需 acados；否则降级 Lead）
fastervla-deploy run --config configs/deploy/h6_k1p4_qp_phase_mpc_ctrl_preview.yaml \
  --mode execute --duration-s 180 --tag M5_MPC_plan
```

机参默认：`robot_aid=G2A0205C900081`，`sdk_source_ip` / `discovery=10.42.1.101`，推理 `127.0.0.1:8013`（容器 `pi05-apex-h6`）。

## 机器复位（羽毛球捡球初始姿态）

与 h7 等价：

```bash
bash /data/project_r/agibotg2_pi0.5_deploy/scripts/run_reset_pose.sh \
  --config configs/badminton651/h6_89999_sync.json \
  --pose results/saved_poses/h6-20261106-150648.json \
  --execute
```

本树入口（姿态/配置已拷入 `fastervla-h6`；复位 SDK=`10.42.1.101`/`xfi0.60`）：

```bash
cd /data/project_r/fastervla-h6
bash scripts/h6/run_reset.sh              # 只读规划
bash scripts/h6/run_reset.sh --execute    # 真机 → 羽毛球捡球姿态
bash scripts/h6/run_reset_arms.sh --execute   # 仅手臂到该姿态
bash scripts/h6/run_reset_fullbody.sh --execute  # 旧全身 h6_reset_pose
```
