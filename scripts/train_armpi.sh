#!/usr/bin/env bash
# =============================================================================
# SmolVLA 微调启动脚本 —— ArmPi Ultra（1 路 RGB 相机 + 6 轴）
#
# 用法:
#   ./scripts/train_armpi.sh <数据集目录名> [训练步数] [-- 其它 lerobot-train 参数]
#
# 例:
#   ./scripts/train_armpi.sh pick_cube              # 默认 30000 步
#   ./scripts/train_armpi.sh pick_cube 50000
#   ./scripts/train_armpi.sh pick_cube 30000 --resume=true
#   ./scripts/train_armpi.sh pick_cube 30000 --batch_size=64 --num_workers=6
#
# 数据集放在 ${LEROBOT_DATASETS_DIR}/<数据集目录名>/
# 可从本地录制机器 rsync/scp 上来的 LeRobotDataset 目录。
#
# 可用环境变量覆盖默认值:
#   LEROBOT_ENV            conda 环境前缀   默认 /home/zyy00005602/.conda/envs/lerobot
#   LEROBOT_DATASETS_DIR   数据集根目录     默认 /home/zyy00005602/MyWork/Lerobot/dataset
#   LEROBOT_RUNS_DIR       训练输出根目录   默认 <repo>/outputs
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LEROBOT_ENV="${LEROBOT_ENV:-/home/zyy00005602/.conda/envs/lerobot}"
LEROBOT_DATASETS_DIR="${LEROBOT_DATASETS_DIR:-/home/zyy00005602/MyWork/Lerobot/dataset}"
LEROBOT_RUNS_DIR="${LEROBOT_RUNS_DIR:-${REPO_ROOT}/outputs}"

# 依据本机实测（1 相机、bf16、chunk_size=50）: batch 32 峰值 ~5.0 GiB, 64 峰值 ~8.7 GiB
BATCH_SIZE="${LEROBOT_BATCH_SIZE:-32}"
NUM_WORKERS="${LEROBOT_NUM_WORKERS:-4}"        # 容器只有 12 核，建议 4~6
SAVE_FREQ="${LEROBOT_SAVE_FREQ:-5000}"
LOG_FREQ="${LEROBOT_LOG_FREQ:-50}"

usage() { sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

[[ $# -ge 1 ]] || usage
DATASET_NAME="$1"; shift
STEPS="${1:-30000}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")   # 其余参数原样透传给 lerobot-train

DATASET_ROOT="${LEROBOT_DATASETS_DIR}/${DATASET_NAME}"

# ------------------------------ 预检查 --------------------------------------
if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "错误: 找不到数据集目录 ${DATASET_ROOT}" >&2
    echo "      先把录制好的数据集传到那里（rsync/scp），目录名就是第一个参数。" >&2
    exit 1
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "错误: ${DATASET_ROOT} 不像是一个 LeRobotDataset（缺少 meta/info.json）" >&2
    exit 1
fi
if [[ ! -x "${LEROBOT_ENV}/bin/lerobot-train-bf16" ]]; then
    echo "错误: 找不到 ${LEROBOT_ENV}/bin/lerobot-train-bf16" >&2
    echo "      见 docs/environment.md「坑 4」——必须用它代替 lerobot-train。" >&2
    exit 1
fi

# ------------------------------ 环境 ----------------------------------------
source /opt/conda/etc/profile.d/conda.sh
conda activate "${LEROBOT_ENV}"

# 脚本通常不在登录 shell 里运行，不能依赖 ~/.bashrc
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # huggingface.co 被墙
export HF_HOME="${HF_HOME:-/home/zyy00005602/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp}"                      # NFS 上 multiprocessing 清理会报 busy
export WANDB_MODE="${WANDB_MODE:-offline}"                   # 外网不可达
mkdir -p "${TMPDIR}"

# ------------------------------ 信息 ----------------------------------------
META="$(python - "${DATASET_ROOT}" <<'PY'
import json, sys, pathlib
d = json.loads(pathlib.Path(sys.argv[1], "meta", "info.json").read_text())
print(f"{d.get('codebase_version','?')}|{d.get('robot_type','?')}|{d.get('total_episodes','?')}|{d.get('total_frames','?')}|{d.get('fps','?')}|{len(d.get('features',{}))}")
PY
)"
IFS='|' read -r CBVER ROBOT EPS FRAMES FPS NFEAT <<<"${META}"

RUN_NAME="${LEROBOT_RUN_NAME:-${DATASET_NAME}_smolvla_$(date +%m%d_%H%M)}"
OUT_DIR="${LEROBOT_RUNS_DIR}/${RUN_NAME}"

echo "=============================================================="
echo " 数据集   : ${DATASET_ROOT}"
echo "            版本=${CBVER}  机器人=${ROBOT}  episodes=${EPS}  frames=${FRAMES}  fps=${FPS}"
echo " 输出     : ${OUT_DIR}"
echo " 训练     : steps=${STEPS}  batch=${BATCH_SIZE}  workers=${NUM_WORKERS}"
echo " base 权重: lerobot/smolvla_base（从 hf-mirror 拉取）"
echo "=============================================================="
echo

# ------------------------------ 训练 ----------------------------------------
# 必须用 lerobot-train-bf16：lerobot 0.6.1 用 torch.get_autocast_dtype('cuda') 推断
# mixed_precision，PyTorch 默认给 float16，与 SmolVLA 的 bf16 梯度冲突直接崩。
# 详见 docs/environment.md「坑 4」。
lerobot-train-bf16 \
  --dataset.repo_id="local/${DATASET_NAME}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="local/${RUN_NAME}" \
  --policy.push_to_hub=false \
  --policy.use_amp=true \
  --policy.device=cuda \
  --output_dir="${OUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --eval_steps=0 \
  --env_eval_freq=0 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "训练结束。"
echo "  checkpoint : ${OUT_DIR}/checkpoints/last/pretrained_model"
echo "  断点续训   : ./scripts/train_armpi.sh ${DATASET_NAME} ${STEPS} --resume=true --output_dir=${OUT_DIR}"
