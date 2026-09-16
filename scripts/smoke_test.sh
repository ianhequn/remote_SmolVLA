#!/usr/bin/env bash
# =============================================================================
# 全链路冒烟测试：合成数据集 → 训练几十步 → 存/读 checkpoint → 清理
#
# 用途：换机器、升级依赖、改完环境后，快速确认 SmolVLA 训练链路是否还通。
# 产物全部落在 /dev/shm（内存盘），跑完自动删除，不碰 NFS 家目录。
#
# 用法:
#   ./scripts/smoke_test.sh            # 跑完自动清理
#   ./scripts/smoke_test.sh --keep     # 保留产物以便排查
#
# 注意：合成数据是随机噪声，只验证「代码路径通不通」，不能评估模型效果。
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ENV="${LEROBOT_ENV:-/home/zyy00005602/.conda/envs/lerobot}"
SMOKE_DIR="${SMOKE_DIR:-/dev/shm/lerobot_smoke}"
STEPS="${STEPS:-20}"
KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

cleanup() {
    if [[ "${KEEP}" == "1" ]]; then
        echo "产物保留在: ${SMOKE_DIR}"
    else
        rm -rf "${SMOKE_DIR}"
        echo "已清理 ${SMOKE_DIR}"
    fi
}
trap cleanup EXIT

source /opt/conda/etc/profile.d/conda.sh
conda activate "${LEROBOT_ENV}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/zyy00005602/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "${TMPDIR}" "${SMOKE_DIR}"

echo "### 1/3 环境自检"
python "${REPO_ROOT}/scripts/verify_env.py" --quick

echo
echo "### 2/3 生成合成数据集 (${SMOKE_DIR}/dummy_armpi)"
python "${REPO_ROOT}/scripts/make_dummy_dataset.py" \
    --root "${SMOKE_DIR}/dummy_armpi" --force

echo
echo "### 3/3 训练 ${STEPS} 步"
LEROBOT_DATASETS_DIR="${SMOKE_DIR}" \
LEROBOT_RUNS_DIR="${SMOKE_DIR}/runs" \
LEROBOT_RUN_NAME="dummy_armpi_smoke" \
LEROBOT_SAVE_FREQ="${STEPS}" \
LEROBOT_LOG_FREQ=5 \
    "${REPO_ROOT}/scripts/train_armpi.sh" dummy_armpi "${STEPS}"

CKPT="${SMOKE_DIR}/runs/dummy_armpi_smoke/checkpoints/last/pretrained_model"
echo
echo "### 回读 checkpoint: ${CKPT}"
python - "${CKPT}" <<'PY'
import sys, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
p = SmolVLAPolicy.from_pretrained(sys.argv[1])
print(f"  ✅ 加载成功, action dim = {p.config.action_feature.shape}, "
      f"可训练 {sum(x.numel() for x in p.parameters() if x.requires_grad)/1e6:.1f}M")
PY

echo
echo "✅ 冒烟测试通过：数据集读取 → 视频解码 → forward/backward → 存/读 checkpoint 全部正常"
