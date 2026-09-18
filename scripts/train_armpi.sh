#!/usr/bin/env bash
# =============================================================================
# SmolVLA 微调启动脚本 —— ArmPi Ultra（1 路 RGB 相机 + 6 轴）
#
# 用法:  ./scripts/train_armpi.sh <数据集目录名> [训练步数] [其它 lerobot-train 参数]
# 详情:  ./scripts/train_armpi.sh --help
#
# 本脚本负责在调用 lerobot-train 之前把三件容易踩的事处理掉：
#   1. 用 lerobot-train-bf16 包装器（0.6.1 的 autocast 推断成 fp16 会和 bf16 梯度冲突）
#   2. 导出 HF_ENDPOINT 等环境变量（huggingface.co 被墙，脚本常不在登录 shell 里跑）
#   3. 按数据集实际的相机名自动生成 --rename_map
#      （SmolVLA 底座的 image_features 固定是 camera1/2/3，名字不匹配会直接报错）
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

usage() {
    cat <<'EOF'
用法:
  ./scripts/train_armpi.sh <数据集目录名> [训练步数] [其它 lerobot-train 参数]

例:
  ./scripts/train_armpi.sh armpi_train_v1                      # 默认 30000 步
  ./scripts/train_armpi.sh armpi_train_v1 20000
  ./scripts/train_armpi.sh armpi_train_v1 20000 --resume=true  # 见下方「断点续训」
  ./scripts/train_armpi.sh armpi_train_v1 20000 --policy.train_expert_only=false

数据集放在 ${LEROBOT_DATASETS_DIR}/<数据集目录名>/

可用环境变量覆盖默认值:
  LEROBOT_ENV            conda 环境前缀    默认 /home/zyy00005602/.conda/envs/lerobot
  LEROBOT_DATASETS_DIR   数据集根目录      默认 /home/zyy00005602/MyWork/Lerobot/dataset
  LEROBOT_RUNS_DIR       训练输出根目录    默认 <repo>/outputs
  LEROBOT_BATCH_SIZE     批大小            默认 32（1 相机 batch 64 峰值 8.7 GiB）
  LEROBOT_NUM_WORKERS    dataloader 进程数 默认 4（容器 12 核）
  LEROBOT_SAVE_FREQ      存档间隔          默认 5000
  LEROBOT_LOG_FREQ       日志间隔          默认 50
  LEROBOT_RUN_NAME       运行名，决定输出目录名

相机名不匹配时脚本会自动补 --rename_map（数据集相机按名排序，第 i 个 -> camera{i+1}）。
详见 docs/design.md「相机命名与 rename_map」。

断点续训: 必须复用同一个运行名，让输出目录指向原目录（脚本已经设好 --output_dir，
不要再手动传，否则会和脚本的参数冲突）:
  LEROBOT_RUN_NAME=<原运行名> ./scripts/train_armpi.sh <数据集名> <步数> --resume=true
EOF
    exit 1
}

[[ $# -ge 1 && "$1" != "-h" && "$1" != "--help" ]] || usage
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

# ------------------------------ 元数据 + 相机映射 ---------------------------
# SmolVLA 底座的 config.json 把 image_features 写死为 camera1/2/3（empty_cameras=0，
# 缺失的相机不会被补占位，只喂实际存在的那几路）。数据集里的相机名必须落在这个集合内：
#   · 训练开始前 validate_visual_features_consistency() 会做子集校验，不匹配直接抛
#     ValueError: Feature mismatch
#   · 运行时 SmolVLA.prepare_images() 按 key 匹配 config.image_features，认不出来就是
#     ValueError: All image features are missing from the batch
# 所以这里读 meta/info.json，把数据集相机按名字排序后第 i 个映射成 camera{i+1}。
# 已经合规（名字全在 camera1/2/3 里）就不生成任何参数，部署侧也不需要映射。
# 详见 docs/design.md「相机命名与 rename_map」。
META="$(python - "${DATASET_ROOT}" <<'PY'
import json
import sys
import pathlib

root = pathlib.Path(sys.argv[1])
meta = json.loads((root / "meta" / "info.json").read_text())
feats = meta.get("features", {})

POLICY_CAMS = {f"observation.images.camera{i}" for i in (1, 2, 3)}
ds_cams = sorted(k for k in feats if k.startswith("observation.images."))

if not ds_cams:
    sys.stderr.write(f"错误: {root} 的 features 里没有任何 observation.images.* 相机\n")
    sys.exit(2)

rename = {}
# 与 validate_visual_features_consistency 的判定保持一致：数据集是 policy 的子集就放行
if not set(ds_cams) <= POLICY_CAMS:
    if len(ds_cams) > 3:
        sys.stderr.write(
            f"错误: 数据集有 {len(ds_cams)} 路相机 {ds_cams}，"
            "SmolVLA 底座只支持 camera1/2/3 三路，无法自动映射。\n"
        )
        sys.exit(2)
    rename = {old: f"observation.images.camera{i + 1}" for i, old in enumerate(ds_cams)}

print("|".join([
    str(meta.get("codebase_version", "?")),
    str(meta.get("robot_type", "?")),
    str(meta.get("total_episodes", "?")),
    str(meta.get("total_frames", "?")),
    str(meta.get("fps", "?")),
    str(len(feats)),
    ",".join(ds_cams),
    json.dumps(rename, separators=(",", ":")),
]))
PY
)"
IFS='|' read -r CBVER ROBOT EPS FRAMES FPS NFEAT CAM_LIST RENAME_JSON <<<"${META}"

RENAME_ARGS=()
RENAME_NOTE="不需要（相机名已经是 camera1/2/3 之一）"
if [[ "${RENAME_JSON}" != "{}" ]]; then
    RENAME_ARGS=(--rename_map="${RENAME_JSON}")
    RENAME_NOTE="自动生成 ⚠ 部署侧必须用同一套映射"
fi
# 用户显式传了 --rename_map 就尊重用户的，不叠加
for _arg in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
    if [[ "${_arg}" == --rename_map* ]]; then
        RENAME_ARGS=()
        RENAME_NOTE="命令行指定（已覆盖自动生成）"
    fi
done

RUN_NAME="${LEROBOT_RUN_NAME:-${DATASET_NAME}_smolvla_$(date +%m%d_%H%M)}"
OUT_DIR="${LEROBOT_RUNS_DIR}/${RUN_NAME}"

echo "=============================================================="
echo " 数据集   : ${DATASET_ROOT}"
echo "            版本=${CBVER}  机器人=${ROBOT}  episodes=${EPS}  frames=${FRAMES}  fps=${FPS}"
echo " 相机     : ${CAM_LIST}"
echo "            映射=${RENAME_JSON}"
echo "            ${RENAME_NOTE}"
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
  ${RENAME_ARGS[@]+"${RENAME_ARGS[@]}"} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "训练结束。"
echo "  checkpoint : ${OUT_DIR}/checkpoints/last/pretrained_model"
if [[ "${RENAME_JSON}" != "{}" ]]; then
    echo "  相机映射   : ${RENAME_JSON}"
    echo "               映射已随 checkpoint 里的 train_config.json 一起存下（checkpoints/*/pretrained_model/）。"
fi
echo "  断点续训   : LEROBOT_RUN_NAME=${RUN_NAME} ./scripts/train_armpi.sh ${DATASET_NAME} ${STEPS} --resume=true"
