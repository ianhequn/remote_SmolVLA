# SmolVLA on GDUT HPC (`n1`)

在 GDUT 高性能计算平台的 JupyterLab(GPU) 2.0 实例上微调 **SmolVLA**，用于 **ArmPi Ultra** 六轴机械臂（1 路 RGB 相机）。

本机只负责**训练**：数据在本地录制机器上采集，上传到本机 NFS 后进行微调。

```
┌──────────────────────────────┐
│ 本地录制机 (ArmPi Ultra)      │
│   lerobot-record             │
└───────────────┬──────────────┘
                │  LeRobotDataset v3
                │  rsync / scp（不走 HF）
                ▼
   ~/MyWork/Lerobot/dataset/<名称>/
                │
                ▼
┌──────────────────────────────┐
│ GDUT HPC n1  (1×RTX3090)     │
│   check_dataset.py → 训练     │
└──────────────────────────────┘
```

---

## 快速开始

```bash
# 0) 激活环境（脚本内部也会自己激活，这里是为了交互式使用）
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/zyy00005602/.conda/envs/lerobot
cd /home/zyy00005602/MyWork/Lerobot/remote_SmolVLA

# 1) 环境自检（约 1 分钟，首次会拉权重）
python scripts/verify_env.py

# 2) 全链路冒烟测试（用合成假数据，约 1 分钟，跑完自动清理）
./scripts/smoke_test.sh

# 3) 数据到位后先校验，再训练
python scripts/check_dataset.py ~/MyWork/Lerobot/dataset/pick_cube --decode
./scripts/train_armpi.sh pick_cube 30000
```

数据集放在 `/home/zyy00005602/MyWork/Lerobot/dataset/<数据集目录名>/`，即

```bash
./scripts/train_armpi.sh pick_cube 30000
# → 读取 /home/zyy00005602/MyWork/Lerobot/dataset/pick_cube/
# → 输出 remote_SmolVLA/outputs/pick_cube_smolvla_0916_1700/
```

---

## 脚本清单

| 脚本 | 作用 | 典型耗时 |
|---|---|---|
| `scripts/verify_env.py` | 依赖 / GPU / 权重点 / 前向 自检。`--quick` 跳过权重加载 | 5s ~ 1min |
| `scripts/smoke_test.sh` | 合成数据集 → 训练 20 步 → 存/读 checkpoint → 自动清理。换机器/升级依赖后先跑这个 | ~1min |
| `scripts/train_armpi.sh` | **正式训练入口**。参数化调用 `lerobot-train-bf16` | 数小时 |
| `scripts/bench_train_mem.py` | 扫 batch size 出显存/速度表，用来定 `batch_size` | ~3min |
| `scripts/check_dataset.py` | **校验上传上来的真实数据集**（版本/feature/完整性/抽帧解码） | 5s |
| `scripts/make_dummy_dataset.py` | 生成合成 LeRobotDataset；同时是**数据格式参考实现** | 10s |

`train_armpi.sh` 可用环境变量覆盖默认值：

```bash
LEROBOT_BATCH_SIZE=64 LEROBOT_NUM_WORKERS=6 ./scripts/train_armpi.sh pick_cube 30000
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `LEROBOT_ENV` | `/home/zyy00005602/.conda/envs/lerobot` | conda 环境前缀 |
| `LEROBOT_DATASETS_DIR` | `/home/zyy00005602/MyWork/Lerobot/dataset` | 数据集根目录 |
| `LEROBOT_RUNS_DIR` | `<repo>/outputs` | 训练输出根目录 |
| `LEROBOT_BATCH_SIZE` | `32` | 实测 1 相机峰值 ~5.0 GiB |
| `LEROBOT_NUM_WORKERS` | `4` | 容器只有 12 核 |
| `LEROBOT_SAVE_FREQ` / `LEROBOT_LOG_FREQ` | `5000` / `50` | |
| `LEROBOT_RUN_NAME` | `<数据集>_smolvla_<月日_时分>` | 覆盖输出目录名 |

---

## 环境速查

| 组件 | 版本 |
|---|---|
| conda 环境 | `/home/zyy00005602/.conda/envs/lerobot` (Python 3.12.14) |
| lerobot | 0.6.1（extras `[smolvla,training]`） |
| torch / torchvision | **2.11.0+cu128** / 0.26.0+cu128 — 驱动 570 只支持 cu128 |
| torchcodec / transformers | 0.11.1 / 5.5.4 |
| ffmpeg | 8.1.2（来自 conda `defaults`） |
| GPU | 1× RTX 3090 24G（sm_86，支持 bf16） |
| CPU / 内存 | cgroup 限 12 核 / 30 GiB |

**网络**：`huggingface.co` 被墙 → 走 `HF_ENDPOINT=https://hf-mirror.com`；`conda-forge` 被墙 → conda 只用 `defaults`。

---

## ⚠️ 三个必须知道的点

1. **训练必须用 `lerobot-train-bf16`，不能用 `lerobot-train`。**
   lerobot 0.6.1 用 `torch.get_autocast_dtype('cuda')` 推断精度，PyTorch 默认给 `float16`，
   与 SmolVLA 的 bf16 梯度冲突直接崩。包装脚本先设成 bf16 再调 lerobot。
   （模型卡上的 `--policy.dtype=bfloat16` 在 0.6.1 里**是无效参数**，别照抄。）

2. **这台机器不能接机器人。** 没有 `/dev/ttyUSB*`、`/dev/video*`，`lerobot-record` 跑不了。
   采集请在 ArmPi Ultra 本地做，详见 [docs/data-collection.md](docs/data-collection.md)。

3. **数据集别推 HuggingFace。** `hf-mirror.com` 是只读镜像不支持上传，`huggingface.co` 又被墙。
   用 rsync/scp 传到 `~/MyWork/Lerobot/dataset/`。

---

## 文档

| 文档 | 内容 |
|---|---|
| [docs/design.md](docs/design.md) | **设计文档**：数据流、约束→决策推导、精度与显存方案、失败模式集 |
| [docs/environment.md](docs/environment.md) | 环境搭建记录、依赖版本闭环、**5 个已修复的坑**及原理 |
| [docs/data-collection.md](docs/data-collection.md) | 录制侧说明：ArmPi Ultra 数据格式、编码器选择、上传与校验清单 |
| [lerobot_doc/](lerobot_doc/) | 官方参考文档（安装 / 自定义硬件接入） |

---

## 目录结构

```
Lerobot/
├── README.md                 # 本文件
├── scripts/                  # 全部可执行脚本
│   ├── train_armpi.sh        #   正式训练
│   ├── smoke_test.sh         #   全链路冒烟
│   ├── verify_env.py         #   环境自检
│   ├── check_dataset.py      #   数据集校验
│   ├── bench_train_mem.py    #   显存基准
│   └── make_dummy_dataset.py #   合成数据集 / 格式模板
├── docs/                     # 文档
├── lerobot_doc/              # 官方参考文档
└── remote_SmolVLA/           # 部署侧仓库（独立 git，本仓库忽略）
```

训练输出默认落在 `outputs/<run_name>/`（已 gitignore）：

```
outputs/<run_name>/
├── checkpoints/
│   ├── last -> 00003000/               # 最新
│   └── 00003000/pretrained_model/      # config.json + model.safetensors + 预处理器
└── train_config.json
```
