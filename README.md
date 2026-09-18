# SmolVLA on GDUT HPC (`n1`)

在 GDUT 高性能计算平台的 JupyterLab(GPU) 2.0 实例上微调 **SmolVLA**，用于 **ArmPi Ultra** 六轴机械臂（1 路 RGB 相机）。

本机只负责**训练**：数据在本地录制机器上采集，上传到本机 NFS 后进行微调，训完把 checkpoint 拷回机器人侧部署。

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
└──────────────┬───────────────┘
               │  手动拷回 checkpoint
               ▼
        ArmPi Ultra 部署
     （见 docs/deployment-notes.md）
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

# 2) 全链路冒烟测试（合成假数据，约 1 分钟，跑完自动清理）
./scripts/smoke_test.sh

# 3) 数据到位后先校验，再训练
python scripts/check_dataset.py ~/MyWork/Lerobot/dataset/armpi_train_v1 --decode
./scripts/train_armpi.sh armpi_train_v1 20000
```

数据集放在 `/home/zyy00005602/MyWork/Lerobot/dataset/<数据集目录名>/`：

```bash
LEROBOT_BATCH_SIZE=64 LEROBOT_NUM_WORKERS=6 LEROBOT_SAVE_FREQ=2000 \
  nohup ./scripts/train_armpi.sh armpi_train_v1 20000 > train_armpi_v1.log 2>&1 &
tail -f train_armpi_v1.log
```

> 训练要跑几小时，**别在 Jupyter 前台跑**，用 `nohup ... &`，否则页面一断进程就没了。

### 当前数据集 `armpi_train_v1`

| 项 | 值 |
|---|---|
| episodes / frames | 61 / 6662（均 109 帧 ≈ 4.4 s） |
| fps | 25 |
| 分辨率 / 编码 | 480×640 / AV1 (`yuv420p`) |
| state / action | 6 维 —— `joint1.pos … joint5.pos, r_joint.pos` |
| 相机 | `observation.images.front`（1 路） |
| task | 1 |

### 该跑多少步

61 集 6662 帧，`batch 64` 时每 epoch 约 104 步：

| steps | 大致耗时 | 说明 |
|---|---|---|
| 10000 | ~1.7 h | 想先看效果 |
| **20000** | **~3.3 h** | **推荐** |
| 30000 | ~5 h | 数据再多些才值得 |

⚠️ **最后一个 checkpoint 通常不是最好的。** 61 集很容易过拟合，
建议 `LEROBOT_SAVE_FREQ=2000` 多存几个，训完把 `002000 / 004000 / … /last` 都拿到真机试，挑成功率最高的。
过拟合的信号是 loss 掉到 0.0x 还在降、但真机不涨。

### 该用多大 batch

本机实测（1 相机、bf16、chunk_size=50，真 forward+backward）：

| batch | 峰值显存 | ms/step | 样本/s |
|---|---|---|---|
| 16 | 3.19 GiB | 183 | 88 |
| 32 | 5.02 GiB | 311 | 103 |
| **64** | **8.72 GiB** | **583** | **110** |
| 96 | 12.39 GiB | 870 | 110 |

吞吐从 64 起就压满了，再往上只涨显存不加速度 → **推荐 64**。
lerobot 0.6.1 **没有梯度累积**，有效 batch = `batch_size × GPU 数`（这里 1 张卡）。
想复测自己的配置：`python scripts/bench_train_mem.py --cams 1 --batches 16 32 64 96`。

---

## ⚠️ 相机命名（最容易踩的坑）

SmolVLA 底座的 `config.json` 把 `image_features` **写死为 `camera1 / camera2 / camera3`**，
而且 `empty_cameras = 0`（缺失的相机不会被补占位）。数据集里的相机名必须落在这个集合内，
否则有**两处**会炸：

| 关卡 | 报错 |
|---|---|
| 训练前校验 `validate_visual_features_consistency()` | `ValueError: Feature mismatch … Extra features: ['observation.images.front']` |
| 运行时 `SmolVLA.prepare_images()` | `ValueError: All image features are missing from the batch` |

判定规则是「数据集相机集合」和「policy 相机集合」**任意一边是另一边的子集**即放行：

| 数据集相机名 | 能否直接跑 |
|---|---|
| `camera1` / `camera2` / `camera3` | ✅ 单个也可以，不要求连号 |
| `camera1` + `camera3` | ✅ 子集 |
| **`front`** | ❌ 互不包含 → 必须映射 |
| `front` + `camera1` | ❌ 仍然互不包含 |

**`train_armpi.sh` 会自动处理**：读 `<数据集>/meta/info.json`，把相机按名字排序后第 i 个映射成
`camera{i+1}`，自动补上 `--rename_map`；已经合规就不加任何参数。

```
 相机     : observation.images.front
            映射={"observation.images.front":"observation.images.camera1"}
            自动生成 ⚠ 部署侧必须用同一套映射
```

三点注意：

1. **映射会写进 checkpoint 里的 `train_config.json`**，部署前可核对，别搞丢。
2. **部署侧必须喂 `observation.images.camera1`**（或使用同一套 `rename_map`，lerobot 的 eval 侧也支持——
   `configs/eval.py` 有 `rename_map` 字段）。详见 [docs/deployment-notes.md](docs/deployment-notes.md)。
3. **多相机时「按名字排序映射」如果和部署侧不一致，会静默把两路相机调换**——不报错，只是效果变差。
   1 路相机没有这个问题。**根治办法是在 ArmPi Ultra 的 robot 配置里直接把相机键名定义成
   `observation.images.camera1`**，以后录数据和推理都不用映射。

---

## 脚本清单

| 脚本 | 作用 | 典型耗时 |
|---|---|---|
| `scripts/verify_env.py` | 依赖 / GPU / 权重点 / 前向 自检。`--quick` 跳过权重加载 | 5s ~ 1min |
| `scripts/smoke_test.sh` | 合成数据集 → 训练 20 步 → 存/读 checkpoint → 自动清理。换机器/升级依赖后先跑这个 | ~1min |
| `scripts/train_armpi.sh` | **正式训练入口**。参数化调用 `lerobot-train-bf16`，自动处理相机映射 | 数小时 |
| `scripts/bench_train_mem.py` | 扫 batch size 出显存/速度表，用来定 `batch_size` | ~3min |
| `scripts/check_dataset.py` | **校验上传上来的真实数据集**（版本/feature/完整性/抽帧解码） | 5s |
| `scripts/make_dummy_dataset.py` | 生成合成 LeRobotDataset；同时是**数据格式参考实现** | 10s |

完整用法 `./scripts/train_armpi.sh --help`。可用环境变量覆盖：

```bash
LEROBOT_BATCH_SIZE=64 LEROBOT_NUM_WORKERS=6 ./scripts/train_armpi.sh armpi_train_v1 20000
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `LEROBOT_ENV` | `/home/zyy00005602/.conda/envs/lerobot` | conda 环境前缀 |
| `LEROBOT_DATASETS_DIR` | `/home/zyy00005602/MyWork/Lerobot/dataset` | 数据集根目录 |
| `LEROBOT_RUNS_DIR` | `<repo>/outputs` | 训练输出根目录 |
| `LEROBOT_BATCH_SIZE` | `32` | 实测 1 相机 batch 64 峰值 8.7 GiB |
| `LEROBOT_NUM_WORKERS` | `4` | 容器只有 12 核，建议 4~6 |
| `LEROBOT_SAVE_FREQ` / `LEROBOT_LOG_FREQ` | `5000` / `50` | |
| `LEROBOT_RUN_NAME` | `<数据集>_smolvla_<月日_时分>` | 覆盖输出目录名 |

命令末尾的参数原样透传给 `lerobot-train`，例如解冻 VLM 主干：

```bash
./scripts/train_armpi.sh armpi_train_v1 20000 --policy.train_expert_only=false
```

> `train_expert_only=false` 会解冻 VLM，显存和时间都明显上升，数据不到几十条时别开。

### 断点续训

必须复用同一个运行名，让输出目录指向原目录：

```bash
LEROBOT_RUN_NAME=<原运行名> ./scripts/train_armpi.sh armpi_train_v1 20000 --resume=true
```

不要再手动传 `--output_dir`，脚本里已经设了，重复传会参数冲突。

---

## 训练输出

```
outputs/<run_name>/
└── checkpoints/
    ├── 000020/                        # 每 save_freq 步存一个
    │   ├── pretrained_model/          # ← 部署侧要的就是这个目录
    │   │   ├── config.json            #   策略配置（含 input_features）
    │   │   ├── model.safetensors
    │   │   ├── policy_preprocessor.json / .safetensors    # 归一化统计
    │   │   ├── policy_postprocessor.json / .safetensors
    │   │   └── train_config.json      #   含 rename_map
    │   └── training_state/            # 优化器/RNG 状态，续训用
    └── last -> 000020/
```

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

## ⚠️ 另外三个必须知道的点

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
| [docs/deployment-notes.md](docs/deployment-notes.md) | **给部署侧的说明**：模型输入契约、相机映射、常见报错对照 |
| [lerobot_doc/](lerobot_doc/) | 官方参考文档（安装 / 自定义硬件接入） |

---

## 目录结构

```
remote_SmolVLA/
├── README.md                      # 本文件
├── .gitignore
├── scripts/
│   ├── train_armpi.sh             #   正式训练（自动相机映射）
│   ├── smoke_test.sh              #   全链路冒烟
│   ├── verify_env.py              #   环境自检
│   ├── check_dataset.py           #   数据集校验
│   ├── bench_train_mem.py         #   显存基准
│   └── make_dummy_dataset.py      #   合成数据集 / 格式模板
├── docs/
│   ├── design.md                  #   设计文档
│   ├── environment.md             #   环境记录
│   ├── data-collection.md         #   录制侧说明
│   └── deployment-notes.md        #   部署侧说明
├── lerobot_doc/                   #   官方参考文档
└── outputs/                       #   训练产物（gitignore）
```
