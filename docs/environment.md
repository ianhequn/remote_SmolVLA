# SmolVLA 训练环境说明（GDUT HPC 实例 `n1`）

> 记录日期：2026-09-16 ｜ 实例：JupyterLab(GPU) 2.0 镜像 / Nomad 容器
> 用途：在本机（1× RTX 3090）微调 SmolVLA，机器人数据在本地 ArmPi Ultra 录制后上传

---

## 1. 硬件与系统

| 项目 | 规格 |
|---|---|
| GPU | 1× NVIDIA GeForce RTX 3090 24GB（sm_86 Ampere，**支持 bf16**） |
| 驱动 / CUDA | 570.195.03 / 驱动支持至 CUDA 12.8 |
| CPU | cgroup 限额 **12 核**（物理机 96 核，容器只给 12） |
| 内存 | **30 GiB**（+31G swap） |
| `/dev/shm` | 64G（tmpfs，受 30G cgroup 内存约束） |
| 系统盘 `/` | **仅 49G**（剩 47G）→ 包缓存/临时文件不要放这里 |
| 家目录 | NFS `192.168.10.203:/eaas/users/...`，**142T 可用** |
| **外设** | **无** `/dev/ttyUSB*`、`/dev/ttyACM*`、`/dev/video*` → **不能接机械臂/相机** |

## 2. 环境位置

```
conda 环境 : /home/zyy00005602/.conda/envs/lerobot   (Python 3.12.14)
激活       : conda activate /home/zyy00005602/.conda/envs/lerobot
Jupyter    : kernel 名 "lerobot"（Python (lerobot 0.6.1 / torch 2.11 cu128)）
```

### 已装版本（版本闭环已实测）

| 包 | 版本 | 说明 |
|---|---|---|
| lerobot | 0.6.1 | extras: `[smolvla,training]` |
| torch | **2.11.0+cu128** | 必须 cu128；PyPI 默认是 cu130（要驱动 ≥580.65） |
| torchvision | 0.26.0+cu128 | |
| torchcodec | 0.11.1 | 视频解码，需 ffmpeg 8 |
| transformers | 5.5.4 | lerobot 要求 `>=5.4,<5.6` |
| datasets | 4.8.5 | |
| ffmpeg | 8.1.2 | 来自 conda `defaults`（含 `libsvtav1`） |
| numpy | 2.2.6 | lerobot 要求 `<2.3` |

## 3. 本机网络约束

| 目标 | 状态 | 用途 |
|---|---|---|
| pypi.org | ✅ | pip 安装 |
| download.pytorch.org | ✅ | cu128 轮子 |
| **huggingface.co** | ❌ 超时 | — |
| **hf-mirror.com** | ✅ | `HF_ENDPOINT` 指向它 |
| **conda-forge / anaconda.org** | ❌ 超时 | 所以没用 miniforge |
| repo.anaconda.com (defaults) | ✅ | conda 包 |
| github.com | ✅ | 克隆仓库 |

`~/.bashrc` 已写入：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/zyy00005602/.cache/huggingface
export CONDA_PKGS_DIRS=/home/zyy00005602/.conda/pkgs   # 别写到 49G 的系统盘
export PIP_CACHE_DIR=/home/zyy00005602/.cache/pip
export TMPDIR=/dev/shm/tmp                             # NFS 上 multiprocessing 清理会报 busy
export OMP_NUM_THREADS=8 / MKL_NUM_THREADS=8
export WANDB_MODE=offline                              # 外网不可达
```

## 4. 踩过的 4 个坑（都已修好，换环境要注意）

### 坑 1 — 驱动装不下 cu130 的 torch
驱动 570.195.03 只支持到 CUDA 12.8，而 PyPI 上 torch 默认轮子是 cu130（要驱动 ≥580.65）。
**做法**：先显式装 `--index-url https://download.pytorch.org/whl/cu128` 的 torch/torchvision，再装 lerobot（否则 lerobot 会把 torch 换成跑不起来的版本）。

### 坑 2 — torchcodec 找不到 ffmpeg
`conda activate` 不会自动设 `LD_LIBRARY_PATH`，torchcodec 因此 dlopen 失败、自动降级到 pyav。
**做法**：环境里加了 `etc/conda/activate.d/00_ld_library_path.sh`，激活时把 `$CONDA_PREFIX/lib` 加进 `LD_LIBRARY_PATH`。

### 坑 3 — torchvision 的 libjpeg 遮蔽 conda 的 libjpeg（最难的一个）
```
libtiff.so.6: undefined symbol: jpeg12_write_raw_data, version LIBJPEG_8.0
```
torchvision 打包了 `torchvision.libs/libjpeg.b1a87dc0.so.8`（缺 12-bit 符号）。它先被载入后，
FFmpeg 的 `libavcodec` 再去 dlopen conda 的 `libtiff.so.6` 就找不到符号，torchcodec 加载失败。
**做法**：同一个 `activate.d` 脚本里 `LD_PRELOAD=$CONDA_PREFIX/lib/libjpeg.so.8`，让 conda 的 libjpeg 先占位。

### 坑 4 — `--policy.use_amp=true` 在 3090 上直接崩
```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
not implemented for 'BFloat16'
```
lerobot 0.6.1 用 `torch.get_autocast_dtype('cuda')` 推断 Accelerator 的 mixed_precision，
而 PyTorch 对 CUDA 的 autocast 默认值是 **float16**；于是 `use_amp=true` 被翻译成 fp16+GradScaler，
但 SmolVLA 前向产生的是 **bf16** 梯度 → 崩溃。
另外 lerobot 0.6.1 的 SmolVLAConfig **没有 `dtype` 字段**，所以模型卡上写的 `--policy.dtype=bfloat16` 在 0.6.1 里是无效参数。
**做法**：写了包装脚本 `lerobot-train-bf16`（在环境 `bin/` 下），先 `torch.set_autocast_dtype("cuda", torch.bfloat16)` 再调 lerobot。
**→ 所有训练命令都必须用 `lerobot-train-bf16`，不要用 `lerobot-train`。**

### 坑 5 — 非登录 shell 里 `HF_ENDPOINT` 没生效

脚本用 `./train_armpi.sh` 直接跑时不会读 `~/.bashrc`，`HF_ENDPOINT` 为空，
transformers 就去连被墙的 `huggingface.co`，在 `list_repo_templates` 处抛
`httpx.ConnectTimeout`（这个异常没被 transformers 捕获，直接往外冒）。
**做法**：`train_armpi.sh` 自己 export 所需环境变量（`HF_ENDPOINT` / `HF_HOME` / `TMPDIR` / `WANDB_MODE`），不依赖 `~/.bashrc`。
自己写脚本时也要注意这一点。

## 5. 实测性能（合成数据，1 相机 + 6 轴 state）

### 训练显存 / 速度（bf16 autocast，`chunk_size=50`）

| 相机数 | batch | 峰值显存 | ms/step |
|---|---|---|---|
| 1（ArmPi 实际） | 16 | 3.18 GiB | 201 |
| 1 | 32 | 5.01 GiB | 314 |
| 1 | 64 | **8.73 GiB** | 585 |
| 1 | 96 | 12.39 GiB | 870 |
| 1 | 128 | 16.05 GiB | 1150 |
| 1 | 160 | 19.76 GiB | 1417 |
| 1 | 192 | **OOM** | — |
| 3（base 默认） | 64 | 17.95 GiB | 1443 |
| 3 | 96 | **OOM** | — |

**建议**：1 相机用 `batch_size=32~64`，显存宽裕且吞吐好。真实训练实测 batch=32 时约 **100 samples/s**。

### 可训练参数

base 配置 `train_expert_only=True` / `freeze_vision_encoder=True`：
总参数 **450.0M**，其中可训练 **99.9M**（只有 action expert）。
想解冻 VLM 一起训：`--policy.train_expert_only=false`（显存与时间都会明显上升，数据量足够时再试）。

## 6. 冒烟测试工具

| 脚本 | 作用 |
|---|---|
| `scripts/verify_env.py` | 从 hub 加载 `lerobot/smolvla_base` 并真实前向，验证依赖/GPU/权重点（`--quick` 跳过权重） |
| `scripts/smoke_test.sh` | 全链路冒烟：合成数据集 → 训练 20 步 → 存/读 checkpoint → 自动清理（产物在 `/dev/shm`） |
| `scripts/bench_train_mem.py` | 真实 forward+backward 扫 batch size，输出显存/耗时表（`--cams` / `--batches` 参数） |
| `scripts/make_dummy_dataset.py` | 造合成 LeRobotDataset（1 相机 + 6 轴），**可作为数据格式模板** |
| `scripts/check_dataset.py` | 校验上传上来的真实数据集（版本/feature/完整性/解码） |

全链路已验证：`torchcodec` 解码 MP4 → DataLoader → SmolVLA forward/backward → checkpoint 保存 → checkpoint 回读。

## 7. 数据从本地录制机器传过来时注意

详见 [data-collection.md](data-collection.md)。要点：

1. **lerobot 版本要对齐 0.6.1**。本机数据集 `codebase_version = v3.0`，录制端用别的版本可能 schema 不兼容。
2. 目录放 `/home/zyy00005602/datasets/<名字>/`（NFS，142T）。
3. **不要把数据集 push 到 HF**：`hf-mirror.com` 是只读镜像，不支持上传；`huggingface.co` 又被墙。用 rsync/scp。
4. 录制端默认视频编码是 **`libsvtav1`（AV1）**——在树莓派上编码很吃 CPU。如果掉帧，
   在录制端把 `rgb_encoder` 换成 h264（`lerobot.configs.video.RGBEncoderConfig`）。
5. ArmPi Ultra 只有 1 路相机，SmolVLA 会把缺失的 camera2/camera3 用空图 + `img_masks=False` 屏蔽掉，**不需要改配置**（已实测）。
6. **训练前务必跑** `python scripts/check_dataset.py ~/datasets/<名字> --decode`。

## 8. 怎么开始训练

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/zyy00005602/.conda/envs/lerobot
cd /home/zyy00005602/MyWork/Lerobot/remote_SmolVLA

# 环境自检 + 全链路冒烟（换机器/升级依赖后先跑这个，产物自动清理）
python scripts/verify_env.py --quick
./scripts/smoke_test.sh

# 数据放到 ~/datasets/my_armpi/ 并校验通过之后:
python scripts/check_dataset.py ~/datasets/my_armpi --decode
./scripts/train_armpi.sh my_armpi 30000
```
