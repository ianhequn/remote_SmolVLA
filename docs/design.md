# 设计文档

SmolVLA 在 GDUT HPC 实例 `n1` 上的微调方案设计。

---

## 1. 目标与边界

**目标**：用 ArmPi Ultra 六轴机械臂（1 路 RGB 相机）自采数据，微调出可部署的 SmolVLA 策略。

**职责划分**

| 环节 | 在哪做 | 为什么 |
|---|---|---|
| 数据采集 | ArmPi Ultra 本地机器 | 机械臂/相机物理连接在那台机器上 |
| 数据上传 | scp / rsync → NFS | hf-mirror 只读、huggingface.co 被墙 |
| 训练 | 本机（1×RTX3090） | 唯一有 GPU 的地方 |
| 部署推理 | 回 ArmPi Ultra 本地 | 本机无外设 |

**明确不做**：本机不做数据采集、不做真机 rollout 评测（无串口/无相机）。

---

## 2. 系统全景

```
                    ┌─────────────────────────────────────────┐
                    │  ArmPi Ultra (本地录制机)                │
                    │   ├─ 6× 舵机总线                         │
                    │   ├─ 1× RGB 相机                         │
                    │   └─ lerobot 0.6.1 (版本必须对齐)        │
                    └───────────────┬─────────────────────────┘
                                    │ lerobot-record
                                    ▼
                        LeRobotDataset  (codebase_version=v3.0)
                        <name>/meta/info.json, data/*.parquet, videos/*.mp4
                                    │
                                    │ rsync / scp（不走 HF）
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  GDUT HPC  n1  (Nomad 容器)                                     │
   │   ~/MyWork/Lerobot/dataset/<name>/           ← NFS 142T         │
   │            │                                                    │
   │            ▼                                                    │
   │   lerobot-train-bf16  ──►  SmolVLA 微调  ──►  checkpoints/      │
   │            ▲                                   (NFS)            │
   │            │                                                    │
   │   lerobot/smolvla_base ──◄── hf-mirror.com (HF_ENDPOINT)        │
   └────────────────────────────────────────────────────────────────┘
                                    │
                                    │ 手动拷回
                                    ▼
                    ┌─────────────────────────────────────────┐
                    │  ArmPi Ultra  →  lerobot-record 回放评测 │
                    └─────────────────────────────────────────┘
```

---

## 3. 约束 → 决策

每条决策都来自一条实测约束，不是拍脑袋。

| # | 实测约束 | 决策 | 依据 |
|---|---|---|---|
| 1 | 驱动 `570.195.03`，最高支持 CUDA 12.8 | torch 必须装 **cu128** 轮子 | PyPI 默认 torch 是 cu130（驱动下限 580.65），装了直接跑不起来 |
| 2 | `conda.anaconda.org` / `anaconda.org` 超时 | 用**已装的 Anaconda 24.5.0 + `defaults` 通道**，不装 miniforge | miniforge 默认通道就是 conda-forge，装了也解不出包 |
| 3 | `conda-forge` 不可达 | ffmpeg 从 `defaults` 装 8.1.2 | 文档推荐 `conda install ffmpeg -c conda-forge`，这里走不通；apt 只有 4.2.7，太旧（torchcodec 要 7/8） |
| 4 | `huggingface.co` 超时、`hf-mirror.com` 正常 | `HF_ENDPOINT=https://hf-mirror.com`，且脚本自带（不依赖 `~/.bashrc`） | 非登录 shell 不读 `.bashrc`，会导致 transformers 去连被墙域名 |
| 5 | 系统盘 `/` 只有 49G | `CONDA_PKGS_DIRS`、`PIP_CACHE_DIR` 指向 NFS；`TMPDIR` 指向 `/dev/shm` | conda 默认包缓存在 `/opt/conda/pkgs`，写在 49G 盘上 |
| 6 | NFS 上 multiprocessing 清理报 `Device or resource busy` | `TMPDIR=/dev/shm/tmp` | NFS 上 unlink 已 mmap 的文件会 busy |
| 7 | `conda activate` 不设 `LD_LIBRARY_PATH` | 环境内加 `etc/conda/activate.d/` 脚本 | torchcodec 靠 `dlopen` 找 ffmpeg 共享库，找不到就静默降级到 pyav |
| 8 | torchvision 打包的 `libjpeg.so.8` 遮蔽 conda 的 libjpeg | 同上的 activate 脚本里 `LD_PRELOAD` conda libjpeg | 见"失败模式 F3" |
| 9 | PyTorch CUDA autocast 默认 dtype 是 `float16` | 自建 `lerobot-train-bf16` 包装脚本强制 bf16 | 见"失败模式 F4" |
| 10 | 容器只有 12 核 / 30 GiB | `num_workers=4~6` | 再高会和 GPU 抢 CPU，且 worker 会吃内存 |
| 11 | RTX 3090 24G，sm_86 支持 bf16 | bf16 混合精度训练 | 显存约为 fp32 的一半，速度更快 |
| 12 | lerobot 0.6.1 无梯度累积参数 | 直接用 `batch_size` 控制有效 batch（= batch_size × GPU 数） | 源码 `effective_bs = cfg.batch_size * num_processes` |

---

## 4. 训练方案

### 4.1 模型

`lerobot/smolvla_base`：SmolVLM2-500M-Video-Instruct（VLM）+ action expert，共 **450M 参数**。

base 配置默认 `train_expert_only=True`、`freeze_vision_encoder=True`
→ **实际可训练 99.9M**（只有 action expert）。

想解冻 VLM 一起微调：`--policy.train_expert_only=false`（显存与时间都会明显上升，数据量足够时再试）。

### 4.2 超参

| 参数 | 取值 | 说明 |
|---|---|---|
| `steps` | 30000（默认） | 小数据集（~50 episode）一般 20k~50k |
| `batch_size` | 32（可到 64） | 1 相机实测 32 → 5.0 GiB，64 → 8.7 GiB |
| `policy.use_amp` | `true`（经 bf16 包装脚本） | |
| `optimizer_lr` | `1e-4`（lerobot 默认） | 带 warmup/decay 调度 |
| `chunk_size` / `n_action_steps` | 50 / 50 | base 默认值 |
| `num_workers` | 4 | |
| `eval_steps` / `env_eval_freq` | 0 | 真机数据集，无可仿真环境可评 |

### 4.3 显存预算（实测，`chunk_size=50`，bf16）

| 相机数 | batch | 峰值显存 | ms/step | 样本/s |
|---|---|---|---|---|
| 1 | 16 | 3.19 GiB | 183 | 88 |
| 1 | **32** | **5.02 GiB** | 315 | **101** |
| 1 | 64 | 8.73 GiB | 585 | 109 |
| 1 | 96 | 12.39 GiB | 870 | 110 |
| 1 | 128 | 16.05 GiB | 1150 | 111 |
| 1 | 160 | 19.76 GiB | 1417 | 113 |
| 1 | 192 | **OOM** | — | — |
| 3 | 64 | 17.95 GiB | 1443 | 44 |
| 3 | 96 | **OOM** | — | — |

单相机方案的显存开销主要来自 **action expert 的 chunk_size=50 序列**（50 个 action token），而不是图像——
所以 64→128 显存涨得比 16→32 还快。24G 上限在 batch ≈ 160~192 之间。

复现命令：`python scripts/bench_train_mem.py --cams 1 --batches 16 32 64 96 128 160 192`

### 4.4 单相机处理

ArmPi Ultra 只有 1 路相机，但 base 配置声明了 3 路。**不需要改配置**：

base config 里 `empty_cameras = 0`（`configuration_smolvla.py` 的默认值），而
`prepare_images()` 补空图的循环第一轮就 `break`：

```python
for num_empty_cameras in range(len(missing_img_keys)):
    if num_empty_cameras >= self.config.empty_cameras:   # 0 >= 0 → 立刻 break
        break
```

也就是说缺失的相机**根本不会被补进来**，实际只喂 batch 里存在的那几路。
实测：只给 1 路相机时，送进模型的图片数是 **1**（`shape=(B, 3, 512, 512)`，mask 全 True），
既没有占位空图也不浪费算力。

代价是**相机键名必须匹配**，见 4.5。

### 4.5 相机命名与 rename_map

base config 把 `image_features` 写死为 `camera1 / camera2 / camera3`，又因为 `empty_cameras=0`
（缺失相机不补位），数据集里的相机键名**必须落在这个集合内**。有**两处**会拦住你：

| 关卡 | 位置 | 行为 |
|---|---|---|
| 训练前校验 | `make_policy()` → `validate_visual_features_consistency()` | 不匹配 → `ValueError: Feature mismatch` |
| 运行时取图 | `SmolVLA.prepare_images()` | 按 key 匹配 `config.image_features`，全不认 → `ValueError: All image features are missing from the batch` |

判定规则不是“必须完全相等”，而是「数据集相机集合」与「policy 相机集合」
**任意一边是另一边的子集**即放行：

| 数据集相机名 | 直接跑 |
|---|---|
| `camera1` / `camera2` / `camera3` | ✅ 单个也行，不要求连号 |
| `camera1` + `camera3` | ✅ 子集 |
| `front` | ❌ 互不包含 |
| `front` + `camera1` | ❌ 仍然互不包含 |

**决策：不改数据集，在训练脚本里自动补 `--rename_map`。**

`train_armpi.sh` 读 `<数据集>/meta/info.json` 的 `features`，筛出 `observation.images.*` 排序，
第 i 个映射成 `camera{i+1}`；已经合规就不生成任何参数（no-op）。相机数 > 3 直接报错退出，不静默丢相机。

理由：

- 不动物理数据，可逆，可追溯（映射写进 checkpoint 的 `train_config.json`）
- 以后按规范录的数据集（相机就叫 `camera1`）自动 no-op，不用删代码
- 官方支持路径：`make_policy(rename_map=...)` + `RenameObservationsProcessorStep`，且
  `lerobot_train.py` 把 `cfg.rename_map` 同时传给 policy 和 preprocessor pipeline

**副作用（重要）**：映射会被烘进 checkpoint 的 `policy_preprocessor.json` 第一步，所以

- 部署侧 **原样喂 `observation.images.front`** 也能跑（preprocessor 会改名）
- 但更好的是在 ArmPi 的 robot 配置里直接把相机键名定义成 `observation.images.camera1`，
  以后录制 → 训练 → 部署全程不需要任何映射
- 多相机时“按名字排序映射”如果和部署侧不一致，会**静默把两路相机调换**——不报错，只是效果变差。
  1 路相机没有这个问题。

详细部署约定见 [deployment-notes.md](deployment-notes.md)。

---

## 5. 数据契约

### 5.1 格式

必须是 **LeRobotDataset `codebase_version=v3.0`**，与录制端 lerobot 0.6.1 生成的一致。

```
<name>/
├── meta/
│   ├── info.json          # 版本、fps、features、总帧数/ episode 数  ← 本仓库脚本会读它做预检查
│   ├── stats.json         # 归一化统计量
│   ├── tasks.parquet
│   └── episodes/
├── data/chunk-000/*.parquet       # state / action / 时间戳
└── videos/<相机键>/chunk-000/*.mp4 # 图像（默认 AV1/libsvtav1）
```

### 5.2 Feature 定义（与 `make_dummy_dataset.py` 一致，可作为转换脚本模板）

| key | dtype | shape |
|---|---|---|
| `observation.state` | `float32` | `(6,)` + `names` |
| `action` | `float32` | `(6,)` + `names` |
| `observation.images.camera1` | `video` | `(256, 256, 3)` |
| `task` | str（每帧） | 语言指令 |

真实数据集 `armpi_train_v1` 的差异：相机键名是 `observation.images.front`（需 rename，见 4.5），
分辨率 480×640（训练时内部 resize+pad 到 512×512，分辨率不影响显存/耗时），
`names` 为 `joint1.pos … joint5.pos, r_joint.pos`。参见 README 的“当前数据集”表。

### 5.3 目录与命名约定

| 项 | 约定 |
|---|---|
| 数据集位置 | `${LEROBOT_DATASETS_DIR}/<name>/`，默认 `~/MyWork/Lerobot/dataset/` |
| `repo_id` | 本地训练统一用 `local/<name>`（避免误触发 Hub 上传） |
| 训练输出 | `${LEROBOT_RUNS_DIR}/<name>_smolvla_<月日_时分>/` |
| checkpoint | `.../checkpoints/last/pretrained_model/` |

### 5.4 防误上传

`train_armpi.sh` 固定带 `--policy.push_to_hub=false`。
（lerobot 0.6.1 的 `PushToHubMixin.push_to_hub` 默认为 `True`，不显式关掉会在训练**结束后**尝试
`create_repo` 并报 401 —— 因为镜像站不支持写。）

---

## 6. 失败模式（FMEA）

按"症状 → 根因 → 对策"记录。前 4 个是本机实际踩到的。

### F1 — torchcodec 静默降级到 pyav

- **症状**：`WARNING: Could not load libtorchcodec ... Falling back to 'pyav'`，训练能跑但视频解码变慢
- **根因**：`conda activate` 不设置 `LD_LIBRARY_PATH`，`dlopen("libavcodec.so.62")` 找不到
- **对策**：`$LEROBOT_ENV/etc/conda/activate.d/00_ld_library_path.sh` 注入 `$CONDA_PREFIX/lib`
- **验证**：`python -c "from lerobot.utils.import_utils import get_safe_default_video_backend as g; print(g())"` 应输出 `torchcodec`

### F2 — `libtiff.so.6: undefined symbol: jpeg12_write_raw_data`

- **症状**：torchcodec 加载失败，报 `libtorchcodec_core8.so` 加载不了
- **根因**：torchvision 打包的 `torchvision.libs/libjpeg.b1a87dc0.so.8` 缺 12-bit 符号，
  它先被载入后，FFmpeg 的 `libavcodec` 再 `dlopen` conda 的 `libtiff.so.6` 就解析不到该符号
- **对策**：同一个 activate 脚本里 `LD_PRELOAD=$CONDA_PREFIX/lib/libjpeg.so.8`，让 conda 的 libjpeg 先占位
- **排查手法**：`objdump -T $CONDA_PREFIX/lib/libtiff.so.6 | grep jpeg` 看它要哪个符号版本

### F3 — `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'`

- **症状**：`--policy.use_amp=true` 训练第一步就崩在 `clip_grad_norm_`
- **根因**：`lerobot_train.py` 用 `torch.get_autocast_dtype(device_type)` 推断 Accelerator 的
  `mixed_precision`，而 PyTorch 对 CUDA 的 autocast 默认是 **float16**（不是 bf16）。
  于是 `use_amp=true` 被翻译成 fp16 + GradScaler，但 SmolVLA 前向产生的是 **bf16** 梯度 → 不匹配。
  且 lerobot 0.6.1 的 `SmolVLAConfig` **没有 `dtype` 字段**，所以模型卡上的 `--policy.dtype=bfloat16`
  是无参可传的，只能走 `use_amp` 这条路。
- **对策**：`$LEROBOT_ENV/bin/lerobot-train-bf16` —— 先 `torch.set_autocast_dtype("cuda", torch.bfloat16)`，
  再调 lerobot 的 `main()`。这样 `mixed_precision` 会被算成 `bf16`，不带 GradScaler。

### F4 — `httpx.ConnectTimeout` 在 `list_repo_templates`

- **症状**：训练刚要开始就超时退出，栈顶是 `transformers/utils/hub.py: list_repo_templates`
- **根因**：脚本在非登录 shell 里跑，`~/.bashrc` 的 `HF_ENDPOINT` 没生效 → 去连被墙的 `huggingface.co`。
  且 `httpx.ConnectTimeout` 不是 `httpx.NetworkError` 的子类，transformers 的 `except` 接不住，直接往外抛。
- **对策**：所有脚本自行 `export HF_ENDPOINT`（`train_armpi.sh`、`smoke_test.sh`、`verify_env.py`、`bench_train_mem.py` 都做了）

### F5 — 训练结束报 `401 Unauthorized` at `api/repos/create`

- **症状**：40 步都跑完了，最后 push 到 Hub 失败
- **根因**：`push_to_hub` 默认 `True`
- **对策**：`--policy.push_to_hub=false`（`train_armpi.sh` 已内置）

### F6 — 系统盘写满

- **症状**：`No space left on device`，但 NFS 还有 142T
- **根因**：`/` 只有 49G，conda 包缓存默认落 `/opt/conda/pkgs`
- **对策**：`CONDA_PKGS_DIRS` / `PIP_CACHE_DIR` / `TMPDIR` 全部指向 NFS 或 `/dev/shm`

---

## 7. 目录结构设计

```
remote_SmolVLA/
├── README.md                  # 入口：快速开始 + 相机命名坑 + 脚本表
├── scripts/                   # 所有可执行物，零散落在根目录不利于交接
│   ├── train_armpi.sh         #   唯一训练入口（bf16 包装 + 环境变量 + 自动相机映射）
│   ├── smoke_test.sh          #   自检：合成数据 + 20 步训练 + 回读 + 自动清理
│   ├── verify_env.py          #   --quick 模式供 smoke_test 复用
│   ├── check_dataset.py       #   上传上来的真实数据集校验
│   ├── bench_train_mem.py     #   显存基准，定 batch_size 用
│   └── make_dummy_dataset.py  #   既做夹具又做格式文档
├── docs/                      # 设计 / 环境 / 采集 / 部署 四份文档
├── lerobot_doc/               # 官方原文（安装 & 自定义硬件），保留以便对照上游变化
└── outputs/                   # 训练产物，gitignore
```

数据集不在仓库里，固定放 `/home/zyy00005602/MyWork/Lerobot/dataset/`（NFS，142T）。

**为什么脚本里硬编码 `/home/zyy00005602/...` 作为默认值**：这台机器是单用户固定路径，默认值写死好用；
但都留了 `LEROBOT_*` 环境变量可覆盖，换机器不用改脚本。

**为什么 `smoke_test.sh` 把产物放 `/dev/shm`**：训练产物单个 checkpoint 就 906MB，
放内存盘既快又能靠进程退出/trap 自动清理，绝不会污染 NFS 家目录。

---

## 8. 已知限制与演进

| 限制 | 影响 | 后续方向 |
|---|---|---|
| 无梯度累积（lerobot 0.6.1） | 有效 batch 只能靠加大 batch_size，受 24G 显存限制 | 升到支持 grad accum 的版本，或自己包一层 |
| 本机不能接机器人 | 无法真机闭环评测（`lerobot-record` 跑不了） | 在 ArmPi Ultra 侧做 rollout，只把失败 episode 回传做增量训练 |
| lerobot 无 `dummy` 机器人类型 | 官方没有 ArmPi Ultra 的 `Robot` 实现 | 按 `lerobot_doc/lerobot_integrate_hardware.md` 的插件约定写 `lerobot_robot_armpi_ultra` |
| 单卡 | 无法分布式;`num_processes` 恒为 1 | — |
| 数据集编码默认 AV1 | 录制侧（树莓派）编码吃 CPU，可能掉帧 | 录制端把 `rgb_encoder.vcodec` 换成 h264 |
| `hf-mirror.com` 只读 | 训练产物不能自动上云备份 | 产物留在 NFS（已在 RAID 上），或另找对象存储 |
