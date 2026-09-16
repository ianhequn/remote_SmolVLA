# 数据采集与上传

ArmPi Ultra 六轴机械臂（1 路 RGB 相机）的数据采集流程，以及把数据交到本机训练前的校验要求。

> **本机不能采集**：GDUT HPC 实例没有 `/dev/ttyUSB*` / `/dev/ttyACM*` / `/dev/video*`，
> `lerobot-record` 在这里必然失败。采集全部在 ArmPi Ultra 本地那台机器上完成。

---

## 1. 总体流程

```
① 录制机准备   装 lerobot 0.6.1 / 写 Robot 插件 / 相机与关节标定
      │
      ▼
② 采集         lerobot-record，多个 episode，每集固定一条 task
      │
      ▼
③ 上传         rsync / scp 到本机 ${LEROBOT_DATASETS_DIR}/（不走 HF Hub）
      │
      ▼
④ 校验         check_dataset.py 通过 ➜ 开始训练
```

---

## 2. 录制机（ArmPi Ultra）环境准备

### 2.1 lerobot 版本必须对齐

训练端是 **lerobot 0.6.1**，数据集格式 `codebase_version = v3.0`。
录制端用别的版本可能产生不兼容的 schema，上传后 `check_dataset.py` 会直接报错。

```bash
# 录制机上
conda create -y -n lerobot python=3.12
conda activate lerobot
pip install "lerobot[core_scripts]==0.6.1"     # core_scripts = dataset + hardware + viz
```

> `core_scripts` 包含 `dataset`（读写数据）、`hardware`（pynput/pyserial）、`viz`（rerun 可视化），
> 正好覆盖 `lerobot-record` 所需。如果录制机也要跑策略推理，再加 `evaluation` 和 `smolvla`。

### 2.2 ArmPi Ultra 需要一个自定义 `Robot` 插件

lerobot 官方没有 ArmPi Ultra 的实现（官方只有 SO-100/SO-101、Koch、LeKiwi 等）。
需要按 [`lerobot_doc/lerobot_integrate_hardware.md`](../lerobot_doc/lerobot_integrate_hardware.md) 写一个插件包：

**要满足 4 条约定**（缺一不可，否则 `lerobot-record --robot.type=...` 找不到它）：

| 约定 | 要求 |
|---|---|
| 包名前缀 | `lerobot_robot_armpi_ultra`（必须是 `lerobot_robot_` 开头） |
| 类名 | `ArmPiUltraConfig` / `ArmPiUltra`（去掉 `Config` 后缀就是设备类名） |
| 文件位置 | 设备类放在与配置类同模块，或名为 `armpi_ultra.py` 的子模块 |
| `__init__.py` | 同时导出 Config 类和设备类 |

目录结构：

```
lerobot_robot_armpi_ultra/
├── pyproject.toml                     # 依赖里要有 lerobot
└── lerobot_robot_armpi_ultra/
    ├── __init__.py                    # from .config_armpi_ultra import ArmPiUltraConfig
    │                                  # from .armpi_ultra import ArmPiUltra
    ├── config_armpi_ultra.py
    └── armpi_ultra.py
```

**必须实现的接口**（详见官方文档，这里只列要点）：

| 成员 | 内容 |
|---|---|
| `name = "armpi_ultra"` | CLI 里 `--robot.type=armpi_ultra`（若用了 `register_subclass`） |
| `config_class` | 指向 `ArmPiUltraConfig` |
| `observation_features` | `{"joint_1.pos": float, ..., "joint_6.pos": float, "camera1": (H, W, 3)}` |
| `action_features` | 6 个 `*.pos`（与 state 同名同序） |
| `is_connected` / `connect()` / `disconnect()` | 串口/总线 + 相机的开关 |
| `get_observation()` | 读 6 个关节角 + 抓一帧 RGB，键名与 `observation_features` 完全一致 |
| `send_action()` | 写 6 个目标角度 |
| `is_calibrated` / `calibrate()` / `configure()` | 关节零位/量程标定；不需要就写成 no-op |

其中"和舵机总线通信"这部分 lerobot 不会替你实现——用厂商 SDK 或自己封装串口协议。

### 2.3 相机

相机键名必须叫 **`camera1`**（`observation.images.camera1`），与训练端 feature 对齐。
分辨率建议录成 **256×256**（`resize_imgs_with_padding` 最终也是 256），录制时就用目标分辨率可省一次重采样。

---

## 3. 数据契约（必须与训练端一致）

| feature key | dtype | shape | 说明 |
|---|---|---|---|
| `observation.state` | `float32` | `(6,)` | 6 个关节角，`names = [joint_1.pos, ..., joint_6.pos]` |
| `action` | `float32` | `(6,)` | 6 个目标关节角，维度和命名必须与 state 一致 |
| `observation.images.camera1` | `video` | `(256, 256, 3)` | |
| `task` | `str` | — | 每帧一条语言指令，会进 `meta/tasks.parquet` |

参考实现：本仓库 [`scripts/make_dummy_dataset.py`](../scripts/make_dummy_dataset.py)——
它就是一个用 `LeRobotDataset.create()` 写 v3 数据集的完整例子，直接把里面的 feature 定义抄过来即可。

> **注意**：如果你是用 `lerobot-record` + 上面写的 `Robot` 插件录制，
> feature 会由 `robot.observation_features` / `action_features` 自动推导，不用手写。
> 上面那张表是用来核对"推导出来的是不是对的"。

---

## 4. 采集命令模板

```bash
# 录制机上
conda activate lerobot

lerobot-record \
  --robot.type=armpi_ultra \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=armpi_ultra_01 \
  --robot.cameras="{ camera1: {type: opencv, index_or_path: /dev/video0, width: 256, height: 256, fps: 30} }" \
  --teleop.type=...                      # 有主臂就填，没有就手动驱动/脚本驱动 \
  --dataset.repo_id=local/pick_cube \
  --dataset.single_task="pick up the red cube and place it in the box" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.fps=30 \
  --dataset.video=true \
  --display_data=true
```

`--teleop.*` 取决于你有没有主臂/手柄。ArmPi Ultra 若用手动拖拽或脚本回放采集，可省略 teleop。

---

## 5. 视频编码：注意 AV1

lerobot 默认 `rgb_encoder.vcodec = "libsvtav1"`（AV1）。

- **训练端解码**没问题（本机实测 torchcodec + ffmpeg 8 解码正常）。
- **录制端编码**在树莓派这类低功耗设备上非常吃 CPU，**很容易掉帧**（录到 30fps 但实际只有十几帧，
  时间戳对不上，训练时会被 `tolerance_s` 检查拦住）。

如果录制时出现掉帧 / 丢帧告警，在录制机上换成 h264：

```python
from lerobot.configs.video import RGBEncoderConfig
# vcodec="libx264" 或 "h264"，crf 用默认 30 即可
```

或者录完之后在录制机上用 ffmpeg 转码（本机 12 核也能转，但没必要占用 GPU 机器）。

---

## 6. 采集质量建议（SmolVLA 微调）

| 项 | 建议 | 原因 |
|---|---|---|
| episode 数 | **≥ 30~50 条**，同一条 task 下 | base 只有 99.9M 可训练参数，数据少容易过拟合 |
| 单集帧数 | **≥ 50 帧**（30fps 下约 2s） | SmolVLA `chunk_size=50`，太短的 episode 撑不满一个 chunk |
| 总帧数 | 3000~20000 帧起步 | |
| 语言指令 | **一条 task 一个固定说法**，简短、动词开头 | 语言 token 会参与训练，说法不统一会稀释信号 |
| 场景多样性 | 物体位置/光照/相机角度多换 | 这是泛化性的主要来源，比堆 episode 数更重要 |
| 失败 episode | 宁可重录，不要留 | 模仿学习会直接学到失败动作 |
| 相机固定 | 每次采集前确认相机没被碰歪 | 相机位姿一变，等于换了个任务 |

---

## 7. 上传到本机

```bash
# 在录制机上，把数据集推到 HPC
rsync -avP --exclude='.cache/' \
    ~/.cache/huggingface/lerobot/local/pick_cube/ \
    zyy00005602@<hpc-host>:/home/zyy00005602/MyWork/Lerobot/dataset/pick_cube/
```

要点：

- **不要用 `--dataset.push_to_hub` / `hf upload`**：`hf-mirror.com` 是只读镜像不支持上传，
  `huggingface.co` 在本机被墙。只能走 rsync/scp。
- **排除 `.cache/`**：LeRobotDataset 目录里的 `.cache/` 是本地索引缓存，传过去反而可能不一致。
- 加 `-P` 显示进度并支持断点续传；传输中断产生的 0 字节文件会被第 8 步的校验抓到。
- 目录名就是训练脚本的第一个参数：`~/MyWork/Lerobot/dataset/pick_cube/` → `./scripts/train_armpi.sh pick_cube`

---

## 8. 上传后校验（**训练前必做**）

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/zyy00005602/.conda/envs/lerobot
cd /home/zyy00005602/MyWork/Lerobot/remote_SmolVLA

python scripts/check_dataset.py ~/MyWork/Lerobot/dataset/pick_cube --decode
```

它会检查：

| 检查项 | 不通过会怎样 |
|---|---|
| `codebase_version == v3.0` | **error** — 录制端 lerobot 版本不对，训练会直接崩 |
| `fps`、episode/帧数、平均集长 | warning 提示数据量是否偏少 |
| `observation.state` / `action` 存在且维度一致 | **error** |
| 相机特征 dtype 是否为 `video` | warning |
| `task`（`meta/tasks.parquet`）非空 | warning — 语言输入会为空 |
| `data/` `videos/` `meta/stats.json` 存在 | **error** — 上传不完整 |
| 0 字节文件 | **error** — 传输中断 |
| 抽帧解码（`--decode`） | **error** + 顺带验证 torchcodec 是否可用 |

退出码 0 = 可训练；1 = 有 error，**先修数据再训练**。

---

## 9. 常见问题

**Q: 上传后校验说 `codebase_version` 不是 v3.0**
录制机的 lerobot 版本和训练端不一致。录制机上 `pip install "lerobot==0.6.1"` 后重录，
或在本机 `python -c "import lerobot; print(lerobot.__version__)"` 确认训练端版本再对齐。

**Q: 训练时报 `tolerance_s` / 时间戳不同步**
录制时掉帧了（见第 5 节 AV1 编码问题）。重新录，别试图修补。

**Q: 只能用 1 路相机，会不会和 base 模型对不上？**
不会。base 声明了 3 路，缺失的会被补成空图并在注意力里 mask 掉。已实测跑通。

**Q: 想加第二路相机？**
训练端不用改任何东西——`input_features` 是从数据集的 feature 推导的，
录制时多配一个相机即可（键名 `camera2`）。但要注意：**加了相机后，
之前只用单相机训出来的 checkpoint 场景不通用**，因为相机布局变了。

**Q: 数据集传到别处了**
用 `LEROBOT_DATASETS_DIR=/path/to/datasets ./scripts/train_armpi.sh pick_cube` 覆盖默认路径。
