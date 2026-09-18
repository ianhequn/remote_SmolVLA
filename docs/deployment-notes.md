# 部署说明（ArmPi Ultra 侧）

> 面向把训练好的 SmolVLA 模型跑回机械臂的开发者。
> 训练在 GDUT HPC `n1` 上完成，本文只讲**模型侧的输入输出契约**和容易对不上的地方。

---

## 1. 拿什么

```
<run>/checkpoints/<step>/pretrained_model/
├── config.json                    # 策略配置（含 input_features）
├── model.safetensors              # 权重
├── policy_preprocessor.json       # 推理前处理链（含相机改名 + 归一化统计）
├── policy_preprocessor_step_*_*.safetensors
├── policy_postprocessor.json      # 输出反归一化
├── policy_postprocessor_step_*_*.safetensors
└── train_config.json              # 训练时的完整配置（含相机映射表）
```

**整个 `pretrained_model/` 目录一起拷**，缺一个文件都加载不了。
先读 `train_config.json` 的 `rename_map`，确认相机映射；再看 `config.json` 的 `input_features`。

```bash
rsync -avP user@<hpc-host>:.../checkpoints/last/pretrained_model/ ./models/armpi_v1/
```

## 2. 版本必须对齐

**lerobot 0.6.1**。训练侧就是这个版本，数据集是 `codebase_version = v3.0`。
版本不一致时 `config.json` 的字段可能对不上，加载会报错或静默行为不一致。

## 3. 输入契约

| 键 | 形状 / 类型 | 说明 |
|---|---|---|
| `observation.images.front` | `uint8 (480, 640, 3)` | 主相机。**也可直接按 `observation.images.camera1` 喂**，见第 4 节 |
| `observation.state` | `float32 (6,)` | 顺序固定：`joint1.pos, joint2.pos, joint3.pos, joint4.pos, joint5.pos, r_joint.pos` —— 与录制时 `observation.state` 完全一致 |
| `task` | `str` | **训练时用的那条指令**：`Pick up the square wooden block in front and bring it closer.` |

两点提醒：

- **`task` 不是随便写的。** SmolVLA 是语言条件策略，换一句意思相近但措辞不同的话，行为会变。
  部署时请直接使用上面这条原文。它会被 tokenizer 截断到 48 token。
- **`state` 的顺序和单位必须和录制时一致**，模型靠数值分布（已固化在 preprocessor 的归一化统计里）理解状态。

## 4. 相机键名：两条路，任选

训练时数据集里叫 `front`，但 SmolVLA 底座的 `image_features` 写死为 `camera1/2/3`，
所以训练脚本自动加了一条映射：

```json
{"observation.images.front": "observation.images.camera1"}
```

这条映射**已经被存进 checkpoint 的 `policy_preprocessor.json`，是推理链的第一步**。
所以：

| 做法 | 结果 |
|---|---|
| **A. 原样喂 `observation.images.front`** | ✅ 最省事。只要用 checkpoint 自带的 preprocessor，它会自动改名成 `camera1` |
| **B. 把相机键名定义成 `observation.images.camera1`** | ✅ 最干净。映射变成 no-op，不依赖 preprocessor |

⚠️ 如果你**自己手工搭 pipeline / 不加载 checkpoint 的 preprocessor**，那必须在送进模型前
把 `front` 改成 `camera1`，否则会报 `All image features are missing from the batch`。

**给后续数据集的建议**：直接在 robot 配置里把相机键名定义成 `observation.images.camera1`，
以后录制、训练、部署全程不需要任何映射。

## 5. 输出契约

| 项 | 值 |
|---|---|
| 动作维度 | `(6,)` float32 |
| 顺序 | 同 `observation.state`：`joint1.pos … r_joint.pos` |
| 语义 | 各关节的**绝对位置**（依据特征名 `*.pos` 推断，请对照录制脚本确认） |
| 值域 | 反归一化后 = 数据集 `action` 的原始值域 |
| chunk | `chunk_size = 50`，`n_action_steps = 50` |

**关于 50 步**：`select_action()` 内部维护一个动作队列。队列空时会一次性预测 50 步，
然后每次调用弹出一个。`n_action_steps = 50` 意味着这 50 步是**开环执行**的
（25 fps 下 = 2 秒），中途不会用新观测修正。

因此控制循环必须**按固定 25 Hz 调用 `select_action()`**，并在每个 episode 开始时调用
`policy.reset()` 清空队列，否则队列会错位。

## 6. 最小推理代码

```python
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

CKPT = "./models/armpi_v1"          # pretrained_model/ 目录
TASK = "Pick up the square wooden block in front and bring it closer."

policy = SmolVLAPolicy.from_pretrained(CKPT)
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=CKPT,
    # 非 CUDA 机器（比如直接跑在树莓派上）必须加这一行：
    # 训练环境存的 device_processor 里写死了 "cuda"
    # preprocessor_overrides={"device_processor": {"device": "cpu"}},
)

policy.reset()                       # 每个 episode 开始时调用

while True:
    frame = read_camera()            # uint8 (480, 640, 3)
    state = read_joints()            # float32 (6,)

    obs = {
        "observation.images.front": frame,   # 或 "...camera1"，见第 4 节
        "observation.state": state,
        "task": TASK,
    }
    obs = preprocessor(obs)
    action = policy.select_action(obs)
    action = postprocessor(action)   # -> (6,) float32

    send_to_servos(action)           # 顺序：joint1..joint5, r_joint
    sleep_until_next_tick()          # 固定 25 Hz
```

## 7. 常见报错对照

| 报错 | 原因 | 处理 |
|---|---|---|
| `Feature mismatch … Extra features: ['observation.images.front']` | 没走 rename（训练侧已自动处理，部署侧一般是手工搭 pipeline 才会撞） | 喂 `camera1`，或加载 checkpoint 自带的 preprocessor |
| `All image features are missing from the batch` | 相机键名不在 `camera1/2/3` 里 | 同上 |
| `KeyError: 'task'` | observation 里没给指令 | 补上第 3 节的 `task` 原文 |
| 动作为 NaN / 数值离谱 | `state` 顺序或单位不对 | 对照 `meta/info.json` 的 `features.observation.state.names` |
| `torch.cuda.is_available() == False` 时报 cuda 相关错 | checkpoint 的 `device_processor` 写死 `cuda` | 用 `preprocessor_overrides` 改成 `cpu` |
| 加载 `config.json` 报未知字段 | lerobot 版本不是 0.6.1 | 对齐版本 |

## 8. 交付核对清单

- [ ] lerobot 版本 = **0.6.1**
- [ ] `pretrained_model/` 整个目录拷全（含 `policy_preprocessor*.json/.safetensors`）
- [ ] 读过 `train_config.json` 的 `rename_map`
- [ ] 相机键名 = `observation.images.front`（配自带 preprocessor）或 `observation.images.camera1`
- [ ] `observation.state` 顺序 = `joint1..joint5, r_joint`，单位与录制时一致
- [ ] `task` = `Pick up the square wooden block in front and bring it closer.`
- [ ] 控制循环固定 **25 Hz**，每个 episode 开始 `policy.reset()`
- [ ] 动作下发顺序 = `joint1..joint5, r_joint`，绝对位置

## 9. 注意：本模型只覆盖一种技能

`armpi_train_v1` 只有 **1 个 task、61 集、6662 帧**，而且**最后一个 checkpoint 通常不是最好的**。
上线前请把 `002000 / 004000 / … /last` 几个 checkpoint 都在真机上跑一遍，挑成功率最高的那个，
而不是直接拿 `last`。数据量小，过拟合很常见。
