#!/usr/bin/env python
"""生成一个**合成** LeRobotDataset，用作训练链路的冒烟测试夹具。

⚠️ 这里的数据全部是 numpy 随机生成的假数据，不是任何机器人的真实采集：
   - 6 轴 state / action 是人为编的正弦函数
   - 图像是黑底上移动的噪声方块
   它唯一的作用是让「数据集读取 → 视频解码 → forward/backward → 存 checkpoint」
   这条代码路径跑起来，**不能用来评估模型效果**。

同时它也是一份「LeRobotDataset v3 数据格式」的参考实现：
如果要自己写一个把 ArmPi Ultra 数据转成 LeRobot 格式的转换脚本，可以照着改。

用法：
    conda activate /home/zyy00005602/.conda/envs/lerobot
    python scripts/make_dummy_dataset.py                       # 生成到 /dev/shm/lerobot_smoke/dummy_armpi
    python scripts/make_dummy_dataset.py --root /tmp/foo       # 指定位置
    python scripts/make_dummy_dataset.py --episodes 10 --frames 60
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_ROOT = "/dev/shm/lerobot_smoke/dummy_armpi"  # 放 shm，不污染 NFS 家目录
IMG = 256
N_JOINTS = 6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"输出目录（默认 {DEFAULT_ROOT}）")
    ap.add_argument("--repo-id", default="local/dummy_armpi", help="数据集 repo_id（本地用 local/xxx）")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--frames", type=int, default=40, help="每个 episode 的帧数")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--robot-type", default="armpi_ultra")
    ap.add_argument("--force", action="store_true", help="目录存在时先删除")
    args = ap.parse_args()

    root = Path(args.root)
    if root.exists():
        if not args.force:
            print(f"目录已存在: {root}\n加 --force 覆盖，或换个 --root。", file=sys.stderr)
            return 1
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (N_JOINTS,),
            "names": [f"joint_{i}.pos" for i in range(N_JOINTS)],
        },
        "action": {
            "dtype": "float32",
            "shape": (N_JOINTS,),
            "names": [f"joint_{i}.pos" for i in range(N_JOINTS)],
        },
        "observation.images.camera1": {
            "dtype": "video",  # 走 MP4 + torchcodec，和真实录制一致
            "shape": (IMG, IMG, 3),
            "names": ["height", "width", "channels"],
        },
    }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=root,
        robot_type=args.robot_type,
        use_videos=True,
        image_writer_threads=4,
    )

    rng = np.random.default_rng(0)
    for ep in range(args.episodes):
        phase = rng.uniform(0, 2 * np.pi, size=N_JOINTS)
        for i in range(args.frames):
            t = i / args.frames
            state = (np.sin(2 * np.pi * t + phase) * 0.5).astype(np.float32)
            action = (np.sin(2 * np.pi * (t + 1 / args.frames) + phase) * 0.5).astype(np.float32)

            img = np.zeros((IMG, IMG, 3), dtype=np.uint8)
            cx = int(40 + t * (IMG - 80))
            cy = int(40 + (0.5 + 0.5 * np.sin(2 * np.pi * t + phase[0])) * (IMG - 80))
            img[max(0, cy - 12) : cy + 12, max(0, cx - 12) : cx + 12] = (
                (200 + rng.integers(0, 55, size=(24, 24, 3))).clip(0, 255).astype(np.uint8)
            )
            img[:8, :8] = (ep * 40 % 256, i * 6 % 256, 128)  # 左上角标记 episode / 帧号

            ds.add_frame(
                {
                    "observation.state": torch.from_numpy(state),
                    "action": torch.from_numpy(action),
                    "observation.images.camera1": img,
                    "task": "move the red cube to the target position",
                }
            )
        ds.save_episode()
    ds.finalize()

    print(f"✅ 合成数据集已生成: {root}")
    print(f"   repo_id={args.repo_id}  episodes={ds.meta.total_episodes}  frames={ds.meta.total_frames}")
    print()
    print("   全链路冒烟测试（生成 + 训练 + 回读 + 清理）:")
    print("     ./scripts/smoke_test.sh")
    print()
    print("   单独拿它训练:")
    print(f"     LEROBOT_DATASETS_DIR={root.parent} ./scripts/train_armpi.sh {root.name} 20")
    print()
    print("   用完删掉（只删这个合成数据集，别手滑删到父目录）:")
    print(f"     rm -rf {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
