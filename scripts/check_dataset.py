#!/usr/bin/env python
"""校验一个（刚从录制机上传上来的）LeRobotDataset 能不能直接拿来训练。

不加载模型、不占显存，几秒钟就能给出结论。上传完数据先跑这个。

用法：
    conda activate /home/zyy00005602/.conda/envs/lerobot
    python scripts/check_dataset.py ~/datasets/pick_cube
    python scripts/check_dataset.py ~/datasets/pick_cube --decode   # 额外抽帧解码（较慢）

退出码：0 = 通过（可能有 warning），1 = 有 error，不能训练。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_FEATURES = ("observation.state", "action")
IMAGE_PREFIX = "observation.images."
EXPECTED_VERSION = "v3.0"


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warns: list[str] = []
        self.infos: list[str] = []

    def err(self, m: str) -> None:
        self.errors.append(m)

    def warn(self, m: str) -> None:
        self.warns.append(m)

    def info(self, m: str) -> None:
        self.infos.append(m)

    def dump(self) -> int:
        for m in self.infos:
            print(f"  · {m}")
        for m in self.warns:
            print(f"  ⚠ {m}")
        for m in self.errors:
            print(f"  ✗ {m}")
        print()
        if self.errors:
            print(f"❌ 校验失败：{len(self.errors)} 个 error，{len(self.warns)} 个 warning")
            return 1
        if self.warns:
            print(f"✅ 校验通过（{len(self.warns)} 个 warning，可训练）")
        else:
            print("✅ 校验通过")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="数据集目录")
    ap.add_argument("--decode", action="store_true", help="随机抽一个视频文件解码首帧，验证 torchcodec 可用")
    ap.add_argument("--expect-cams", type=int, default=None, help="期望的相机数量（默认不检查）")
    args = ap.parse_args()

    root: Path = args.dataset.expanduser().resolve()
    r = Report()

    print(f"数据集: {root}")
    print()

    if not root.is_dir():
        print(f"✗ 目录不存在: {root}")
        return 1

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        print(f"✗ 不是 LeRobotDataset：缺少 {info_path}")
        return 1

    info = json.loads(info_path.read_text())

    # ---------------------------- 基本信息 ----------------------------------
    version = info.get("codebase_version")
    if version != EXPECTED_VERSION:
        r.err(f"codebase_version = {version!r}，训练端要求 {EXPECTED_VERSION!r}（录制端 lerobot 版本不对）")
    else:
        r.info(f"codebase_version = {version}")

    r.info(f"robot_type = {info.get('robot_type')}")

    fps = info.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        r.err(f"fps 非法: {fps!r}")
    elif not 5 <= fps <= 60:
        r.warn(f"fps = {fps}，超出常见范围 5~60")
    else:
        r.info(f"fps = {fps}")

    eps = info.get("total_episodes", 0)
    frames = info.get("total_frames", 0)
    if not eps or not frames:
        r.err(f"数据集为空: episodes={eps}, frames={frames}")
    else:
        r.info(f"episodes = {eps}, frames = {frames}, 平均 {frames / eps:.1f} 帧/episode")
        if eps < 20:
            r.warn(f"只有 {eps} 条 episode，SmolVLA 微调建议 ≥ 30~50 条")
        if frames / eps < 20:
            r.warn(f"平均 episode 只有 {frames / eps:.0f} 帧，太短（SmolVLA chunk_size=50，建议 ≥ 50 帧）")

    # ---------------------------- features ----------------------------------
    feats: dict = info.get("features", {})
    if not feats:
        r.err("info.json 里没有 features")
        return r.dump()

    for key in REQUIRED_FEATURES:
        if key not in feats:
            r.err(f"缺少必需 feature: {key}")

    state_shape = tuple(feats.get("observation.state", {}).get("shape", ()))
    action_shape = tuple(feats.get("action", {}).get("shape", ()))
    if state_shape and action_shape and state_shape != action_shape:
        r.warn(f"state {state_shape} 与 action {action_shape} 维度不同（通常二者一致）")
    if state_shape:
        r.info(f"state / action 维度 = {state_shape[0] if state_shape else '?'}")

    cams = [k for k in feats if k.startswith(IMAGE_PREFIX)]
    if not cams:
        r.err("没有任何 observation.images.* 相机特征")
    else:
        r.info(f"相机: {cams}")
        for c in cams:
            f = feats[c]
            if f.get("dtype") not in ("video", "image"):
                r.warn(f"{c} 的 dtype = {f.get('dtype')!r}，预期 'video'")
        if args.expect_cams is not None and len(cams) != args.expect_cams:
            r.err(f"相机数量 {len(cams)} != 期望 {args.expect_cams}")

    tasks_file = root / "meta" / "tasks.parquet"
    if not tasks_file.is_file():
        r.warn("缺少 meta/tasks.parquet，没有语言指令，SmolVLA 的语言输入会是空的")
    else:
        try:
            import pandas as pd  # noqa: PLC0415

            n_tasks = len(pd.read_parquet(tasks_file))
            if n_tasks == 0:
                r.warn("meta/tasks.parquet 里没有任何 task")
            else:
                r.info(f"task 数量 = {n_tasks}")
        except Exception as e:  # noqa: BLE001
            r.warn(f"读取 meta/tasks.parquet 失败: {type(e).__name__}: {e}")

    # ---------------------------- 文件完整性 ---------------------------------
    for sub in ("data", "videos", "meta/stats.json"):
        if not (root / sub).exists():
            r.err(f"缺少 {sub}")

    parquets = list((root / "data").rglob("*.parquet")) if (root / "data").is_dir() else []
    if not parquets:
        r.err("data/ 下没有 parquet 文件（上传中断？）")
    else:
        r.info(f"parquet 文件 {len(parquets)} 个")

    videos = list((root / "videos").rglob("*.mp4")) if (root / "videos").is_dir() else []
    if cams and not videos:
        r.err("videos/ 下没有 mp4 文件")
    else:
        total_mb = sum(v.stat().st_size for v in videos) / 2**20
        r.info(f"mp4 文件 {len(videos)} 个，共 {total_mb:.1f} MiB")

    # 检查是不是被 rsync 传坏了（0 字节文件是典型症状）
    broken = [p for p in parquets + videos if p.stat().st_size == 0]
    if broken:
        r.err(f"{len(broken)} 个文件是 0 字节，传输可能中断: {[str(p.relative_to(root)) for p in broken[:3]]}")

    # ---------------------------- 解码验证 ----------------------------------
    if args.decode and videos:
        import torch  # noqa: PLC0415

        from lerobot.utils.import_utils import get_safe_default_video_backend  # noqa: PLC0415

        backend = get_safe_default_video_backend()
        r.info(f"video backend = {backend}")
        if backend != "torchcodec":
            r.warn("torchcodec 不可用，会退回 pyav（慢）。见 docs/environment.md 的 F1/F2")

        try:
            from torchcodec.decoders import VideoDecoder  # noqa: PLC0415

            v = sorted(videos)[0]
            dec = VideoDecoder(str(v))
            frame = dec[0]
            r.info(f"解码抽检 {v.name}: {len(dec)} 帧, 首帧 {tuple(frame.shape)} {frame.dtype}")
        except Exception as e:  # noqa: BLE001
            r.err(f"视频解码失败: {type(e).__name__}: {e}")

    print()
    return r.dump()


if __name__ == "__main__":
    sys.exit(main())
