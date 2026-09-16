#!/usr/bin/env python
"""SmolVLA 运行环境自检。

检查 lerobot / torch / torchcodec / ffmpeg 是否可用，能否从 hf-mirror 加载
SmolVLA 权重，并用合成 batch 跑一次真实前向。

用法：
    conda activate /home/zyy00005602/.conda/envs/lerobot
    python scripts/verify_env.py            # 含权重下载与前向（首次约 1~2 分钟）
    python scripts/verify_env.py --quick    # 只查版本与 GPU，不下载权重
"""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_HF_HOME = "/home/zyy00005602/.cache/huggingface"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def setup_env() -> None:
    """本仓库脚本不依赖登录 shell，自己兜底设置关键环境变量。"""
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="只检查版本与 GPU，不加载权重")
    ap.add_argument("--model", default="lerobot/smolvla_base", help="要验证的模型 id")
    args = ap.parse_args()

    setup_env()

    import torch  # noqa: PLC0415

    print("=" * 72)
    print("1) 依赖与硬件")
    print("=" * 72)

    import lerobot  # noqa: PLC0415
    import torchcodec  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    print(f"  lerobot        : {lerobot.__version__}")
    print(f"  transformers   : {transformers.__version__}")
    print(f"  torch          : {torch.__version__}  (built for cuda {torch.version.cuda})")
    print(f"  torchcodec     : {torchcodec.__version__}")
    print(f"  HF_ENDPOINT    : {os.environ.get('HF_ENDPOINT')}")
    print(f"  HF_HOME        : {os.environ.get('HF_HOME')}")

    from lerobot.utils.import_utils import get_safe_default_video_backend  # noqa: PLC0415

    backend = get_safe_default_video_backend()
    print(f"  video backend  : {backend}{'  <-- 期望 torchcodec' if backend != 'torchcodec' else ''}")

    ok = True
    if not torch.cuda.is_available():
        print("  cuda           : False   <-- 不可用！")
        ok = False
        return 1

    free, total = torch.cuda.mem_get_info()
    print(f"  cuda           : {torch.cuda.get_device_name(0)}")
    print(f"  compute cap    : {torch.cuda.get_device_capability(0)}")
    print(f"  bf16 支持      : {torch.cuda.is_bf16_supported()}")
    print(f"  VRAM           : {free / 2**30:.1f} GiB 空闲 / {total / 2**30:.1f} GiB 总量")

    # 真正跑一次 matmul，确认内核/驱动匹配
    a = torch.randn(1024, 1024, device="cuda")
    _ = (a @ a).sum().item()
    print("  GPU 计算自检   : OK")

    if args.quick:
        print("\n--quick 模式，跳过权重加载。")
        return 0 if ok else 1

    print()
    print("=" * 72)
    print(f"2) 加载 {args.model}")
    print("=" * 72)

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: PLC0415

    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(args.model)
    print(f"  加载耗时       : {time.time() - t0:.1f}s")
    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  参数量         : 总计 {n_total / 1e6:.1f}M / 可训练 {n_train / 1e6:.1f}M")
    print(f"  输入特征       : {list(policy.config.input_features)}")
    print(f"  输出动作维度   : {policy.config.action_feature.shape}")
    print(f"  chunk_size     : {policy.config.chunk_size}")

    print()
    print("=" * 72)
    print("3) 合成 batch 前向 + select_action")
    print("=" * 72)

    from lerobot.processor import PolicyProcessorPipeline  # noqa: PLC0415

    torch.manual_seed(0)
    b = 2
    raw = {"task": ["pick up the cube and place it in the box"] * b}
    for key, feat in policy.config.input_features.items():
        shape = tuple(feat.shape)
        raw[key] = torch.rand(b, *shape) if key.startswith("observation.images") else torch.randn(b, *shape)
        print(f"  {key:<40} {tuple(raw[key].shape)}")

    pre = PolicyProcessorPipeline.from_pretrained(args.model, config_filename="policy_preprocessor.json")
    batch = pre(raw)
    # 模型为 fp32，输入也要 fp32；训练时的混合精度由 lerobot-train-bf16 处理
    batch = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    policy = policy.to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        t0 = time.time()
        out = policy.select_action(batch)
        torch.cuda.synchronize()
    print(f"\n  select_action  : {tuple(out.shape)} {out.dtype}")
    print(f"  首次调用耗时   : {(time.time() - t0) * 1000:.0f} ms（含首次 kernel 编译，非稳态）")
    print(f"  峰值显存       : {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    print()
    print("=" * 72)
    print("✅ 环境自检通过")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
