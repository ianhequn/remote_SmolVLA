#!/usr/bin/env python
"""SmolVLA 单卡 RTX3090 训练显存 / 速度基准。

用合成张量跑真实的 forward + backward + optimizer.step，扫描 batch size，
输出峰值显存与单步耗时，用来决定实际训练的 batch_size。

用法：
    conda activate /home/zyy00005602/.conda/envs/lerobot
    python scripts/bench_train_mem.py                        # 默认 1 相机（ArmPi Ultra 配置）
    python scripts/bench_train_mem.py --cams 3               # 模拟 smolvla_base 默认的 3 相机
    python scripts/bench_train_mem.py --batches 16 32 64 128
    python scripts/bench_train_mem.py --gradient-checkpointing
"""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_HF_HOME = "/home/zyy00005602/.cache/huggingface"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)

import torch  # noqa: E402

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402

IMG = 256
CHUNK = 50
ACTION_DIM = 6
STATE_DIM = 6
TASK = "pick up the red cube and place it in the box"


def loss_of(out):
    """lerobot 各策略的 loss 返回形式不完全统一，这里做个兼容。"""
    if isinstance(out, dict):
        for k in ("loss", "losses"):
            if k in out:
                return out[k]
        scalars = [v for v in out.values() if isinstance(v, torch.Tensor) and v.ndim == 0]
        if scalars:
            return scalars[0]
        raise KeyError(f"返回值里找不到标量 loss: {list(out)}")
    return out[0] if isinstance(out, tuple) else out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="lerobot/smolvla_base")
    ap.add_argument("--cams", type=int, default=1, help="相机数量（ArmPi Ultra = 1）")
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 32, 64, 96, 128])
    ap.add_argument("--steps", type=int, default=3, help="每个 batch size 计时步数")
    ap.add_argument("--gradient-checkpointing", action="store_true", help="开启梯度检查点（省显存）")
    args = ap.parse_args()

    print(f"配置: cams={args.cams}  chunk_size={CHUNK}  action_dim={ACTION_DIM}  AMP=bf16")
    print(f"      gradient_checkpointing={args.gradient_checkpointing}")

    policy = SmolVLAPolicy.from_pretrained(args.model)
    if args.gradient_checkpointing:
        policy.config.gradient_checkpointing = True
    pre = PolicyProcessorPipeline.from_pretrained(
        args.model, config_filename="policy_preprocessor.json"
    )
    policy = policy.to("cuda").train()
    policy.zero_grad()

    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    print(f"可训练参数: {sum(p.numel() for p in params) / 1e6:.1f}M")

    def make_batch(bs: int) -> dict:
        torch.manual_seed(0)
        raw = {"observation.state": torch.randn(bs, STATE_DIM), "task": [TASK] * bs}
        for i in range(1, args.cams + 1):
            raw[f"observation.images.camera{i}"] = torch.rand(bs, 3, IMG, IMG)
        b = pre(raw)
        b = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        b["action"] = torch.randn(bs, CHUNK, ACTION_DIM, device="cuda")
        return b

    def run_step(b) -> float:
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = policy.forward(b)
        loss = loss_of(out)
        loss.backward()
        opt.step()
        return loss.item()

    print()
    print(f"{'batch':>6} | {'峰值显存 GiB':>13} | {'ms/step':>9} | {'样本/s':>8} | 结果")
    print("-" * 62)

    results: dict[int, tuple[float, float]] = {}
    for bs in args.batches:
        try:
            torch.cuda.empty_cache()
            b = make_batch(bs)
            run_step(b)  # warmup（首次含 kernel 编译）
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            loss_val = 0.0
            for _ in range(args.steps):
                loss_val = run_step(b)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / args.steps

            peak = torch.cuda.max_memory_allocated() / 2**30
            results[bs] = (peak, dt)
            print(f"{bs:>6} | {peak:>13.2f} | {dt * 1000:>9.0f} | {bs / dt:>8.0f} | ok  loss={loss_val:.4f}")
            del b
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6} | {'OOM':>13} | {'-':>9} | {'-':>8} | 显存不足，停止扫描")
            torch.cuda.empty_cache()
            break
        except Exception as e:  # noqa: BLE001
            print(f"{bs:>6} | {'ERR':>13} | {'-':>9} | {'-':>8} | {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            break

    if results:
        best = max(results)
        peak, dt = results[best]
        print()
        print(f"最大可跑 batch = {best}  (峰值 {peak:.2f} GiB, {dt * 1000:.0f} ms/step, {best / dt:.0f} 样本/s)")
        print("注意: lerobot 0.6.1 没有梯度累积，有效 batch = batch_size × GPU 数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
