"""跑全数据集统计，输出 tokenizer 参数推荐值。

usage:
    python -m TrajTokenizers.demos.analyze_data \
        --data_dir /Users/huanglichen/Desktop/Air_AD/TrajTokenizers/egomotion.chunk_0000

会输出:
  - 训练 / 验证 样本数
  - per-step Δxy 分位数 → VqTrajTokenizer 的 ``norm_scale`` 建议
  - per-segment(8 步) Δxy 分位数 → DxyBinTokenizer 的 ``dx_range`` / ``dy_range``
  - anchor 速度、曲率分布
  - 直方图保存到 demos/outputs/data_stats.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# 允许直接 ``python -m TrajTokenizers.demos.analyze_data``
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # parent of "TrajTokenizers"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TrajTokenizers.dataset import EgomotionDataset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def percentiles(name: str, arr: np.ndarray, ps=(1, 5, 25, 50, 75, 95, 99)) -> str:
    qs = np.percentile(arr, ps)
    inner = "  ".join(f"p{p}={q:+.3f}" for p, q in zip(ps, qs))
    return f"{name:30s}  N={len(arr):8d}  mean={arr.mean():+.3f} std={arr.std():.3f}\n  {inner}"


def collect_stats(ds: EgomotionDataset, hist_len: int, fut_len: int, num_segments: int = 8,
                  max_samples: int = -1):
    """把 dataset 里每个样本的关键张量拍平做统计。"""
    seg_len = fut_len // num_segments  # 段内步数

    # 收集容器
    pstep_dx, pstep_dy = [], []          # per-step diff (future)
    seg_dx, seg_dy = [], []              # per-segment 累计位移 (future)
    fut_x_end, fut_y_end = [], []        # 64 步后终点
    anchor_speed, anchor_kappa = [], []
    anchor_yaw_rate = []

    n = len(ds) if max_samples <= 0 else min(len(ds), max_samples)
    for i in range(n):
        s = ds[i]
        traj = s["trajectory_future"].numpy()  # (T, 7) [x, y, sin, cos, vel, ax, ay]
        xy = traj[:, :2]
        diff = np.diff(xy, axis=0, prepend=np.zeros((1, 2), dtype=xy.dtype))
        # 第 0 步是 (x_0 - 0)，因为 anchor 是当前帧 = 历史最后一点；future[0] 已偏离原点
        # 这里我们关心的是 "段内每步位移" 和 "段累计位移"
        pstep_dx.append(diff[:, 0])
        pstep_dy.append(diff[:, 1])

        # per-segment 累计 Δxy = xy[seg_end] - xy[seg_start]
        for s_i in range(num_segments):
            lo, hi = s_i * seg_len, (s_i + 1) * seg_len
            start_xy = np.zeros(2) if lo == 0 else xy[lo - 1]
            end_xy = xy[hi - 1]
            d = end_xy - start_xy
            seg_dx.append(d[0])
            seg_dy.append(d[1])

        fut_x_end.append(xy[-1, 0])
        fut_y_end.append(xy[-1, 1])

        ego = s["ego_lcf"].numpy()  # (9,)
        anchor_speed.append(ego[7])    # vel_abs
        anchor_kappa.append(ego[8])    # curvature
        anchor_yaw_rate.append(ego[4])

    return dict(
        pstep_dx=np.concatenate(pstep_dx),
        pstep_dy=np.concatenate(pstep_dy),
        seg_dx=np.array(seg_dx),
        seg_dy=np.array(seg_dy),
        fut_x_end=np.array(fut_x_end),
        fut_y_end=np.array(fut_y_end),
        anchor_speed=np.array(anchor_speed),
        anchor_kappa=np.array(anchor_kappa),
        anchor_yaw_rate=np.array(anchor_yaw_rate),
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_stats(stats, out_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].hist(stats["pstep_dx"], bins=80, color="steelblue")
    axes[0, 0].set_title("per-step Δx (future, m)")
    axes[0, 1].hist(stats["pstep_dy"], bins=80, color="steelblue")
    axes[0, 1].set_title("per-step Δy (future, m)")
    axes[0, 2].scatter(stats["fut_x_end"], stats["fut_y_end"], s=2, alpha=0.3, c="steelblue")
    axes[0, 2].set_title("future endpoint xy (m, ego frame)")
    axes[0, 2].set_xlabel("x"); axes[0, 2].set_ylabel("y"); axes[0, 2].axis("equal")

    axes[1, 0].hist(stats["seg_dx"], bins=80, color="darkorange")
    axes[1, 0].set_title("per-segment Δx (8-step / 0.8s, m)")
    axes[1, 1].hist(stats["seg_dy"], bins=80, color="darkorange")
    axes[1, 1].set_title("per-segment Δy (8-step / 0.8s, m)")
    axes[1, 2].hist(stats["anchor_speed"], bins=60, color="seagreen")
    axes[1, 2].set_title("anchor |v| (m/s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def recommend(stats):
    """根据分位数给两个 tokenizer 的关键参数建议。"""
    # VQ: norm_scale 用 per-step diff 的 |·| 的 95% 分位
    step_mag = np.sqrt(stats["pstep_dx"] ** 2 + stats["pstep_dy"] ** 2)
    norm_scale = float(np.percentile(step_mag, 95))

    # Bin: dx/dy range 用 per-segment 1% 和 99% 分位（稍微外扩 10%）
    def bracket(arr, p_lo=1.0, p_hi=99.0, expand=0.1):
        lo, hi = np.percentile(arr, [p_lo, p_hi])
        span = hi - lo
        return float(lo - expand * span), float(hi + expand * span)

    dx_range = bracket(stats["seg_dx"])
    dy_range = bracket(stats["seg_dy"])

    return {
        "vq_norm_scale": round(norm_scale, 2),
        "bin_dx_range": (round(dx_range[0], 2), round(dx_range[1], 2)),
        "bin_dy_range": (round(dy_range[0], 2), round(dy_range[1], 2)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/Users/huanglichen/Desktop/Air_AD/TrajTokenizers/egomotion.chunk_0000")
    parser.add_argument("--hist_len", type=int, default=16)
    parser.add_argument("--fut_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--num_segments", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_samples", type=int, default=4000,
                        help="统计取多少样本（设 -1 跑全量）")
    parser.add_argument("--out_dir", default=os.path.join(HERE, "outputs"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_files, val_files = EgomotionDataset.split_files(args.data_dir, args.val_ratio)
    print(f"train files = {len(train_files)},  val files = {len(val_files)}")

    print("\n[1/3] building train dataset ...")
    train_ds = EgomotionDataset(args.data_dir, hist_len=args.hist_len, fut_len=args.fut_len,
                                stride=args.stride, files=train_files)
    print(f"  train samples = {len(train_ds)}")

    print("\n[2/3] building val dataset ...")
    val_ds = EgomotionDataset(args.data_dir, hist_len=args.hist_len, fut_len=args.fut_len,
                              stride=args.stride, files=val_files)
    print(f"  val samples = {len(val_ds)}")

    print("\n[3/3] collecting stats ...")
    stats = collect_stats(train_ds, args.hist_len, args.fut_len, args.num_segments,
                          max_samples=args.max_samples)

    print()
    print(percentiles("future per-step Δx", stats["pstep_dx"]))
    print(percentiles("future per-step Δy", stats["pstep_dy"]))
    print(percentiles("future per-segment Δx", stats["seg_dx"]))
    print(percentiles("future per-segment Δy", stats["seg_dy"]))
    print(percentiles("anchor |v| (m/s)", stats["anchor_speed"]))
    print(percentiles("anchor curvature", stats["anchor_kappa"]))
    print(percentiles("anchor yaw_rate", stats["anchor_yaw_rate"]))

    rec = recommend(stats)
    print("\n========= recommended tokenizer params =========")
    print(f"VqTrajTokenizer:    norm_scale = {rec['vq_norm_scale']}")
    print(f"DxyBinTokenizer:    dx_range   = {rec['bin_dx_range']}")
    print(f"DxyBinTokenizer:    dy_range   = {rec['bin_dy_range']}")
    print("================================================")

    out_png = os.path.join(args.out_dir, "data_stats.png")
    plot_stats(stats, out_png)
    print(f"\nstats figure saved to {out_png}")


if __name__ == "__main__":
    main()
