"""
DxyBinTokenizer 的 mock 训练 + 可视化 demo。

运行方式（在 TrajTokenizers 的父目录下）：

    python -m TrajTokenizers.demos.demo_bin

或：

    python TrajTokenizers/demos/demo_bin.py
"""

import os
import sys

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    __package__ = "TrajTokenizers.demos"

import matplotlib.pyplot as plt
import torch

from ..binTokenizer.binTokenizer import DxyBinTokenizer
from ._common import build_dataset, train, visualize_xy


# ---------------------------------------------------------------------------
# Bin grid 专属可视化
# ---------------------------------------------------------------------------
@torch.no_grad()
def visualize_bins(model, traj, labels, out_path):
    """在 (dx, dy) 空间中画出 2D bin 网格，并叠加 3 类轨迹的采样点和所分配的 bin 中心。"""
    dx_c = model.dx_centers.cpu().numpy()
    dy_c = model.dy_centers.cpu().numpy()
    dx_lo, dx_hi = model.dx_range
    dy_lo, dy_hi = model.dy_range

    dx_edges = [dx_lo] + list(0.5 * (dx_c[1:] + dx_c[:-1])) + [dx_hi]
    dy_edges = [dy_lo] + list(0.5 * (dy_c[1:] + dy_c[:-1])) + [dy_hi]

    sample_ids = {}
    for i, l in enumerate(labels):
        sample_ids.setdefault(l, i)
    classes = [("straight", "tab:green"),
               ("left",     "tab:orange"),
               ("right",    "tab:purple")]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [3, 2]})

    ax = axes[0]
    for e in dx_edges:
        ax.axvline(e, color="lightgray", lw=0.4)
    for e in dy_edges:
        ax.axhline(e, color="lightgray", lw=0.4)

    gx, gy = torch.meshgrid(torch.tensor(dx_c), torch.tensor(dy_c), indexing="ij")
    ax.scatter(gx.numpy().flatten(), gy.numpy().flatten(),
               s=1, color="lightgray", label=f"bin centers ({len(dx_c)}x{len(dy_c)})")

    for cls, color in classes:
        i = sample_ids[cls]
        gt_diff = model._trajectory_to_sample_diffs(traj[i:i + 1])[0].cpu().numpy()
        tokens = model.encode(traj[i:i + 1])[0].cpu().numpy().reshape(-1, 2)
        picked = list(zip(dx_c[tokens[:, 0]], dy_c[tokens[:, 1]]))

        ax.plot(gt_diff[:, 0], gt_diff[:, 1], "-o", color=color,
                label=f"{cls} GT (dx,dy)", markersize=6)
        ax.scatter([p[0] for p in picked], [p[1] for p in picked],
                   marker="s", s=80, facecolors="none",
                   edgecolors=color, linewidths=1.8,
                   label=f"{cls} picked bin center")

    ax.set_xlim(dx_lo - 0.5, dx_hi + 0.5)
    ax.set_ylim(dy_lo - 0.5, dy_hi + 0.5)
    ax.set_xlabel("dx (m)  -- per-sample x increment")
    ax.set_ylabel("dy (m)  -- per-sample y increment")
    ax.set_title("2D bin grid = dx_bins x dy_bins  (squares = picked bin centers)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="upper left")

    # 放大左转
    ax = axes[1]
    i = sample_ids["left"]
    gt_diff = model._trajectory_to_sample_diffs(traj[i:i + 1])[0].cpu().numpy()
    tokens = model.encode(traj[i:i + 1])[0].cpu().numpy().reshape(-1, 2)
    picked = list(zip(dx_c[tokens[:, 0]], dy_c[tokens[:, 1]]))

    x_min = min(gt_diff[:, 0].min(), min(p[0] for p in picked)) - 0.5
    x_max = max(gt_diff[:, 0].max(), max(p[0] for p in picked)) + 0.5
    y_min = min(gt_diff[:, 1].min(), min(p[1] for p in picked)) - 0.5
    y_max = max(gt_diff[:, 1].max(), max(p[1] for p in picked)) + 0.5
    for e in dx_edges:
        if x_min <= e <= x_max:
            ax.axvline(e, color="lightgray", lw=0.6)
    for e in dy_edges:
        if y_min <= e <= y_max:
            ax.axhline(e, color="lightgray", lw=0.6)
    for cx in dx_c:
        if x_min <= cx <= x_max:
            for cy in dy_c:
                if y_min <= cy <= y_max:
                    ax.plot(cx, cy, ".", color="lightgray", markersize=3)
    ax.plot(gt_diff[:, 0], gt_diff[:, 1], "-o",
            color="tab:orange", label="left GT (dx,dy)")
    ax.scatter([p[0] for p in picked], [p[1] for p in picked],
               marker="s", s=140, facecolors="none",
               edgecolors="tab:orange", linewidths=2.0, label="picked bin center")
    for (gxv, gyv), (bx, by) in zip(gt_diff, picked):
        ax.plot([gxv, bx], [gyv, by], ":", color="tab:orange", lw=1)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("dx (m)")
    ax.set_ylabel("dy (m)")
    ax.set_title("Zoom-in (left turn): GT (dx,dy) -> nearest bin center (quantization error)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=9, loc="best")

    bin_size_x = (model.dx_range[1] - model.dx_range[0]) / model.num_dx_bins
    bin_size_y = (model.dy_range[1] - model.dy_range[0]) / model.num_dy_bins
    fig.suptitle(
        f"Bin grid  |  dx ∈ {model.dx_range} /{model.num_dx_bins} bins "
        f"≈ {bin_size_x:.3f} m/bin   ;   dy ∈ {model.dy_range} /{model.num_dy_bins} bins "
        f"≈ {bin_size_y:.3f} m/bin",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved bin grid to: {out_path}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    T = 12

    model = DxyBinTokenizer(
        num_future_points=T,
        num_skills=12,
        dx_range=(-2.0, 12.0),
        dy_range=(-6.0, 6.0),
        num_dx_bins=64,
        num_dy_bins=64,
        hidden_dim=128,
    )
    print(model)

    traj, lcf, labels = build_dataset(n_per_class=64, T=T)
    print(f"dataset: traj={tuple(traj.shape)}  lcf={tuple(lcf.shape)}")

    losses = train(model, traj, lcf, n_iters=1500, batch_size=32, lr=1e-3)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

    @torch.no_grad()
    def decode_for(cls, i):
        tokens = model.encode(traj[i:i + 1])
        return model.decode(lcf[i:i + 1], tokens)[0]

    def caption(cls, i):
        tk = model.encode(traj[i:i + 1])[0].tolist()
        return f"tokens={tk}"

    visualize_xy(
        decode_for, traj, labels, losses,
        out_path=os.path.join(out_dir, "bin_recon.png"),
        title="DxyBinTokenizer — mock data reconstruction",
        extra_caption=caption,
    )
    visualize_bins(model, traj, labels,
                   os.path.join(out_dir, "bin_grid.png"))


if __name__ == "__main__":
    main()
