"""
VqTrajTokenizer 的 mock 训练 + 可视化 demo。

运行方式：

    python -m TrajTokenizers.demos.demo_vq
    或
    python TrajTokenizers/demos/demo_vq.py
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

from ..vqTokenizer.vqTokenizer import VqTrajTokenizer
from ._common import build_dataset, train, visualize_xy


@torch.no_grad()
def visualize_codebook_usage(model, traj, labels, out_path):
    """统计三类轨迹各自使用了哪些 code。"""
    model.eval()
    classes = ["straight", "left", "right"]
    usage = {c: torch.zeros(model.codebook_size) for c in classes}
    per_sample_tokens = {}
    for cls in classes:
        idxs = [i for i, l in enumerate(labels) if l == cls]
        sub = traj[idxs]
        toks = model.encode(sub)  # (n, S)
        for k in toks.flatten().tolist():
            usage[cls][k] += 1
        per_sample_tokens[cls] = toks[0].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    colors = {"straight": "tab:green", "left": "tab:orange", "right": "tab:purple"}
    for ax, cls in zip(axes, classes):
        ax.bar(range(model.codebook_size), usage[cls].numpy(),
               color=colors[cls], width=1.0)
        nz = (usage[cls] > 0).sum().item()
        ax.set_title(f"{cls}: used codes = {nz}/{model.codebook_size}\n"
                     f"sample tokens = {per_sample_tokens[cls]}",
                     fontsize=10)
        ax.set_xlabel("code id")
        ax.set_ylabel("count")
        ax.set_xlim(-0.5, model.codebook_size - 0.5)

    fig.suptitle(
        f"VQ codebook usage (K={model.codebook_size}, S={model.num_segments})",
        fontsize=13,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved codebook usage to: {out_path}")


def main():
    torch.manual_seed(0)
    T = 64
    S = 8

    model = VqTrajTokenizer(
        num_future_points=T,
        num_segments=S,
        in_dim=2,
        out_dim=2,
        latent_dim=64,
        codebook_size=512,
        commit_beta=0.25,
        hidden_dim=256,
        norm_scale=5.0,        # per-step diff 的尺度
    )
    print(model)

    # T=64, dt=0.5 -> 32s 视野；用 yaw_rate_scale 调小弯曲程度避免轨迹爆炸
    traj, lcf, labels = build_dataset(
        n_per_class=128, T=T, dt=0.5, yaw_rate_scale=0.25,
    )
    print(f"dataset: traj={tuple(traj.shape)}  lcf={tuple(lcf.shape)}")

    losses = train(model, traj, lcf, n_iters=2000, batch_size=64, lr=1e-3)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

    @torch.no_grad()
    def decode_for(cls, i):
        tokens = model.encode(traj[i:i + 1])
        recon = model.decode(lcf[i:i + 1], tokens)[0]   # (T, 2)
        return recon

    def caption(cls, i):
        tk = model.encode(traj[i:i + 1])[0].tolist()
        return f"tokens (S={len(tk)}, V={model.codebook_size})={tk}"

    visualize_xy(
        decode_for, traj, labels, losses,
        out_path=os.path.join(out_dir, "vq_recon.png"),
        title=f"VqTrajTokenizer — T={T}, S={S}, K={model.codebook_size}",
        extra_caption=caption,
    )
    visualize_codebook_usage(
        model, traj, labels,
        out_path=os.path.join(out_dir, "vq_codebook_usage.png"),
    )


if __name__ == "__main__":
    main()
