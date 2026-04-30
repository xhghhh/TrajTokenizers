"""Shared helpers for the demo scripts (mock data, training loop, plotting)."""

import math
import os
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------
def gen_trajectory(
    vel: float,
    yaw_rate: float,
    kappa: float = 0.0,
    acc: float = 0.0,
    T: int = 12,
    dt: float = 0.5,
):
    """Per step: [x, y, sin_yaw, cos_yaw, vel, kappa, acc]."""
    yaw = 0.0
    x, y = 0.0, 0.0
    pts = []
    for _ in range(T):
        yaw = yaw + yaw_rate * dt
        x = x + vel * math.cos(yaw) * dt
        y = y + vel * math.sin(yaw) * dt
        pts.append([x, y, math.sin(yaw), math.cos(yaw), vel, kappa, acc])
    return pts


def build_dataset(
    n_per_class: int = 64,
    T: int = 12,
    dt: float = 0.5,
    seed: int = 42,
    yaw_rate_scale: float = 1.0,
):
    """Mock dataset of straight / left / right turn trajectories."""
    rng = torch.Generator().manual_seed(seed)

    def randn():
        return float(torch.randn((), generator=rng))

    classes = [
        ("straight", 0.00 * yaw_rate_scale, 0.000),
        ("left",     0.08 * yaw_rate_scale, 0.015),
        ("right",   -0.08 * yaw_rate_scale, -0.015),
    ]

    trajs, lcfs, labels = [], [], []
    for cls, base_yaw_rate, base_kappa in classes:
        for _ in range(n_per_class):
            vel = 8.0 + randn() * 0.8
            yaw_rate = base_yaw_rate + randn() * 0.01 * yaw_rate_scale
            kappa = base_kappa + randn() * 0.003
            acc = randn() * 0.2

            pts = gen_trajectory(vel, yaw_rate, kappa=kappa, acc=acc, T=T, dt=dt)
            trajs.append(pts)
            lcfs.append([
                vel, 0.0,
                acc, 0.0,
                yaw_rate,
                4.8, 1.9,
                vel,
                kappa,
            ])
            labels.append(cls)

    traj = torch.tensor(trajs, dtype=torch.float32)   # (N, T, 7)
    lcf = torch.tensor(lcfs, dtype=torch.float32)     # (N, 9)
    return traj, lcf, labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    model,
    traj,
    lcf,
    n_iters: int = 1500,
    batch_size: int = 32,
    lr: float = 1e-3,
    log_every: int = 200,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = traj.shape[0]
    losses: List[float] = []
    for it in range(n_iters):
        idx = torch.randint(0, N, (batch_size,))
        batch = {"trajectory": traj[idx], "ego_lcf": lcf[idx]}
        out = model.forward_train(batch)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (it + 1) % log_every == 0:
            extras = " ".join(
                f"{k}={v.item():.4f}"
                for k, v in out.items()
                if k != "loss" and torch.is_tensor(v) and v.dim() == 0
            )
            print(f"iter {it + 1:5d} | loss={loss.item():.4f}  {extras}")
    return losses


# ---------------------------------------------------------------------------
# Generic XY reconstruction plot
# ---------------------------------------------------------------------------
@torch.no_grad()
def visualize_xy(
    decode_fn: Callable[[str, int], torch.Tensor],
    traj: torch.Tensor,
    labels: List[str],
    losses: List[float],
    out_path: str,
    title: str,
    classes: Tuple[str, ...] = ("straight", "left", "right"),
    extra_caption: Callable[[str, int], str] = lambda cls, i: "",
):
    """
    Generic 1x4 figure: 3 trajectory subplots (GT vs reconstruction) + loss curve.

    ``decode_fn(cls, i)`` returns the reconstructed trajectory (P, >=2) for
    sample ``i`` in class ``cls``.
    """
    sample_ids: Dict[str, int] = {}
    for i, l in enumerate(labels):
        sample_ids.setdefault(l, i)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, cls in zip(axes[:3], classes):
        i = sample_ids[cls]
        gt_xy = traj[i, :, :2].cpu().numpy()
        pred_xy = decode_fn(cls, i)[:, :2].cpu().numpy()

        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-o",
                label="GT dense", color="tab:blue", markersize=4)
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], "--s",
                label="Reconstruction", color="tab:red", markersize=6)
        cap = extra_caption(cls, i)
        ax.set_title(f"{cls}\n{cap}", fontsize=9)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    ax = axes[3]
    ax.plot(losses, color="tab:gray")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("training loss (log)")
    ax.set_title("loss curve")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved visualization to: {out_path}")
