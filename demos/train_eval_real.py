"""\u5728\u771f\u5b9e egomotion \u6570\u636e\u4e0a\u8bad\u7ec3 + \u8bc4\u4f30\u4e24\u79cd tokenizer\u3002

usage:
    python -m TrajTokenizers.demos.train_eval_real --tokenizer vq
    python -m TrajTokenizers.demos.train_eval_real --tokenizer bin

\u8bad\u7ec3\u8f93\u5165\u662f dataset \u7edf\u4e00\u8f93\u51fa\u7684 ``trajectory: (80, 7)``\uff0c
tokenizer \u4e0d\u533a\u5206 history/future\uff0c\u7edf\u4e00\u770b 80 \u70b9\u3002

\u8bc4\u4f30\u6307\u6807\uff1aADE / FDE\uff08\u5168\u8f68\u8ff9\u4ee5\u53ca\u5728\u672a\u6765 64 \u70b9\u4e0a\uff09\uff0c\u6309\u901f\u5ea6\u5206\u6876\u3002
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TrajTokenizers.dataset import EgomotionDataset
from TrajTokenizers.binTokenizer.binTokenizer import DxyBinTokenizer
from TrajTokenizers.vqTokenizer.vqTokenizer import VqTrajTokenizer


# ---------------------------------------------------------------------------
# Tokenizer factory
# ---------------------------------------------------------------------------
def build_tokenizer(name: str, T: int, S: int):
    """\u521b\u5efa tokenizer\u3002T = 80 \u603b\u70b9\u6570\uff0cS = 10 \u6bb5\u3002"""
    if name == "vq":
        return VqTrajTokenizer(
            num_future_points=T,
            num_segments=S,
            in_dim=2, out_dim=2,
            latent_dim=64,
            codebook_size=512,
            commit_beta=0.25,
            hidden_dim=256,
            norm_scale=2.6,            # \u6765\u81ea analyze_data \u63a8\u8350
        )
    elif name == "bin":
        # bin 的 num_skills = dx+dy token 总数，所以要 2*S
        # range 来自 shifted 坐标系（traj[0]→0）下的 80 点 / 10 段累计 dx/dy 实测分位数
        return DxyBinTokenizer(
            num_future_points=T,
            num_skills=2 * S,
            dx_range=(-1.0, 25.0),
            dy_range=(-7.0, 7.0),
            num_dx_bins=64,
            num_dy_bins=64,
            hidden_dim=256,
        )
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Reconstruct: 两种 tokenizer 统一接口
#   - vq:  输出稠密 (B, T, 2)
#   - bin: 输出稀疏 (B, S, 2)（每段终点），训练/评估都在 shifted 坐标系下做
# 返回统一的 10 个段终点 (B, S, 2)，保证公平对比。
# ---------------------------------------------------------------------------
SEG_LEN = 8     # 80 点 / 10 段


def _seg_end_indices(T: int, S: int) -> torch.Tensor:
    return torch.arange(0, S) * (T // S) + (T // S) - 1   # [7, 15, ..., 79]


def _shift_to_first(traj: torch.Tensor) -> torch.Tensor:
    """把 trajectory 的第 0 点平移到原点（只动 x, y 两维）。"""
    out = traj.clone()
    out[..., :2] = out[..., :2] - out[:, 0:1, :2]
    return out


def reconstruct_seg_ends(tokenizer, traj: torch.Tensor, ego_lcf: torch.Tensor
                         ) -> torch.Tensor:
    """返回重建的段终点 (B, S, 2)，absolute 坐标系 (跟 GT 一致)。"""
    B, T, _ = traj.shape
    if isinstance(tokenizer, VqTrajTokenizer):
        tokens = tokenizer.encode(traj[..., :2])
        recon_full = tokenizer.decode(ego_lcf, tokens)              # (B, T, 2)
        idx = _seg_end_indices(T, tokenizer.num_skills).to(recon_full.device)
        return recon_full.index_select(1, idx)                       # (B, S, 2)
    elif isinstance(tokenizer, DxyBinTokenizer):
        traj_shift = _shift_to_first(traj)
        tokens = tokenizer.encode(traj_shift)
        recon = tokenizer.decode(ego_lcf, tokens)[..., :2]           # (B, S, 2) shifted
        recon = recon + traj[:, 0:1, :2]                              # un-shift
        return recon
    raise TypeError(type(tokenizer))


def reconstruct_full(tokenizer, traj: torch.Tensor, ego_lcf: torch.Tensor
                     ) -> torch.Tensor:
    """仅用于可视化：返回重建轨迹。vq 是稠密 80 点，bin 在段终点之间线性插值。"""
    B, T, _ = traj.shape
    if isinstance(tokenizer, VqTrajTokenizer):
        tokens = tokenizer.encode(traj[..., :2])
        return tokenizer.decode(ego_lcf, tokens)                     # (B, T, 2)
    elif isinstance(tokenizer, DxyBinTokenizer):
        seg_ends = reconstruct_seg_ends(tokenizer, traj, ego_lcf)    # (B, S, 2) abs
        # 段终点 在 GT 索引 [7, 15, ..., 79]；插值起点使用 GT traj[0]
        S = seg_ends.shape[1]
        anchors = torch.cat([traj[:, 0:1, :2], seg_ends], dim=1)     # (B, S+1, 2)
        anchor_idx = torch.cat([torch.tensor([0]),
                                _seg_end_indices(T, S) + 1])         # [0, 8, 16, ..., 72, 80]
        anchor_idx = anchor_idx.to(traj.device).float()
        # 对每个样本、每个点做线性插值
        target_idx = torch.arange(T, device=traj.device).float()
        # 对 (B, S+1, 2) 在轴 1 插值到 (B, T, 2)
        # 手工实现：对每个 target_idx 找起止 anchor、插值
        idxs = torch.bucketize(target_idx, anchor_idx) - 1
        idxs = idxs.clamp(0, S - 1)
        lo = anchor_idx[idxs]; hi = anchor_idx[idxs + 1]
        w = ((target_idx - lo) / (hi - lo)).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        out = anchors[:, idxs, :] * (1 - w) + anchors[:, idxs + 1, :] * w
        return out
    raise TypeError(type(tokenizer))


# ---------------------------------------------------------------------------
# Training step (loss \u91c7\u7528\u5404 tokenizer \u81ea\u5e26\u7684 forward_train)
# ---------------------------------------------------------------------------
def train_step(tokenizer, batch, optim) -> Dict[str, float]:
    traj = batch["trajectory"]
    if isinstance(tokenizer, DxyBinTokenizer):
        traj = _shift_to_first(traj)   # bin 要求 traj[0] 在原点
    ann = {
        "trajectory": traj,
        "ego_lcf": batch["ego_lcf"],
    }
    out = tokenizer.forward_train(ann)
    loss = out["loss"]
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
    optim.step()
    return {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
            for k, v in out.items()}


# ---------------------------------------------------------------------------
# Eval: \u8ba1\u7b97 ADE / FDE\uff0c\u6309\u901f\u5ea6\u5206\u6876
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(tokenizer, loader, device, hist_len: int = 16, fut_len: int = 64
             ) -> Dict[str, float]:
    """两种 tokenizer 都在 段终点 (S=10 个点) 上计算 ADE/FDE，在 GT 索引
    [7,15,...,79] 处；ADE_full = 10 个段终点均值，ADE_future = 未来 8 个段终点均值。"""
    tokenizer.eval()
    ade_full, fde_full = [], []
    ade_fut, fde_fut = [], []
    speed_anchor = []

    for batch in loader:
        traj = batch["trajectory"].to(device)        # (B, 80, 7)
        ego = batch["ego_lcf"].to(device)
        seg_recon = reconstruct_seg_ends(tokenizer, traj, ego)   # (B, S, 2)
        S = seg_recon.shape[1]
        seg_idx = _seg_end_indices(traj.shape[1], S).to(device)
        seg_gt = traj[..., :2].index_select(1, seg_idx)          # (B, S, 2)

        err = (seg_recon - seg_gt).norm(dim=-1)                  # (B, S)
        ade_full.append(err.mean(dim=1).cpu().numpy())
        fde_full.append(err[:, -1].cpu().numpy())
        # 未来部分：属于 future 的段终点 —— GT 索引 >= hist_len 的
        future_seg_mask = seg_idx >= hist_len
        err_f = err[:, future_seg_mask]
        ade_fut.append(err_f.mean(dim=1).cpu().numpy())
        fde_fut.append(err_f[:, -1].cpu().numpy())
        speed_anchor.append(ego[:, 7].cpu().numpy())

    ade_full = np.concatenate(ade_full)
    fde_full = np.concatenate(fde_full)
    ade_fut = np.concatenate(ade_fut)
    fde_fut = np.concatenate(fde_fut)
    speed = np.concatenate(speed_anchor)

    metrics = {
        "ADE_full": float(ade_full.mean()),
        "FDE_full": float(fde_full.mean()),
        "ADE_future": float(ade_fut.mean()),
        "FDE_future": float(fde_fut.mean()),
    }
    bins = [(0, 5), (5, 10), (10, 15), (15, 25), (25, 1e9)]
    for lo, hi in bins:
        mask = (speed >= lo) & (speed < hi)
        if mask.sum() == 0:
            continue
        tag = f"v[{lo:.0f},{hi:.0f})" if hi < 1e9 else f"v[{lo:.0f}+)"
        metrics[f"ADE_fut_{tag}"] = float(ade_fut[mask].mean())
        metrics[f"FDE_fut_{tag}"] = float(fde_fut[mask].mean())
        metrics[f"N_{tag}"] = int(mask.sum())
    return metrics


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
@torch.no_grad()
def visualize_recon(tokenizer, ds, out_path: str, n_samples: int = 8,
                    hist_len: int = 16):
    tokenizer.eval()
    device = next(tokenizer.parameters()).device

    # \u6309\u901f\u5ea6\u968f\u673a\u53d6\u4e0d\u540c\u6837\u672c
    speeds = np.array([ds[i]["ego_lcf"][7].item() for i in range(min(len(ds), 1000))])
    rng = np.random.default_rng(0)
    pick_idx = rng.choice(len(speeds), size=n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples // 2, figsize=(3 * (n_samples // 2), 6))
    axes = axes.flatten()
    for ax, i in zip(axes, pick_idx):
        s = ds[int(i)]
        traj = s["trajectory"].unsqueeze(0).to(device)
        ego = s["ego_lcf"].unsqueeze(0).to(device)
        recon = reconstruct_full(tokenizer, traj, ego)[0].cpu().numpy()
        gt = s["trajectory"][:, :2].numpy()
        ax.plot(gt[:hist_len, 0], gt[:hist_len, 1], "k.-", alpha=0.4, label="hist")
        ax.plot(gt[hist_len:, 0], gt[hist_len:, 1], "k.-", label="future GT")
        ax.plot(recon[:, 0], recon[:, 1], "r.-", alpha=0.7, label="recon")
        ax.scatter([0], [0], c="g", s=30, zorder=5)  # anchor
        ax.set_title(f"v={s['ego_lcf'][7].item():.1f} m/s")
        ax.axis("equal"); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", choices=["vq", "bin"], required=True)
    p.add_argument("--data_dir", default="/Users/huanglichen/Desktop/Air_AD/TrajTokenizers/egomotion.chunk_0000")
    p.add_argument("--hist_len", type=int, default=16)
    p.add_argument("--fut_len", type=int, default=64)
    p.add_argument("--num_segments", type=int, default=10)  # 80/10 = 8 step/seg = 0.8s
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", default=os.path.join(HERE, "outputs"))
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ------------------------------------------------------------------ data
    train_files, val_files = EgomotionDataset.split_files(args.data_dir, val_ratio=0.2)
    print(f"train files = {len(train_files)},  val files = {len(val_files)}")
    train_ds = EgomotionDataset(args.data_dir, hist_len=args.hist_len, fut_len=args.fut_len,
                                stride=args.stride, files=train_files)
    val_ds = EgomotionDataset(args.data_dir, hist_len=args.hist_len, fut_len=args.fut_len,
                              stride=args.stride, files=val_files)
    print(f"train samples = {len(train_ds)}, val samples = {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # ------------------------------------------------------------------ model
    T = args.hist_len + args.fut_len
    tokenizer = build_tokenizer(args.tokenizer, T=T, S=args.num_segments).to(device)
    n_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    print(f"\n[{args.tokenizer}] vocab_size = {tokenizer.vocab_size}, "
          f"sequence_len = {tokenizer.num_skills}, params = {n_params/1e6:.2f}M")

    optim = torch.optim.Adam(tokenizer.parameters(), lr=args.lr)

    # ------------------------------------------------------------------ train
    history: List[Dict] = []
    for ep in range(args.epochs):
        tokenizer.train()
        ep_log: Dict[str, float] = {}
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            log = train_step(tokenizer, batch, optim)
            for k, v in log.items():
                ep_log[k] = ep_log.get(k, 0.0) + v
        for k in ep_log:
            ep_log[k] /= max(1, len(train_loader))
        ep_log["epoch_sec"] = time.time() - t0

        val = evaluate(tokenizer, val_loader, device,
                       hist_len=args.hist_len, fut_len=args.fut_len)
        print(f"\n=== epoch {ep + 1}/{args.epochs}  ({ep_log['epoch_sec']:.1f}s) ===")
        print("  train: " + ", ".join(f"{k}={v:.4f}" for k, v in ep_log.items()
                                       if k != "epoch_sec"))
        print(f"  val  : ADE_full={val['ADE_full']:.3f}m  FDE_full={val['FDE_full']:.3f}m"
              f"  ADE_fut={val['ADE_future']:.3f}m  FDE_fut={val['FDE_future']:.3f}m")
        for k, v in val.items():
            if k.startswith("ADE_fut_v"):
                tag = k[len("ADE_fut_"):]
                fde = val[f"FDE_fut_{tag}"]; n = val[f"N_{tag}"]
                print(f"    {tag:14s} N={n:5d}  ADE={v:.3f}m  FDE={fde:.3f}m")
        history.append({"epoch": ep + 1, "train": ep_log, "val": val})

    # ------------------------------------------------------------------ visualise
    out_png = os.path.join(args.out_dir, f"real_recon_{args.tokenizer}.png")
    visualize_recon(tokenizer, val_ds, out_png, n_samples=8, hist_len=args.hist_len)
    print(f"\nrecon figure saved to {out_png}")

    # \u4fdd\u5b58 metrics history
    import json
    with open(os.path.join(args.out_dir, f"real_history_{args.tokenizer}.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
