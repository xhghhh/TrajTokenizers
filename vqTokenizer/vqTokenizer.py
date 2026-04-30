"""
VQ-VAE 风格的轨迹 tokenizer。

Pipeline:
    trajectory  --[encoder]-->  z_e  --[VQ codebook]-->  z_q + idx
                                                    |
                                                    v
                                               [decoder]
                                                    |
                                                    v
                                             reconstructed trajectory

每条 trajectory 被切成 ``num_segments`` 段，每段聚合成一个 latent 向量，
然后被 codebook 量化成一个离散 token。所以 Planner 看到的序列长度就是
``num_segments``，词表大小就是 ``codebook_size``。
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override as overrides

from ..baseTokenizer import BaseTokenizer


# ---------------------------------------------------------------------------
# Vector Quantizer (with straight-through estimator)
# ---------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE quantizer: nearest-neighbour lookup + STE.

    Args:
        num_codes:   K, codebook size.
        code_dim:    D, dimension of every code.
        commit_beta: weight of the commitment loss term.
    """

    def __init__(self, num_codes: int, code_dim: int, commit_beta: float = 0.25):
        super().__init__()
        self.K = int(num_codes)
        self.D = int(code_dim)
        self.commit_beta = float(commit_beta)
        self.codebook = nn.Embedding(self.K, self.D)
        # Initialize with N(0, 1/sqrt(D)) so codes match a LayerNormed latent
        # whose components have unit variance.
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0 / (self.D ** 0.5))

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (...,) long -> (..., D) embedding."""
        return self.codebook(idx)

    def quantize(self, z_e: torch.Tensor) -> torch.Tensor:
        """Hard nearest-code assignment (no gradient). Returns (...,) long."""
        flat = z_e.reshape(-1, self.D)
        # ||z||^2 + ||e||^2 - 2 z . e
        dist = (
            flat.pow(2).sum(-1, keepdim=True)
            - 2.0 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(-1)
        )
        idx = dist.argmin(-1)
        return idx.view(*z_e.shape[:-1])

    def forward(self, z_e: torch.Tensor):
        """
        z_e: (..., D)
        Returns:
            z_q: (..., D), quantized features with straight-through gradient.
            idx: (...,) long, codebook indices.
            loss: scalar, codebook + beta * commitment.
        """
        idx = self.quantize(z_e)
        z_q = self.lookup(idx)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q.detach())
        loss = codebook_loss + self.commit_beta * commit_loss

        # straight-through estimator: forward = z_q, backward = z_e
        z_q = z_e + (z_q - z_e).detach()
        return z_q, idx, loss

    def perplexity(self, idx: torch.Tensor) -> torch.Tensor:
        """Codebook usage perplexity (higher is better)."""
        flat = idx.reshape(-1)
        onehot = F.one_hot(flat, num_classes=self.K).float()
        prob = onehot.mean(0)
        eps = 1e-10
        return torch.exp(-(prob * (prob + eps).log()).sum())


# ---------------------------------------------------------------------------
# VQ-VAE Trajectory Tokenizer
# ---------------------------------------------------------------------------
class VqTrajTokenizer(BaseTokenizer):
    """
    分段 VQ-VAE：

    * 输入轨迹 ``(B, T, in_dim)`` 被均匀切成 ``S = num_segments`` 段，每段
      ``L = T / S`` 个稠密点。
    * 每段被 encoder 压成一个 ``D = latent_dim`` 维的连续 latent。
    * latent 经过 codebook 量化，得到一个属于 ``[0, codebook_size)`` 的离散
      token。
    * decoder 把每个 code 还原成段内 ``L`` 个点的 ``out_dim`` 维输出。

    Planner 训练时的目标就是预测这 ``S`` 个 token；CE 类别数 = ``codebook_size``。
    """

    def __init__(
        self,
        num_future_points: int = 64,
        num_segments: int = 8,
        in_dim: int = 2,
        out_dim: int = 2,
        latent_dim: int = 64,
        codebook_size: int = 1024,
        commit_beta: float = 0.25,
        hidden_dim: int = 256,
        norm_scale: float = 50.0,
    ):
        super().__init__(num_future_points=num_future_points, num_skills=num_segments)

        assert num_future_points % num_segments == 0, (
            f"num_future_points={num_future_points} 必须能被 num_segments="
            f"{num_segments} 整除"
        )
        self.L = num_future_points // num_segments  # 每段稠密点数
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.latent_dim = int(latent_dim)
        self.codebook_size = int(codebook_size)
        self.norm_scale = float(norm_scale)

        # encoder: 段内点 (L * in_dim) -> latent (D)
        self.encoder = nn.Sequential(
            nn.Linear(self.L * self.in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

        # codebook + STE
        self.vq = VectorQuantizer(
            num_codes=self.codebook_size,
            code_dim=self.latent_dim,
            commit_beta=commit_beta,
        )

        # decoder: latent (D) -> 段内点 (L * out_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.L * self.out_dim),
        )

    # --------------------------------------------------------------- helpers
    @property
    def vocab_size(self) -> int:
        return self.codebook_size

    @property
    def num_segments(self) -> int:
        return self.num_skills

    def _trajectory_to_diff(self, traj_xy: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C) per-step increments (第一个点以原点为参考)."""
        diff = torch.zeros_like(traj_xy)
        diff[:, 0] = traj_xy[:, 0]
        diff[:, 1:] = traj_xy[:, 1:] - traj_xy[:, :-1]
        return diff

    def _encode_segments(self, traj_xy: torch.Tensor) -> torch.Tensor:
        """traj_xy: (B, T, in_dim) -> z_e: (B, S, D).

        应用“per-step diff + 归一化”，让 codebook 只需学习“段内下一步位移模式”，
        跨段共享表示，避免路程越远让每段 latent 越压越偏。
        """
        B, T, C = traj_xy.shape
        assert T == self.num_future_points and C == self.in_dim
        diff = self._trajectory_to_diff(traj_xy)             # (B, T, C)
        diff_n = diff / self.norm_scale
        seg = diff_n.reshape(B, self.num_segments, self.L * self.in_dim)
        return self.encoder(seg)

    def _decode_segments(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: (B, S, D) -> traj: (B, T, out_dim).  输出 diff 后 cumsum 回带坐标。"""
        B, S, _ = z_q.shape
        out = self.decoder(z_q)                              # (B, S, L*out_dim)
        diff = out.reshape(B, S * self.L, self.out_dim) * self.norm_scale
        return diff.cumsum(dim=1)

    # ----------------------------------------------------------------- API
    @overrides
    def encode(
        self, trajectory: torch.Tensor, return_continuous: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            trajectory: ``(B, T, >=in_dim)``，前 ``in_dim`` 维参与 encode。
            return_continuous: 若 True，同时返回 encoder 的连续 latent ``z_e``。

        Returns:
            tokens: ``(B, num_segments)`` long, 每个值 ∈ ``[0, codebook_size)``.
        """
        traj_xy = trajectory[..., : self.in_dim]
        z_e = self._encode_segments(traj_xy)
        idx = self.vq.quantize(z_e)
        if return_continuous:
            return idx, z_e
        return idx

    @overrides
    def decode(
        self, ego_lcf: torch.Tensor, token_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        token_indices: ``(B, num_segments)`` long.
        Returns: ``(B, T, out_dim)`` reconstructed trajectory in the ego frame.

        ``ego_lcf`` 暂未被使用（保留接口一致），如果需要带 ego 信息的解码可在
        decoder 里 concat ``ego_lcf`` 即可。
        """
        del ego_lcf
        z_q = self.vq.lookup(token_indices)  # (B, S, D)
        return self._decode_segments(z_q)

    @overrides
    def forward_train(self, planning_ann_info: Dict) -> Dict:
        """
        Args:
            planning_ann_info["trajectory"]: ``(B, T, >=in_dim)``.
        Returns dict of:
            loss_recon:   per-step diff 的 SmoothL1
            loss_recon_xy: 累计后坐标的 SmoothL1 (只看不入总 loss)
            loss_vq:      codebook + commitment
            perplexity:   codebook 使用熵 (越大越好)
            loss:         总损失
        """
        traj = planning_ann_info["trajectory"]
        traj_xy = traj[..., : self.in_dim]

        z_e = self._encode_segments(traj_xy)
        z_q, idx, vq_loss = self.vq(z_e)
        recon_xy = self._decode_segments(z_q)

        # 在 diff 空间里监督 (跟 encode 同一尺度)
        gt_diff = self._trajectory_to_diff(traj_xy) / self.norm_scale
        pred_diff = self._trajectory_to_diff(recon_xy) / self.norm_scale
        recon_loss = F.smooth_l1_loss(pred_diff, gt_diff)

        with torch.no_grad():
            recon_xy_loss = F.smooth_l1_loss(recon_xy, traj_xy)
            perplexity = self.vq.perplexity(idx)

        return {
            "loss_recon": recon_loss,
            "loss_recon_xy": recon_xy_loss,
            "loss_vq": vq_loss,
            "perplexity": perplexity,
            "loss": recon_loss + vq_loss,
        }
