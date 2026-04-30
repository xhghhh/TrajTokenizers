"""Bin-based trajectory tokenizer.

The trajectory is represented by per-step deltas ``(dx, dy)``. Each
dimension is discretized independently into uniform 1-D bins, so the
vocabulary consists of ``num_dx_bins + num_dy_bins`` tokens and every
sampled waypoint contributes exactly two tokens (one for ``dx`` and one
for ``dy``).
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override as overrides

from ..baseTokenizer import BaseTokenizer


# ---------------------------------------------------------------------------
# Feature indices
# ---------------------------------------------------------------------------
# ego_lcf (Ego Local Coordinate Feature): [vx, vy, accx, accy, yaw_rate, length, width, vel_abs, kappa]
LCF_VX_IDX = 0
LCF_VY_IDX = 1
LCF_ACC_X_IDX = 2
LCF_ACC_Y_IDX = 3
LCF_YAW_RATE_IDX = 4
LCF_LENGTH_IDX = 5
LCF_WIDTH_IDX = 6
LCF_VEL_ABS_IDX = 7
LCF_KAPPA_IDX = 8

# regressed per-waypoint state: [x, y, sin(yaw), cos(yaw), vel]
X = 0
Y = 1
SIN_YAW = 2
COS_YAW = 3
VEL = 4


def _build_mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class DxyBinTokenizer(BaseTokenizer):
    """Uniform-bin tokenizer on per-step ``(dx, dy)`` deltas."""

    def __init__(
        self,
        num_future_points: int = 12,
        num_skills: int = 6,
        dx_range: Tuple[float, float] = (-2.0, 20.0),
        dy_range: Tuple[float, float] = (-5.0, 5.0),
        num_dx_bins: int = 128,
        num_dy_bins: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__(num_future_points=num_future_points, num_skills=num_skills)

        assert num_skills % 2 == 0, "num_skills must pair (dx, dy) tokens"
        assert num_future_points % (num_skills // 2) == 0, (
            "num_future_points must be divisible by num_sample_points"
        )

        self.dx_range = tuple(dx_range)
        self.dy_range = tuple(dy_range)
        self.num_dx_bins = int(num_dx_bins)
        self.num_dy_bins = int(num_dy_bins)

        # Pre-compute and register bin centres as buffers so they move with
        # the module across devices / dtypes.
        self.register_buffer(
            "dx_centers", self.uniform_binning(self.dx_range, self.num_dx_bins)
        )
        self.register_buffer(
            "dy_centers", self.uniform_binning(self.dy_range, self.num_dy_bins)
        )

        num_sample_points = self.num_skills // 2

        # Head 1: regress [x, y, sin_yaw, cos_yaw, vel] for each sampled point
        # Input  = vel_abs (1) + flattened cumulative (dx, dy) samples (P*2)
        # Output = P * 5
        self.decode_states = _build_mlp(
            in_dim=1 + num_sample_points * 2,
            out_dim=num_sample_points * 5,
            hidden=hidden_dim,
        )

        # Head 2: regress [kappa, acc] given the reconstructed states
        # Input  = [kappa, acc_x, acc_y] (3) + flattened states (P*5)
        # Output = P * 2
        self.decode_controls = _build_mlp(
            in_dim=3 + num_sample_points * 5,
            out_dim=num_sample_points * 2,
            hidden=hidden_dim,
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def uniform_binning(value_range: Tuple[float, float], num_bins: int) -> torch.Tensor:
        """Return the ``num_bins`` centres of a uniform binning of ``value_range``."""
        lo, hi = value_range
        assert hi > lo and num_bins > 0
        step = (hi - lo) / num_bins
        return torch.linspace(lo + step / 2.0, hi - step / 2.0, num_bins)

    @property
    def vocab_size(self) -> int:
        return self.num_dx_bins + self.num_dy_bins

    # ---------------------------------------------------- trajectory -> diffs
    def _trajectory_to_sample_diffs(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Convert a dense future trajectory ``(B, num_future_points, >=2)`` into
        per-sample ``(dx, dy)`` increments ``(B, num_sample_points, 2)``.

        The first two channels of ``trajectory`` are expected to be ``(x, y)``.
        """
        num_sample_points = self.num_skills // 2
        skip = self.num_future_points // num_sample_points

        xy = trajectory[..., :2]  # (B, T, 2)
        # per-step diff (prepend zero so the first diff is relative to t=0)
        step_diff = torch.zeros_like(xy)
        step_diff[:, 1:] = xy[:, 1:] - xy[:, :-1]
        step_diff[:, 0] = xy[:, 0]  # relative to ego origin

        B = step_diff.shape[0]
        step_diff = step_diff.view(B, num_sample_points, skip, 2).sum(dim=2)
        return step_diff  # (B, P, 2)

    # ---------------------------------------------------------------- encode
    @overrides
    def encode(
        self, trajectory: torch.Tensor, return_centers: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            trajectory: ``(B, num_future_points, >=2)``, first two channels
                are ``(x, y)`` in the ego frame.
            return_centers: if True also return the quantised ``(dx, dy)``
                sample tensor reconstructed from the chosen bin centres.

        Returns:
            token_indices: ``(B, num_skills)`` long tensor, dx and dy tokens
                interleaved as ``[dx_0, dy_0, dx_1, dy_1, ...]``.
            (optional) centers: ``(B, num_sample_points, 2)``.
        """
        sample_diffs = self._trajectory_to_sample_diffs(trajectory)  # B,P,2
        dx = sample_diffs[..., 0].clamp(self.dx_range[0], self.dx_range[1])
        dy = sample_diffs[..., 1].clamp(self.dy_range[0], self.dy_range[1])

        # nearest-centre assignment
        dx_idx = torch.bucketize(dx, self._bucket_edges(self.dx_centers))
        dy_idx = torch.bucketize(dy, self._bucket_edges(self.dy_centers))
        dx_idx = dx_idx.clamp(0, self.num_dx_bins - 1)
        dy_idx = dy_idx.clamp(0, self.num_dy_bins - 1)

        tokens = torch.stack((dx_idx, dy_idx), dim=-1).flatten(1)  # B, P*2

        if return_centers:
            centers = torch.stack(
                (self.dx_centers[dx_idx], self.dy_centers[dy_idx]), dim=-1
            )
            return tokens, centers
        return tokens

    @staticmethod
    def _bucket_edges(centers: torch.Tensor) -> torch.Tensor:
        """Return the midpoints between consecutive bin centres."""
        return 0.5 * (centers[1:] + centers[:-1])

    # ---------------------------------------------------------------- decode
    def compute_trajectory(self, ego_lcf, control_states) -> torch.Tensor:
        """
        compute the trajectory
        Args:
            ego_lcf ego location states features B * 9 [vx, vy, accx, accy, yaw_rate, length, width, vel_abs, kappa]
            control_states sampled trajectory points from the tokens B x H x 2, [dx, dy]
        Returns:
            trajectory: B x P x 7
        """
        # 1. regress states: x, y, sin(yaw), cos(yaw), vel
        batch_size = ego_lcf.shape[0]
        x1 = torch.cat(
            (
                ego_lcf[..., [LCF_VEL_ABS_IDX]],
                control_states.view([batch_size, -1]),
            ),
            dim=-1,
        )
        x1 = self.decode_states(x1).unflatten(-1, (-1, 5))  # BxPx5
        x1 = torch.cat(
            (
                x1[..., :SIN_YAW],
                F.normalize(x1[..., SIN_YAW:COS_YAW + 1], dim=-1),
                x1[..., VEL:],
            ),
            dim=-1,
        )

        # 2. regress controls: kappa, acc
        x2 = torch.cat(
            (
                ego_lcf[..., [LCF_KAPPA_IDX, LCF_ACC_X_IDX, LCF_ACC_Y_IDX]],
                x1.detach().flatten(1),
            ),
            dim=-1,
        )
        controls = self.decode_controls(x2).unflatten(-1, (-1, 2))  # BxPx2
        trajectory = torch.cat((x1, controls), dim=-1)  # BxPx7
        return trajectory

    @overrides
    def decode(self, ego_lcf, token_indices) -> torch.Tensor:
        """Decode interleaved ``(dx, dy)`` token indices to a trajectory."""
        indices = token_indices.unflatten(-1, (-1, 2))
        dx = self.dx_centers[indices[..., 0]]
        dy = self.dy_centers[indices[..., 1]]
        control_states = torch.stack((dx, dy), dim=2).cumsum(dim=1)
        trajectory = self.compute_trajectory(ego_lcf, control_states)
        return trajectory

    # ---------------------------------------------------------------- losses
    def states_reconstruction_error(self, trajectory_diff: torch.Tensor):
        """
        Aggregate per-step diffs into per-sample diffs, clamp into the
        valid range, then return the residual between the raw (dx, dy)
        and their nearest bin centres. Used as a quantisation-error
        auxiliary loss during training.
        """
        skip = self.num_future_points // (self.num_skills // 2)
        trajectory_len = trajectory_diff.shape[1]
        batch_size = trajectory_diff.shape[0]
        assert trajectory_len % skip == 0, "trajectory len {} skip {}".format(
            trajectory_len, skip
        )
        num_sample_points = trajectory_len // skip
        trajectory_diff = trajectory_diff.view(
            [batch_size, num_sample_points, skip, -1]
        )
        trajectory_diff = trajectory_diff.sum(dim=2)

        dx, dy = trajectory_diff.unbind(-1)
        dx = dx.clamp(self.dx_range[0], self.dx_range[1])
        dy = dy.clamp(self.dy_range[0], self.dy_range[1])

        dx_idx = torch.bucketize(dx, self._bucket_edges(self.dx_centers)).clamp(
            0, self.num_dx_bins - 1
        )
        dy_idx = torch.bucketize(dy, self._bucket_edges(self.dy_centers)).clamp(
            0, self.num_dy_bins - 1
        )

        dx_err = dx - self.dx_centers[dx_idx]
        dy_err = dy - self.dy_centers[dy_idx]
        return (dx_err.pow(2).mean() + dy_err.pow(2).mean()) * 0.5

    @overrides
    def forward_train(self, planning_ann_info: Dict) -> Dict:
        """
        Args:
            planning_ann_info: dict with keys
                - ``ego_lcf``:    ``(B, 9)``
                - ``trajectory``: ``(B, num_future_points, >=7)`` GT future,
                                  ordered as ``[x, y, sin_yaw, cos_yaw,
                                                vel, kappa, acc]``.
        Returns:
            dict of scalar losses.
        """
        ego_lcf = planning_ann_info["ego_lcf"]
        gt_traj = planning_ann_info["trajectory"]

        # 1) tokenise + quantisation loss on (dx, dy)
        tokens = self.encode(gt_traj)
        traj_diff = torch.zeros_like(gt_traj[..., :2])
        traj_diff[:, 1:] = gt_traj[:, 1:, :2] - gt_traj[:, :-1, :2]
        traj_diff[:, 0] = gt_traj[:, 0, :2]
        quant_loss = self.states_reconstruction_error(traj_diff)

        # 2) reconstruct trajectory from tokens and regress to GT (at
        #    sampled points only).
        pred_traj = self.decode(ego_lcf, tokens)  # B, P, 7
        num_sample_points = self.num_skills // 2
        skip = self.num_future_points // num_sample_points
        # Sample the GT at the end of every ``skip`` window.
        gt_sample = gt_traj[:, skip - 1::skip, :7]

        state_loss = F.smooth_l1_loss(pred_traj[..., :VEL + 1], gt_sample[..., :VEL + 1])
        control_loss = F.smooth_l1_loss(pred_traj[..., VEL + 1:], gt_sample[..., VEL + 1:])

        return {
            "loss_quant": quant_loss,
            "loss_state": state_loss,
            "loss_control": control_loss,
            "loss": quant_loss + state_loss + control_loss,
        }
