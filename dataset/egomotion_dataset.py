"""EgomotionDataset

把 ``egomotion.chunk_xxxx/*.parquet`` 转成 (history, future) 训练样本。

流程：
  1. 读 parquet → 按 timestamp 排序
  2. 严格 10Hz 线性重采样（yaw 用 unwrap+线性，速度/加速度按时间线性插值）
  3. 凡是相邻原始时间戳 gap > ``gap_threshold_us`` 的位置打断成多个连续段
  4. 在每段上滑窗取 ``hist_len + fut_len`` 个连续点，stride 由参数控制
  5. 以窗口内 "当前时刻"（第 hist_len 个点）为原点 + yaw 朝向 +x，做 SE(2) 逆变换，
     把世界坐标转成 ego 局部坐标
  6. 输出每个样本的 dict：
       - trajectory:        (hist_len+fut_len, 7)   一整段连续轨迹（tokenizer 用这个）
       - trajectory_history:(hist_len, 7)           前 hist_len 点（Planner context）
       - trajectory_future: (fut_len, 7)            后 fut_len 点（Planner GT）
       - ego_lcf: (9,) [vx, vy, ax, ay, yaw_rate, length, width, vel_abs, kappa]

     7 维特征 = [x, y, sin_yaw, cos_yaw, vel, ax, ay]，z 直接丢掉（不影响 PnC）。
     tokenizer 不区分 history/future，看到的就是 80 点的统一轨迹。
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_to_yaw(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) → yaw around world +Z (radians)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def unwrap_yaw(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(yaw)


def world_to_ego(
    xy_world: np.ndarray,
    yaw_world: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply inverse SE(2) so that anchor becomes (0, 0, yaw=0).

    Args:
        xy_world: ``(N, 2)`` world XY.
        yaw_world: ``(N,)`` world yaw (radians).
        anchor_xy: ``(2,)`` world XY of the anchor (usually current frame).
        anchor_yaw: scalar.

    Returns:
        (xy_ego, yaw_ego)
    """
    c, s = np.cos(-anchor_yaw), np.sin(-anchor_yaw)
    R = np.array([[c, -s], [s, c]])
    xy_ego = (xy_world - anchor_xy) @ R.T
    yaw_ego = yaw_world - anchor_yaw
    yaw_ego = np.arctan2(np.sin(yaw_ego), np.cos(yaw_ego))
    return xy_ego, yaw_ego


def vec_world_to_ego(vec_world: np.ndarray, anchor_yaw: float) -> np.ndarray:
    """Rotate a (..., 2) vector field from world to ego frame (no translation)."""
    c, s = np.cos(-anchor_yaw), np.sin(-anchor_yaw)
    R = np.array([[c, -s], [s, c]])
    return vec_world @ R.T


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

@dataclass
class ResampledSegment:
    """One contiguous 10Hz segment after gap-splitting."""
    t: np.ndarray            # (N,) seconds, monotonically increasing, step = dt
    x: np.ndarray            # (N,) world
    y: np.ndarray            # (N,) world
    yaw: np.ndarray          # (N,) world (continuous, NOT wrapped)
    vx: np.ndarray           # (N,) world frame body velocities (raw column)
    vy: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    curvature: np.ndarray
    file: str = ""           # source parquet name


def _resample_linear(t_src_us: np.ndarray, t_dst_us: np.ndarray, val: np.ndarray) -> np.ndarray:
    return np.interp(t_dst_us.astype(np.float64), t_src_us.astype(np.float64), val.astype(np.float64))


def load_resampled_segments(
    parquet_path: str,
    dt: float = 0.1,
    gap_threshold_us: int = 200_000,
    min_segment_len: int = 80,
) -> List[ResampledSegment]:
    """Load one parquet, return a list of contiguous resampled segments (≥ min_segment_len)."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].values.astype(np.int64)

    # 1) split source into contiguous chunks at large gaps
    diffs = np.diff(ts)
    gap_idx = np.where(diffs > gap_threshold_us)[0]  # break AFTER index i
    chunk_bounds: List[Tuple[int, int]] = []
    start = 0
    for i in gap_idx:
        chunk_bounds.append((start, i + 1))
        start = i + 1
    chunk_bounds.append((start, len(df)))

    yaw_full = unwrap_yaw(quat_to_yaw(
        df["qx"].values, df["qy"].values, df["qz"].values, df["qw"].values
    ))

    dt_us = int(round(dt * 1e6))
    segments: List[ResampledSegment] = []
    fname = os.path.basename(parquet_path)

    for (lo, hi) in chunk_bounds:
        if hi - lo < 2:
            continue
        ts_chunk = ts[lo:hi]
        # 2) build target uniform grid covering this chunk
        t0 = int(np.ceil(ts_chunk[0] / dt_us)) * dt_us
        t1 = int(np.floor(ts_chunk[-1] / dt_us)) * dt_us
        if t1 <= t0:
            continue
        t_dst = np.arange(t0, t1 + 1, dt_us, dtype=np.int64)
        if len(t_dst) < min_segment_len:
            continue

        seg = ResampledSegment(
            t=(t_dst - t_dst[0]) * 1e-6,
            x=_resample_linear(ts_chunk, t_dst, df["x"].values[lo:hi]),
            y=_resample_linear(ts_chunk, t_dst, df["y"].values[lo:hi]),
            yaw=_resample_linear(ts_chunk, t_dst, yaw_full[lo:hi]),
            vx=_resample_linear(ts_chunk, t_dst, df["vx"].values[lo:hi]),
            vy=_resample_linear(ts_chunk, t_dst, df["vy"].values[lo:hi]),
            ax=_resample_linear(ts_chunk, t_dst, df["ax"].values[lo:hi]),
            ay=_resample_linear(ts_chunk, t_dst, df["ay"].values[lo:hi]),
            curvature=_resample_linear(ts_chunk, t_dst, df["curvature"].values[lo:hi]),
            file=fname,
        )
        segments.append(seg)
    return segments


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# 数据里没有车长/车宽，给个常见乘用车默认值，BinTokenizer 的 ego_lcf 需要这两位
DEFAULT_LENGTH = 4.7   # m
DEFAULT_WIDTH = 1.9    # m


@dataclass
class _SampleIndex:
    seg_idx: int
    start: int   # inclusive
    # window covers [start, start + hist_len + fut_len)


class EgomotionDataset(Dataset):
    """Sliding-window dataset over a directory of egomotion parquet files."""

    def __init__(
        self,
        parquet_dir: str,
        hist_len: int = 16,
        fut_len: int = 64,
        dt: float = 0.1,
        stride: int = 10,
        gap_threshold_us: int = 200_000,
        files: Optional[Sequence[str]] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.win = hist_len + fut_len
        self.dt = dt
        self.stride = stride

        if files is None:
            files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        self.files = list(files)

        self.segments: List[ResampledSegment] = []
        self.index: List[_SampleIndex] = []

        for f in self.files:
            segs = load_resampled_segments(f, dt=dt, gap_threshold_us=gap_threshold_us,
                                           min_segment_len=self.win)
            for seg in segs:
                base = len(self.segments)
                self.segments.append(seg)
                n = len(seg.t)
                # 滑窗：window 左端 start ∈ [0, n - win], step = stride
                for start in range(0, n - self.win + 1, stride):
                    self.index.append(_SampleIndex(seg_idx=base, start=start))
            if verbose:
                print(f"[EgomotionDataset] loaded {f}: {len(segs)} segs -> "
                      f"{len(self.index)} cumulative samples")

    def __len__(self) -> int:
        return len(self.index)

    # ------------------------------------------------------------------
    def _build_traj_block(
        self, seg: ResampledSegment, sl: slice, anchor_xy: np.ndarray, anchor_yaw: float
    ) -> np.ndarray:
        """Slice a segment and convert to ego frame; return (T, 7) array."""
        xy_w = np.stack([seg.x[sl], seg.y[sl]], axis=-1)
        yaw_w = seg.yaw[sl]
        xy_ego, yaw_ego = world_to_ego(xy_w, yaw_w, anchor_xy, anchor_yaw)
        # vx/vy in raw parquet are world-frame velocity components
        v_w = np.stack([seg.vx[sl], seg.vy[sl]], axis=-1)
        v_ego = vec_world_to_ego(v_w, anchor_yaw)
        a_w = np.stack([seg.ax[sl], seg.ay[sl]], axis=-1)
        a_ego = vec_world_to_ego(a_w, anchor_yaw)
        vel_abs = np.linalg.norm(v_ego, axis=-1)
        return np.stack([
            xy_ego[:, 0], xy_ego[:, 1],
            np.sin(yaw_ego), np.cos(yaw_ego),
            vel_abs,
            a_ego[:, 0], a_ego[:, 1],
        ], axis=-1).astype(np.float32)  # (T, 7)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.index[idx]
        seg = self.segments[rec.seg_idx]
        s = rec.start
        m = s + self.hist_len  # anchor (current frame) index
        e = s + self.win

        anchor_xy = np.array([seg.x[m], seg.y[m]])
        anchor_yaw = float(seg.yaw[m])

        # 直接构建整段 (hist+fut) 一次，tokenizer 用这个；再切出 history/future 给下游
        traj_full = self._build_traj_block(seg, slice(s, e), anchor_xy, anchor_yaw)
        traj_hist = traj_full[:self.hist_len]
        traj_fut = traj_full[self.hist_len:]

        # ego_lcf at anchor frame, in ego frame
        v_anchor = vec_world_to_ego(np.array([seg.vx[m], seg.vy[m]]), anchor_yaw)
        a_anchor = vec_world_to_ego(np.array([seg.ax[m], seg.ay[m]]), anchor_yaw)
        # yaw_rate ≈ d(yaw)/dt 用中心差分
        if 0 < m < len(seg.t) - 1:
            yaw_rate = (seg.yaw[m + 1] - seg.yaw[m - 1]) / (2 * self.dt)
        else:
            yaw_rate = 0.0
        vel_abs = float(np.linalg.norm(v_anchor))
        kappa = float(seg.curvature[m])
        ego_lcf = np.array([
            v_anchor[0], v_anchor[1], a_anchor[0], a_anchor[1], yaw_rate,
            DEFAULT_LENGTH, DEFAULT_WIDTH, vel_abs, kappa,
        ], dtype=np.float32)

        return {
            "trajectory": torch.from_numpy(traj_full),            # (80, 7)  ← tokenizer 用
            "trajectory_history": torch.from_numpy(traj_hist),    # (16, 7)
            "trajectory_future": torch.from_numpy(traj_fut),      # (64, 7)
            "ego_lcf": torch.from_numpy(ego_lcf),                 # (9,)
            "meta_file": seg.file,
            "meta_t_anchor": float(seg.t[m]),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def split_files(parquet_dir: str, val_ratio: float = 0.2, seed: int = 0
                    ) -> Tuple[List[str], List[str]]:
        files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        rng = np.random.default_rng(seed)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        n_val = max(1, int(len(files) * val_ratio))
        val_idx = set(idx[:n_val].tolist())
        train = [f for i, f in enumerate(files) if i not in val_idx]
        val = [f for i, f in enumerate(files) if i in val_idx]
        return train, val
