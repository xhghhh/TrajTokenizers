# DxyBinTokenizer 真实数据实验报告

在 `egomotion.chunk_0000`（100 个 parquet，约 5K 滑窗样本）上训练并评估 bin
tokenizer 的轨迹离散化质量，结论：**5 个 epoch 即收敛到 ADE_future = 0.59m
/ FDE_future = 1.04m**，作为 Planner 的离散动作词表已经足够精细。

## 1. 实验设置

| 项 | 值 |
| --- | --- |
| 数据 | `egomotion.chunk_0000/*.parquet`（100 个 session） |
| 重采样 | 严格 10Hz；>200ms gap 处切段 |
| 窗口 | 80 点（16 历史 + 64 未来），stride = 1.0s |
| 切分 | 80 / 20 split by file（4268 train / 1105 val） |
| 坐标系 | 以 anchor=traj[15] 为原点的 ego frame；bin 内部再平移到 traj[0]→0 |
| 段配置 | 80 点 / 10 段（每段 8 步 = 0.8s） |
| Bin 范围 | `dx_range=(-1, 25) m`，`dy_range=(-7, 7) m`（来自 shifted 系下分位数） |
| Bin 分辨率 | 64×64（dx ≈ 0.41m，dy ≈ 0.22m） |
| 词表 / 序列 | `vocab_size=128`，`seq_len=20`（dx/dy 交织） |
| 模型 | encoder/decoder MLP，0.17M 参数 |
| 优化器 | Adam, lr=2e-4, batch_size=64, grad_clip=1.0 |

## 2. 收敛曲线

| epoch | train loss | val ADE_full | val FDE_full | val ADE_fut | val FDE_fut |
| --- | --- | --- | --- | --- | --- |
| 1 | 6.998 | 6.586 | 20.402 | 7.960 | 20.402 |
| 2 | 0.834 | 1.682 | 4.840 | 1.882 | 4.840 |
| 3 | 0.444 | 0.859 | 1.950 | 0.922 | 1.950 |
| 4 | 0.274 | 0.621 | 1.111 | 0.639 | 1.111 |
| **5** | **0.244** | **0.581** | **1.044** | **0.594** | **1.044** |

> 训练 loss 由三项组成：`loss_quant`（bin 量化 MSE）、`loss_state`（5 维状态
> SmoothL1）、`loss_control`（2 维控制 SmoothL1）。`loss_quant ≈ 0.009` 全程
> 不变，说明 bin range 设的合理（量化误差稳定在分辨率 1/4 量级），主导收
> 敛的是 decoder MLP 学习 (dx, dy) → 段终点 5 维状态。

## 3. 按速度分桶（epoch 5）

| 速度桶 | N | ADE_fut | FDE_fut |
| --- | --- | --- | --- |
| v ∈ [0, 5) m/s | 70 | 0.97m | 1.81m |
| v ∈ [5, 10) m/s | 455 | 0.63m | 1.18m |
| v ∈ [10, 15) m/s | 346 | 0.51m | 0.88m |
| v ∈ [15, 25) m/s | 233 | 0.55m | 0.80m |
| v ∈ [25, ∞) m/s | 1 | 0.51m | 0.89m |

观察：
- **低速段误差最大**（0.97m / 1.81m）：低速场景 = 启停 / 转弯 / 排队，状态
  变化复杂，5 维状态 MLP 较难精确还原段内中间细节。
- **中高速段最干净**（≤0.55m）：直行段为主，bin 网格分辨率绰绰有余。
- 高速桶样本极少（N=1），统计意义有限。

## 4. 可视化

[real_recon_bin.png](../demos/outputs/real_recon_bin.png)：从 val set 按速度
随机抽 8 个样本，展示 GT history（淡色） + GT future（黑色） + 重建（红色），
绿点是 anchor。

[data_stats.png](../demos/outputs/data_stats.png)：训练集的 per-step / per-
segment Δxy 分布与 anchor 速度直方图，用于 bin range 的合理性判断。

## 5. 复现

```bash
# 数据画像（可选）：决定 bin range
python -m TrajTokenizers.demos.analyze_data

# bin tokenizer 训练 + 评估（默认 10 epoch；5 epoch 已收敛）
python -m TrajTokenizers.demos.train_eval_real --tokenizer bin --epochs 5
```

## 6. 关键观察

1. **bin 在真实数据上的"零调参"特性凸显**：把 dx/dy 范围用数据分位数定好
   后，量化误差稳定，剩下只需要训练 decoder MLP 把 (Δx, Δy) 还原成段终点
   5 维状态。
2. **大尺度的速度跨度（0~30 m/s）不影响 bin**：bin 离散的是位移而非速度，
   不同速度的样本天然落在不同 bin 上。
3. **VqTrajTokenizer 在同一 setup 下严重塌缩**（perplexity≈3/512，ADE_fut
   ≈10m），见对比报告。bin 在小词表（128）下反而完胜更大词表（512）的 vq。

## 7. 局限与改进方向

- **段内细节丢失**：bin 一段只输出一个 (Δx, Δy)，段内 8 步中间形状靠 MLP
  外推，急刹/急转场景可能不够精确（→ 提升 num_segments 到 16，每段 5 步）。
- **Y 方向分辨率紧**：dy_range=(-7, 7) 对极端转弯样本会被 clamp，可考虑
  扩到 (-10, 10) 或者非均匀 bin。
- **5 维状态损失依赖 ego_lcf**：decoder 用 ego_lcf 重建 sin_yaw/cos_yaw/
  vel，需要保证下游推理能拿到一致的 ego_lcf。

## 8. 给 Planner 的接口建议

- Planner 输入：history 16 点的连续特征（不需要 token 化）
- Planner 输出：每个 token 位置 64 类的 logits（dx 头 + dy 头交织），20 个
  位置共 20 个 CE
- 推理：拿 20 个 token → bin tokenizer.decode → 10 个段终点 → 还原成完整
  轨迹（按需插值或外推到 64 个稠密未来点）
