# VqTrajTokenizer

VQ-VAE 风格的轨迹 tokenizer。把轨迹切成等长段，每段经 encoder → 连续 latent
→ codebook 最近邻量化得到一个离散 token，每段对应**一个 latent control
unit**。Planner 训练时只需要预测这串 token，词表大小 = `codebook_size`。

## Pipeline

```
        trajectory  τ ∈ R^{B×T×C_in}
                │
                │  (1) per-step diff:   d_t = τ_t − τ_{t-1}     (d_0 = τ_0)
                │      避免绝对坐标随时间漂移，让 codebook 跨段共享形状
                ▼
        d ∈ R^{B×T×C_in}
                │
                │  (2) /norm_scale  →  reshape:  (B, S, L*C_in)   L = T / S
                ▼
        encoder MLP + LayerNorm  ──→  z_e ∈ R^{B×S×D}
                │
                │  (3) VQ:  idx = argmin_k ||z_e − e_k||²
                ▼
        z_q = e[idx]   |   STE:   z_q = z_e + sg(z_q − z_e)
                │
                │  (4) decoder MLP  →  reshape:  (B, T, C_out)  →  *norm_scale
                ▼
        d̂ ∈ R^{B×T×C_out}
                │
                │  (5) cumsum (恢复绝对坐标)
                ▼
        reconstructed trajectory  τ̂ ∈ R^{B×T×C_out}
```

> ⚠️ **关键设计点**：encoder 看到的是 **per-step diff** 而不是原始坐标。
> 否则同一种"形状"在轨迹不同位置（坐标差 100m+）会被映射成不同 latent，
> 导致 codebook 必须为每个位置各存一个 code，迅速塌缩。

## 关键参数

| 符号 | 默认 | 含义 |
| --- | --- | --- |
| `T = num_future_points` | 64    | 稠密未来点数 |
| `S = num_segments`      | 8     | skill 段数（= Planner 序列长） |
| `L = T / S`             | 8     | 段内稠密点数 |
| `D = latent_dim`        | 64    | 每段 latent 维度 |
| `K = codebook_size`     | 1024  | 词表大小（CE 类别数） |
| `commit_beta`           | 0.25  | commitment loss 权重 |
| `norm_scale`            | 50.0  | per-step diff 的归一化尺度（demo 用 5.0） |

## 损失函数

```
loss_recon   = SmoothL1(d̂_n, d_n)             # 在归一化 diff 空间监督
loss_vq      = ||sg(z_e) − e[idx]||²           # codebook loss
             + β · ||z_e − sg(e[idx])||²       # commitment loss
loss         = loss_recon + loss_vq
```

监控指标（不计入 loss）：

```
loss_recon_xy = SmoothL1(τ̂, τ)                # 还原坐标后的真实重建误差
perplexity    = exp(−Σ p log p)                # batch 内 code 使用熵，越接近 K 越好
```

## 与 [DxyBinTokenizer](../binTokenizer/binTokenizer.py) 的对比

| | DxyBinTokenizer | VqTrajTokenizer |
| --- | --- | --- |
| 量化方式 | 1D bin（硬规则 `argmin`） | 学习的 codebook（VQ-VAE） |
| 是否可学习 | encoder/decoder MLP 可学，bin 中心固定 | encoder/decoder/codebook 全可学 |
| Planner 词表 | `B_x + B_y`（例如 128） | `codebook_size`（例如 1024） |
| Planner 序列长 | `2S`（dx/dy 交织） | `S`（一段一 token） |
| 每段表达 | 受限于矩形格子 | 任意形状（受 codebook 多样性限制） |
| 训练 loss | 量化 MSE + SmoothL1 重建 | SmoothL1（diff 域）+ codebook + commitment |
| 解释性 | 强（每个 bin = 物理位移） | 弱（每个 code = 抽象 latent） |
| 何时用 | 想要可解释、零训练即可量化 | 想要更紧凑的词表 / 学习式表达 |

## 给 Planner 用的接口

```python
from TrajTokenizers.vqTokenizer import VqTrajTokenizer

tokenizer = VqTrajTokenizer(
    num_future_points=64, num_segments=8, codebook_size=1024,
)
# 训练前先把 tokenizer 跑收敛（detach codebook）

# 1. 把 GT 轨迹编码成 token
gt_tokens = tokenizer.encode(gt_traj)               # (B, 8) long, ∈ [0, 1024)

# 2. Planner 预测 logits (B, 8, 1024) → CE
loss_planner = F.cross_entropy(
    logits.reshape(-1, tokenizer.vocab_size),
    gt_tokens.reshape(-1),
)

# 3. 推理时把预测 token 喂回 decode 还原轨迹
pred_traj = tokenizer.decode(ego_lcf, pred_tokens)  # (B, 64, 2)
```

## Demo 结果速览

[demos/demo_vq.py](../demos/demo_vq.py) 在 mock 数据（直行/左转/右转，T=64, S=8, K=512）
上训练 2000 步：

| 类型 | 8 个 token | 含义 |
| --- | --- | --- |
| straight | `[323, 323, 323, 323, 323, 323, 323, 323]` | 全程一种"匀速直行"模式 |
| left     | `[323, 323, 240, 240, 240, 240, 240, 240]` | 前 2 段直行 → 后 6 段左转 |
| right    | `[323, 323, 323, 313, 313, 313, 313, 313]` | 前 3 段直行 → 后 5 段右转 |

- `loss_recon_xy ≈ 6m`（轨迹长 ~250m，相对误差 ~2.4%）
- `perplexity ≈ 2.7` —— 受限于 mock 数据只有 3 类，真实数据会用更多 code
- 输出图：[vq_recon.png](../demos/outputs/vq_recon.png)、[vq_codebook_usage.png](../demos/outputs/vq_codebook_usage.png)

## 实现要点 / 调优经验

1. **encoder 输入用 diff，不用绝对坐标**。否则跨段无法共享 code，重建会
   退化成"水平阶梯段"。
2. **encoder 末尾接 LayerNorm**，让 latent 维度方差≈1；codebook 用
   `N(0, 1/√D)` 初始化（默认 `±1/K` 太小），避免量级不匹配让 commitment
   loss 爆炸。
3. **norm_scale 选成 per-step 位移的典型量级**（demo 中 vel·dt ≈ 4m，所以
   取 5；如果直接编码绝对坐标要 50+）。
4. **STE 梯度**：`z_q = z_e + (z_q - z_e).detach()`，反向直接从 decoder
   流回 encoder，跳过 codebook 的不可导 `argmin`。
5. **codebook 塌缩对策**（如果 perplexity 始终 << K）：
   - 调小 `commit_beta`（让 encoder 不必紧贴 codebook）
   - 改成 **EMA 更新 codebook**（本实现是 STE + 反向 SGD，简单但弱）
   - 周期性 reset 长期未命中的 code 到当前 batch 的随机 z_e
6. **解码侧条件化**：当前 [decode](./vqTokenizer.py) 没用 `ego_lcf`（接口保留）。
   想做条件化解码，把 `ego_lcf` 拼到 decoder 输入即可。
7. **diff 域 vs 坐标域监督**：本实现 `loss_recon` 在归一化 diff 上算，等
   价于鼓励 decoder 输出"局部形状正确"。坐标域 `loss_recon_xy` 仅用作
   监控（cumsum 会放大段内每步误差）。


* 由于是 mock 数据所以效果不太好。