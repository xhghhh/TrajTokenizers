# TrajTokenizers

把规划/预测中的连续未来轨迹离散化成 token 序列，方便下游 Planner 用
**分类（CrossEntropy）** 的方式自回归预测。本仓库提供两套实现，共享同一个
[BaseTokenizer](./baseTokenizer.py) 抽象接口：

| Tokenizer | 思路 | 词表 | 段内表达 | 训练成本 | 文档 |
| --- | --- | --- | --- | --- | --- |
| [DxyBinTokenizer](./binTokenizer/binTokenizer.py) | 1D 均匀 bin 量化 (Δx, Δy) | `Bx + By`（默认 128） | 仅起止位移 | 零（bin 中心固定） | [binTokenizer.md](./binTokenizer/binTokenizer.md) |
| [VqTrajTokenizer](./vqTokenizer/vqTokenizer.py) | VQ-VAE，每段一个 latent code | `codebook_size`（默认 1024） | 段内 L 步全部形状 | 需要训练 encoder/decoder/codebook | [vqTokenizer.md](./vqTokenizer/vqTokenizer.md) |

> 详细 pipeline、loss 公式、调优经验请看各自的 `.md`，本 README 只做总览。

## 目录结构

```
TrajTokenizers/
├── baseTokenizer.py            # BaseTokenizer 抽象基类（统一接口）
├── binTokenizer/
│   ├── binTokenizer.py         # DxyBinTokenizer
│   └── binTokenizer.md
├── vqTokenizer/
│   ├── vqTokenizer.py          # VectorQuantizer + VqTrajTokenizer
│   └── vqTokenizer.md
└── demos/
    ├── _common.py              # mock 数据 / 训练循环 / 通用可视化
    ├── demo_bin.py             # 跑 DxyBinTokenizer
    ├── demo_vq.py              # 跑 VqTrajTokenizer
    └── outputs/                # 输出图（运行后生成）
```

## 统一接口（[BaseTokenizer](./baseTokenizer.py)）

所有 tokenizer 都满足下面四个方法，便于在 Planner 训练 / 推理流程中替换：

```python
tokens = tokenizer.encode(trajectory)               # (B, T, C) → (B, N) long
traj   = tokenizer.decode(ego_lcf, tokens)          # (B, N)    → (B, T, C_out)
loss_dict = tokenizer.forward_train(planning_ann_info)
vocab     = tokenizer.vocab_size                    # Planner CE 的类别数
```

## 快速开始

```bash
# 在仓库根目录运行
python -m TrajTokenizers.demos.demo_bin   # 输出到 demos/outputs/bin_*.png
python -m TrajTokenizers.demos.demo_vq    # 输出到 demos/outputs/vq_*.png
```

两个 demo 都用同一份 mock 数据（直行 / 左转 / 右转），方便横向对比重建效果
和 token 行为。

## 怎么选

- 想要 **零训练、强解释性**、Planner 词表小：选 `DxyBinTokenizer`。
- 想要 **段内形状细节、更紧凑的 token 序列**（一段一 token）：选
  `VqTrajTokenizer`。
- 两者的"分段"概念是统一的：把未来 `T` 个点切成 `S` 段，每段对应一个
  *latent control unit*；Planner 序列长 = `S`（VQ）或 `2S`（bin 默认 dx/dy
  独立 head）。

## 扩展新 tokenizer

继承 `BaseTokenizer` 并实现 `encode` / `decode` / `forward_train` /
`vocab_size`，放到独立子目录（如 `xxxTokenizer/`），并在 `demos/` 加一份
对应的 `demo_xxx.py` 复用 [`_common.py`](./demos/_common.py) 的 mock 数据
与训练循环即可。
