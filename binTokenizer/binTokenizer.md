# DxyBinTokenizer

对轨迹的逐步位移 `(dx, dy)` 分别做 **一维均匀分箱** 的轨迹 tokenizer。
每个采样点产生两个 token（分别对应 `dx` 和 `dy`），词表大小为
`num_dx_bins + num_dy_bins`。

## 参数约定

| 符号                 | 含义                                                  |
| -------------------- | ----------------------------------------------------- |
| `T = num_future_points` | 稠密未来轨迹点数                                   |
| `K = num_skills`        | 每个样本产生的 token 总数（`dx/dy` 交织）           |
| `P = K / 2`             | 采样点数，`T` 必须能被 `P` 整除                     |
| `skip = T / P`          | 每个采样点聚合的稠密步数                             |
| `c_x ∈ R^{B_x}`         | `dx` 方向的 bin 中心（`uniform_binning(dx_range)`） |
| `c_y ∈ R^{B_y}`         | `dy` 方向的 bin 中心（`uniform_binning(dy_range)`） |

`ego_lcf ∈ R^{B×9}`：`[vx, vy, acc_x, acc_y, yaw_rate, length, width, vel_abs, kappa]`。

重建状态 `ŝ ∈ R^{P×5}`：`[x, y, sin(yaw), cos(yaw), vel]`；
重建控制 `ĉ ∈ R^{P×2}`：`[kappa, acc]`；
输出轨迹 `τ̂ ∈ R^{P×7} = concat(ŝ, ĉ)`。

---

## Encode：trajectory → token indices

```
Input:
    trajectory τ ∈ R^{B×T×≥2}   (前两维为 (x, y))

1. 逐步位移并在采样窗口内求和:

    d_t = τ_t - τ_{t-1}                 (t = 1..T, τ_0 = 0)
    Δ_p = Σ_{t ∈ window(p)} d_t         (p = 1..P)

2. 裁剪到合法范围:

    Δx_p ← clip(Δ_p.x, dx_range)
    Δy_p ← clip(Δ_p.y, dy_range)

3. 最近 bin 分配（用中心的中点做 bucketize）:

    z_p^x = argmin_i |Δx_p - c_x[i]|
    z_p^y = argmin_j |Δy_p - c_y[j]|

4. 交织输出:

    z = [z_1^x, z_1^y, z_2^x, z_2^y, ..., z_P^x, z_P^y]  ∈ Z^{B×K}
```

---

## Decode：token indices → trajectory

```
Input:
    ego state s_ego = ego_lcf
    token indices z_{1:K}  （交织形式）
    bin centers c_x, c_y

1. 解 token 得到每采样点位移:

    for p = 1..P:
        Δx_p ← c_x[ z_p^x ]
        Δy_p ← c_y[ z_p^y ]

2. 累积积分得到相对位置:

    u_p ← Σ_{k=1..p} (Δx_k, Δy_k)

3. 回归运动状态 (MLP = decode_states):

    x1 ← decode_states( [vel_abs, flatten(u_{1:P})] )        ∈ R^{P×5}
    x1[:, sin..cos] ← normalize(x1[:, sin..cos])             # 保证单位向量

4. 回归控制量 (MLP = decode_controls):

    x2 ← decode_controls( [kappa, acc_x, acc_y, flatten(x1).detach()] ) ∈ R^{P×2}

5. 拼接输出:

    τ̂ ← concat(x1, x2)   ∈ R^{P×7}
```

---

## 训练损失 (`forward_train`)

给定 GT 轨迹，同时约束三项：

1. **量化残差** `loss_quant = MSE(Δ, c[argmin])`：度量 GT 位移到最近 bin 中心的残差，反映词表容量是否足够。
2. **状态回归** `loss_state = SmoothL1( ŝ, GT_state )`。
3. **控制回归** `loss_control = SmoothL1( ĉ, GT_control )`。

总损失：`loss = loss_quant + loss_state + loss_control`。
