# LogitKV 实现计划——公式修正版

LogitKV 最终使用：

[
\boxed{
S_{l,i}
=======

A_{l,i}
\sqrt{
(V_{l,i}W_{O,l})^\top
G_l
(V_{l,i}W_{O,l})
}
}
]

其中

[
G_l=J_l^\top F_zJ_l.
]

实现时定义：

[
Q_{l,i}
=======

(V_{l,i}W_{O,l})^\top
G_l
(V_{l,i}W_{O,l}),
]

然后：

[
R_{l,i}=\sqrt{Q_{l,i}},
\qquad
S_{l,i}=A_{l,i}R_{l,i}.
]

在 empirical Fisher 下：

[
Q_{l,i}
\approx
\frac1W
\sum_{t\in W}
\left(
g_{l,t}^\top V_iW_O
\right)^2,
]

因此实际代码计算的是：

[
\boxed{
R_{l,i}
=======

\sqrt{
\frac1W
\sum_{t\in W}
\left(
g_{l,t}^\top V_iW_O
\right)^2
}
}
]

也就是 **RMS downstream Fisher sensitivity**。

代码核心应该是：

```python
dot = torch.matmul(
    V[:, head_idx],
    grad_head.transpose(-1, -2),
)

Q = (
    dot.float()
    .square()
    .mean(dim=-1)
)

fisher_rms = torch.sqrt(
    Q.clamp_min(0.0)
    + fisher_eps
)

score = (
    base_attention_score
    * fisher_rms
)
```

而不是：

```python
score = A * Q
```

完整执行顺序仍然是：

```text
Full-cache Prefill
        ↓
capture attention scores A
capture layer outputs o_l
        ↓
Final logits
        ↓
sample y ~ p
        ↓
one torch.autograd.grad(log p_y)
        ↓
compute Q_i
        ↓
sqrt(Q_i)
        ↓
S_i = A_i sqrt(Q_i)
        ↓
CriticalKV-style Stage-1 safeguard
        ↓
Top-K KV eviction
        ↓
compressed-cache decoding
```

实现上继续保留 CriticalKV 的两阶段机制，使实验能够做到：

```text
CriticalKV:
A_i × ||V_i W_O||_1

LogitKV:
A_i × sqrt((V_i W_O)^T G_l (V_i W_O))
```

而其他 compression ratio、Stage-1 ratio、base scorer、Top-K 规则全部保持一致。

另外必须新增一个 sanity check：

```python
score_root = A * torch.sqrt(Q)

score_squared = A.square() * Q
```

因为：

```python
score_squared == score_root.square()
```

所以理论上：

```python
score_root.topk(k)
```

和：

```python
score_squared.topk(k)
```

应该得到完全相同的 KV ranking。

这可以很好地检测实现中的：

```text
sqrt 遗漏
epsilon 位置错误
负数
dtype 数值误差
GQA aggregation 错误
```

等问题。

第一版建议仍然从：

```python
LogitKVPress(
    SnapKVPress(),
    fisher_window=32,
    first_stage_ratio=0.5,
)
```

开始，并依次完成：

* [ ] `fisher_rms_sensitivity()`
* [ ] explicit `VW_O` reference test
* [ ] root-score / squared-score ranking test
* [ ] capture hooks
* [ ] single Fisher backward
* [ ] `A * sqrt(Q)` scoring
* [ ] CriticalKV Stage-1 safeguard
* [ ] in-place cache compression
* [ ] 128-token smoke test
* [ ] RULER 4K 10%
* [ ] SnapKV / CriticalKV / LogitKV 对比
* [ ] backward latency 与 peak-memory 测试

后续所有 LogitKV 理论、代码和实验都应统一使用：

[
\boxed{
S_i
===

A_i
\sqrt{
(V_iW_O)^\top
G_l
(V_iW_O)
}
}.
]
::: 
