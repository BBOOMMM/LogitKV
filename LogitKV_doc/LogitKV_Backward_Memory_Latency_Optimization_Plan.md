# LogitKV 反向传播显存与计算优化方案

> 目标：解决 LogitKV 在线估计 Fisher / logit-space sensitivity 时，由反向传播带来的巨大显存占用与计算时间开销。
>
> 当前 LogitKV score：
>
> \[
> \boxed{
> S_{l,i}
> =
> A_{l,i}
> \sqrt{
> (V_{l,i}W_{O,l})^\top
> G_l
> (V_{l,i}W_{O,l})
> }
> }
> \]
>
> 其中：
>
> \[
> G_l=J_l^\top F_zJ_l,
> \qquad
> J_l=\frac{\partial z}{\partial o_l},
> \qquad
> F_z=\operatorname{diag}(p)-pp^\top.
> \]
>
> 如果直接对完整 4K/8K/16K context 保留 autograd graph 再从最终 logits backward，80GB A800 很容易 OOM，且 backward latency 也会非常高。本文给出推荐的优化路线。

---

# 1. 问题本质

LogitKV 在线版本需要：

\[
g_{l,t}=\nabla_{o_{l,t}}\log p_y,
\qquad y\sim p.
\]

如果完整上下文长度为 \(T\)，并直接执行：

```python
with torch.enable_grad():
    outputs = model(full_context)
```

autograd 会为所有层、所有 token 保存 backward 所需的 activation，包括：

```text
attention intermediates
MLP intermediates
normalization intermediates
residual tensors
saved tensors for backward
```

因此 OOM 的主要来源不是 parameter `.grad`，而是长上下文的 activation graph。

仅把：

```python
loss.backward()
```

换成：

```python
torch.autograd.grad(...)
```

虽然可以避免 parameter gradient accumulation，但不能从根本上解决完整长上下文 backward 的 activation memory。

---

# 2. 最推荐方案：Prefix No-Grad + Window Backward

这是 Online LogitKV 最应该优先实现的优化。

假设：

\[
T=4096,
\qquad
W=32.
\]

将 context 拆成：

\[
\underbrace{x_1,\dots,x_{T-W}}_{\text{prefix}}
+
\underbrace{x_{T-W+1},\dots,x_T}_{\text{probe window}}.
\]

执行：

```text
Prefix T-W tokens
        ↓
torch.no_grad()
        ↓
build detached prefix KV cache
        ↓
Last W tokens
        ↓
torch.enable_grad()
        ↓
attend to full prefix KV
        ↓
final logits
        ↓
Fisher backward / VJP
        ↓
compute LogitKV score
        ↓
compress full KV cache
        ↓
decode
```

核心思想：

> Prefix 只需要提供数值正确的 K/V，不需要参与 autograd。

---

# 3. 为什么只对最后 W 个 token backward 是合理的

当前 LogitKV 本来就使用最后 observation window 的 gradient 来估计 Fisher：

\[
\hat G_l
=
\frac1W
\sum_{t=T-W+1}^{T}
 g_{l,t}g_{l,t}^\top.
\]

其中：

\[
g_{l,t}=\nabla_{o_{l,t}}\log p_y.
\]

对于 causal Transformer，prefix token 不依赖后面的 suffix/window token。

因此，当我们只关心：

\[
\frac{\partial \log p_y}{\partial o_{l,t}},
\qquad t>T-W,
\]

prefix KV 可以作为常量使用。

只要 prefix KV 的数值与正常 full prefill 一致，则最后 \(W\) 个 token 的 suffix forward 所看到的历史信息保持一致。

因此：

```text
prefix no-grad
+
suffix grad
```

不是简单粗暴的 truncated BPTT，而是针对当前 observation-window Fisher 定义的结构性优化。

---

# 4. 推荐在线执行流程

```text
                  Context T
                      │
            split at T - W
                      │
        ┌─────────────┴─────────────┐
        │                           │
     Prefix                       Window
      T-W                           W
        │                           │
 torch.no_grad()             torch.enable_grad()
        │                           │
 build detached KV ──────────► suffix forward
                                 over full prefix
                                      │
                                 final logits
                                      │
                          sample y ~ softmax(z)
                                      │
                                  log p_y
                                      │
                            one backward / VJP
                                      │
                     gradient hooks per layer
                                      │
                       Fisher-RMS sensitivity
                                      │
                    S_i = A_i × Fisher-RMS_i
                                      │
                              KV eviction
                                      │
                                  decode
```

---

# 5. 计算量为什么会大幅下降

## 5.1 MLP backward

Full backward 需要处理约 \(T\) 个 token。

Window backward 只处理 \(W\) 个 token。

比例：

\[
\frac WT.
\]

例如：

\[
T=4096,
\qquad W=32,
\]

则：

\[
\frac{32}{4096}
=
0.0078125
\approx 0.78\%.
\]

因此 token-wise MLP backward 工作量大约降到完整 backward 的 0.78% 量级。

---

## 5.2 Attention backward

完整 causal attention 的 query-key interaction 约为：

\[
\frac{T^2}{2}.
\]

Window backward 中，仅最后 \(W\) 个 query 需要参与 backward，但仍 attend 到完整历史：

\[
O(WT).
\]

比例约：

\[
\frac{WT}{T^2/2}
=
\frac{2W}{T}.
\]

当：

\[
T=4096,
\qquad W=32,
\]

有：

\[
\frac{64}{4096}
\approx1.56\%.
\]

因此 attention backward 的主要 query-key interaction 也会下降一个数量级以上。

> 这些是复杂度层面的估算，不等于真实 wall-clock 会严格按该比例缩短；kernel、cache、launch、I/O 和模型结构都会产生固定开销。

---

# 6. 显存为什么会显著下降

完整 backward 的 token-wise activation 大致随：

\[
O(LT)
\]

增长，其中 \(L\) 为 Transformer 层数。

Window backward 只需要为：

\[
W
\]

个 suffix token 保留图，因此 backward activation 部分大致缩放为：

\[
O(LW).
\]

理论比例约：

\[
\frac WT.
\]

例如 4096→32 为约 0.78%。

实际 peak memory 不会下降到严格的 0.78%，因为还包含：

```text
model weights
full prefix KV cache
CUDA workspace
attention temporary buffers
allocator fragmentation
```

但最容易导致 OOM 的 autograd activation graph 会显著缩小。

---

# 7. Prefix forward 实现

```python
T = context_ids.shape[1]
W = press.fisher_window

prefix_ids = context_ids[:, :-W]
window_ids = context_ids[:, -W:]
```

Prefix：

```python
with torch.no_grad():
    model(
        input_ids=prefix_ids,
        past_key_values=cache,
        use_cache=True,
    )
```

随后确保 cache 不持有 graph：

```python
for layer in cache.layers:
    layer.keys = layer.keys.detach()
    layer.values = layer.values.detach()
```

---

# 8. 模型参数全部冻结

LogitKV 不需要：

\[
\frac{\partial L}{\partial W}.
\]

因此建议：

```python
for p in model.parameters():
    p.requires_grad_(False)
```

此时：

```text
model weights = constants
prefix KV = constants
suffix activations = differentiable
```

这比训练式 full backward 更符合 LogitKV 实际需求。

---

# 9. Suffix embedding 显式设为可导

如果所有 model parameters 都被冻结，而输入仍然只是 integer `input_ids`，为了确保 suffix graph 被正确建立，建议显式构造 embedding：

```python
with torch.no_grad():
    suffix_embeds = model.get_input_embeddings()(window_ids)

suffix_embeds = (
    suffix_embeds
    .detach()
    .requires_grad_(True)
)
```

然后：

```python
with torch.enable_grad():
    outputs = model(
        inputs_embeds=suffix_embeds,
        past_key_values=cache,
        use_cache=True,
        num_logits_to_keep=1,
    )
```

---

# 10. Capture 每层 attention output

LogitKV 需要：

\[
g_{l,t}=\nabla_{o_{l,t}}\log p_y.
\]

所以需要 capture 每层 `self_attn` 的 output：

```python
self._attn_outputs = {}
```

hook：

```python
def capture_hook(self, module, inputs, kwargs, output):
    layer_idx = module.layer_idx
    attn_output = output[0]
    self._attn_outputs[layer_idx] = attn_output
    return output
```

---

# 11. 更推荐：Gradient Hook，而不是最后同时保存全部 grads

相比最后统一：

```python
torch.autograd.grad(
    probe,
    all_layer_outputs,
)
```

更推荐对每层 attention output 直接：

```python
attn_output.register_hook(...)
```

这样某层 gradient 一旦产生，就可以立刻计算该层 LogitKV score，而不需要同时保存所有层的：

```text
[B, W, hidden_size]
```

gradient tensor。

---

# 12. Gradient Hook 设计

```python
def make_grad_hook(
    self,
    layer_idx,
    module,
    values,
    base_scores,
):

    def hook(grad):

        fisher_rms = self.fisher_rms_sensitivity(
            values=values,
            module=module,
            grad_output=grad,
            fisher_window=self.fisher_window,
        )

        score = (
            base_scores
            * fisher_rms
        )

        self._scores[layer_idx] = score.detach()

        return grad

    return hook
```

forward 时：

```python
attn_output.register_hook(
    make_grad_hook(...)
)
```

执行过程：

```text
gradient generated
        ↓
compute Fisher-RMS immediately
        ↓
compute LogitKV score
        ↓
store only [B, H_kv, T] score
        ↓
release gradient tensor
```

---

# 13. Fisher-RMS 计算

当前：

\[
S_{l,i}
=
A_{l,i}
\sqrt{
(V_iW_O)^\top G_l(V_iW_O)
}.
\]

经验 Fisher window 下：

\[
\sqrt{
(V_iW_O)^\top
\hat G_l
(V_iW_O)
}
=
\sqrt{
\frac1W
\sum_t
(g_{l,t}^\top V_iW_O)^2
}.
\]

不要显式生成完整：

\[
V_iW_O\in\mathbb R^{d_{model}}.
\]

利用：

\[
g^\top V_iW_O
=
V_i^\top W_Og.
\]

因此可以先：

```python
grad_head = grad_window @ W_O_h.T
```

再：

```python
dot = V_h @ grad_head.transpose(-1, -2)
```

最后：

```python
Q = dot.float().square().mean(dim=-1)

fisher_rms = torch.sqrt(
    Q.clamp_min(0.0) + eps
)
```

最终：

```python
score = attention_score * fisher_rms
```

---

# 14. SnapKV scorer 需要改成 Window-Aware

原来的 SnapKV scorer 往往假设：

```text
hidden_states length ≈ full prefill length
```

但新方案中：

```text
hidden_states length = W
keys length = T
```

因此需要单独支持：

```python
score_from_window(
    module,
    window_hidden_states,
    full_keys,
    ...
)
```

逻辑：

```text
last W query states
        ↓
attention over all T keys
        ↓
average attention over W queries
        ↓
score historical KV
```

最后 W 个 observation tokens 可直接保护，或者沿用 SnapKV 的最大 score 保护策略。

---

# 15. Window-aware SnapKV 伪代码

```python
query_states = get_query_states(
    window_hidden_states
)

attn = (
    query_states
    @ full_keys.transpose(-1, -2)
)

attn = torch.softmax(
    attn / math.sqrt(head_dim),
    dim=-1,
)

base_scores = attn.mean(dim=-2)
```

然后再做：

```text
GQA grouping
pooling
protect last W tokens
```

最终得到：

```text
[B, num_kv_heads, T]
```

---

# 16. Prefix + Window forward 后 cache 长度应等于 T

Prefix forward：

```text
cache length = T - W
```

Suffix forward：

```text
append W K/V
```

最终：

```text
cache length = T
```

此时 cache 已经是完整 context 的 KV cache，然后再做 LogitKV eviction。

---

# 17. 不要重复做 full context forward

错误方案：

```text
full context no-grad forward
        ↓
再 forward 最后 W token
```

这样会：

```text
重复 suffix computation
重复 append suffix KV
额外增加 latency
```

正确方案是：

```text
prefix no-grad forward
+
suffix grad forward
```

两段加起来就是一次完整 context prefill。

---

# 18. Fisher probe

Suffix forward 得到最终 logits：

```python
logits = outputs.logits[:, -1, :].float()

log_probs = torch.log_softmax(
    logits,
    dim=-1,
)

probs = log_probs.detach().exp()
```

sample：

```python
y = torch.multinomial(
    probs,
    num_samples=1,
    generator=generator,
)
```

objective：

```python
probe = log_probs.gather(
    -1,
    y,
).mean()
```

如果已经注册 gradient hook，可直接：

```python
probe.backward()
```

因为 model parameters 已冻结，不会产生 parameter gradient。

---

# 19. 推荐 Online LogitKV 主流程

```python
def logitkv_prefill(
    model,
    context_ids,
    cache,
    press,
):

    T = context_ids.shape[1]
    W = press.fisher_window

    prefix_ids = context_ids[:, :-W]
    window_ids = context_ids[:, -W:]

    # ==========================================
    # 1. Prefix no-grad
    # ==========================================

    with torch.no_grad():
        model(
            input_ids=prefix_ids,
            past_key_values=cache,
            use_cache=True,
        )

    detach_cache(cache)

    # ==========================================
    # 2. Differentiable suffix embedding
    # ==========================================

    with torch.no_grad():
        suffix_embeds = (
            model
            .get_input_embeddings()(window_ids)
        )

    suffix_embeds = (
        suffix_embeds
        .detach()
        .requires_grad_(True)
    )

    # ==========================================
    # 3. Window forward with graph
    # ==========================================

    with torch.enable_grad():

        with press.capture_and_register_grad_hooks(
            model,
            cache,
        ):
            outputs = model(
                inputs_embeds=suffix_embeds,
                past_key_values=cache,
                use_cache=True,
                num_logits_to_keep=1,
            )

        # ======================================
        # 4. Fisher probe
        # ======================================

        logits = outputs.logits[:, -1, :].float()
        log_probs = logits.log_softmax(dim=-1)
        probs = log_probs.detach().exp()

        y = torch.multinomial(
            probs,
            1,
        )

        probe = log_probs.gather(
            -1,
            y,
        ).mean()

        # ======================================
        # 5. Trigger gradient hooks
        # ======================================

        probe.backward()

    # ==========================================
    # 6. Compress full T-token cache
    # ==========================================

    press.compress_cache_from_scores(
        model=model,
        cache=cache,
    )

    press.clear()
```

---

# 20. 第二层优化：Activation Checkpointing

如果 W=32 的 window backward 仍然 OOM，再考虑 activation checkpointing。

Checkpointing 本质是：

```text
forward 少保存 activation
        ↓
backward 时重新计算
```

即：

\[
\text{memory}\downarrow,
\qquad
\text{latency}\uparrow.
\]

因此不应作为第一方案。

---

# 21. Checkpointing 优先 MLP，不要先 checkpoint attention

推荐优先：

```text
MLP / FFN checkpointing
```

原因是 attention forward 可能修改 KV cache。

如果 checkpoint backward 重新执行 attention forward，可能造成：

```text
cache 被重复 append / mutation
```

因此第一版不要直接对整个 Transformer block 做 checkpoint。

---

# 22. 第三层优化：Saved Tensor CPU Offload

如果目标是“绝对不能 OOM”，而 latency 可以接受，可使用 PyTorch 的 saved tensor CPU offload，例如：

```python
with torch.autograd.graph.save_on_cpu(
    pin_memory=True
):
    outputs = model(...)
```

优点：

```text
GPU activation memory 显著下降
```

缺点：

```text
GPU ↔ CPU transfer
backward latency 增加
```

因此它更适合：

```text
correctness baseline
emergency fallback
```

而不是最终在线方案。

---

# 23. 第四层优化：缩小 Fisher Window

测试：

```text
W = 1
W = 4
W = 8
W = 16
W = 32
```

当前：

\[
R_i
=
\sqrt{
\frac1W
\sum_t
(g_t^\top u_i)^2
}.
\]

当 \(W\) 越小，显存与 latency 越低。

最极端：

\[
W=1,
\]

则：

\[
R_i=|g_l^\top u_i|,
\]

最终：

\[
S_i=A_i|g_l^\top u_i|.
\]

这是非常值得做的效率 ablation。

---

# 24. 第五层优化：Fisher Sample 数保持为 1

Online V1 先固定：

```text
1 Fisher sample
```

即：

\[
y\sim p.
\]

不要一开始就使用 4/8 个 Fisher samples，否则需要额外 backward 或更复杂的 Jacobian/VJP 计算。

可以离线 ablate：

```text
1 sample
2 samples
4 samples
```

但 main online implementation 先用 1 sample。

---

# 25. 第六层优化：只对部分层使用 LogitKV

可以测试：

```text
all layers
last 16 layers
last 8 layers
```

例如：

```text
layers 0-15:
CriticalKV

layers 16-31:
LogitKV
```

这会进一步降低 backward/hook/score 开销，但属于方法近似，需要实验验证。

---

# 26. 最终部署方向：Offline Low-Rank Fisher

如果最终希望：

```text
online zero backward
```

则可将 input-specific：

\[
G_l(x)
\]

近似成 calibration distribution 下的：

\[
\bar G_l
=
\mathbb E_x[G_l(x)].
\]

对其低秩分解：

\[
\bar G_l
\approx
B_lB_l^\top.
\]

则：

\[
\sqrt{u_i^\top\bar G_lu_i}
=
\|B_l^\top u_i\|_2.
\]

因为：

\[
u_i=V_iW_O,
\]

所以：

\[
\boxed{
S_i
=
A_i
\|B_l^\top V_iW_O\|_2
}.
\]

---

# 27. 进一步预计算：把 Fisher geometry 融合进 output projection

令：

\[
P_l=W_OB_l.
\]

则在线 score 变成：

\[
\boxed{
S_i
=
A_i
\|V_iP_l\|_2
}.
\]

代码：

```python
projected = torch.matmul(
    values,
    P_l,
)

fisher_norm = projected.norm(
    p=2,
    dim=-1,
)

score = attention_score * fisher_norm
```

这样完全不需要：

```text
autograd
online backward
activation graph
online Fisher estimation
```

---

# 28. Offline LogitKV 与 CriticalKV 的关系

CriticalKV：

\[
S_i=A_i\|V_iW_O\|_1.
\]

Offline LogitKV：

\[
S_i=A_i\|V_iP_l\|_2,
\qquad
P_l=W_OB_l.
\]

代码结构几乎重新回到 CriticalKV 的范式，但 \(P_l\) 已经编码了：

```text
W_O
+
downstream Fisher geometry
```

---

# 29. 建议最终形成两个 LogitKV 版本

## 29.1 LogitKV-Online

\[
S_i
=
A_i
\sqrt{u_i^\top\hat G_l(x)u_i
}
.
\]

实现：

```text
prefix no-grad
+
window backward
```

特点：

```text
input-adaptive
training-free
one lightweight Fisher backward
```

---

## 29.2 LogitKV-Calib / LogitKV-LR

\[
S_i
=
A_i
\sqrt{u_i^\top\bar G_lu_i
}
.
\]

低秩近似：

\[
\bar G_l\approx B_lB_l^\top.
\]

最终：

\[
S_i=A_i\|V_iP_l\|_2.
\]

特点：

```text
zero online backward
low memory
low latency
easy deployment
```

---

# 30. 推荐优先级

| 方法 | 显存 | 延迟 | 是否改变当前 LogitKV 定义 | 推荐程度 |
|---|---:|---:|---|---|
| Full-context backward | 极高 | 极高 | 否 | 不推荐 |
| Activation checkpointing | 中 | 更高 | 否 | 备用 |
| CPU offload | 低 | 很高 | 否 | correctness fallback |
| **Prefix no-grad + Window backward** | **低** | **低** | **基本不变** | **主推** |
| **Window backward + grad hooks** | **更低** | **低** | 不变 | **主推** |
| Smaller Fisher window | 更低 | 更低 | 轻微近似 | 推荐 ablation |
| Partial-layer LogitKV | 更低 | 更低 | 方法近似 | 可选 |
| **Offline low-rank Fisher** | **极低** | **极低** | dataset-level approximation | **最终部署版** |

---

# 31. 推荐开发顺序

## Commit 1：Split Forward

实现：

```text
prefix no-grad
+
suffix grad
```

先不计算 LogitKV。

验证：

```text
split forward logits
≈
full forward logits
```

---

## Commit 2：Gradient Correctness Test

在小 context 上比较：

```text
full-graph backward gradient
vs
prefix-detached window gradient
```

例如：

```text
T = 128
W = 16
```

逐层检查：

```python
torch.testing.assert_close(
    grad_full[:, -W:],
    grad_window,
    rtol=1e-4,
    atol=1e-5,
)
```

这是整个优化方案最重要的 correctness test。

---

## Commit 3：Window-aware SnapKV

支持：

```text
W query
对
T full keys
```

得到完整 base attention score。

---

## Commit 4：Gradient Hooks

每层 gradient 到达时立即计算：

```text
Fisher-RMS
LogitKV score
```

---

## Commit 5：KV Compression

加入：

```text
CriticalKV-style Stage-1 safeguard
Top-K
cache gather
detach
```

然后正常 decode。

---

## Commit 6：Memory / Latency Profiling

比较：

```text
Full backward
vs
Window backward
```

至少记录：

```text
peak GPU memory
prefix prefill latency
suffix forward latency
backward latency
score latency
compression latency
total pre-decode latency
```

---

## Commit 7：Window Ablation

测试：

```text
W = 1 / 4 / 8 / 16 / 32
```

观察：

```text
accuracy
peak memory
latency
score stability
```

---

## Commit 8：Offline Low-Rank Fisher

只有当 Online LogitKV 明显优于 CriticalKV 时再做。

目标：

```text
zero online backward
```

---

# 32. 必做 Correctness Tests

- [ ] Split forward logits ≈ full forward logits
- [ ] Prefix detached 不改变 suffix forward representation
- [ ] Window gradient ≈ full-graph gradient on same suffix positions
- [ ] Fisher-RMS fast implementation ≈ explicit `VW_O`
- [ ] Root score ranking == squared score ranking
- [ ] Prefix + suffix forward 后 cache length == T
- [ ] Compression ratio 正确
- [ ] Fisher score 非负且 finite
- [ ] 固定 probe seed 后 selected indices 可复现
- [ ] Model parameters 不产生 `.grad`
- [ ] Decode 前 cache 已 `.detach()`

---

# 33. 必做效率实验

建议记录：

| Variant | Peak Memory | Prefix | Suffix Fwd | Backward | Score | Total |
|---|---:|---:|---:|---:|---:|---:|
| Full backward |  |  |  |  |  |  |
| Window W=32 |  |  |  |  |  |  |
| Window W=16 |  |  |  |  |  |  |
| Window W=8 |  |  |  |  |  |  |
| Window W=1 |  |  |  |  |  |  |
| Offline LR |  |  |  |  |  |  |

建议 context length：

```text
2K
4K
8K
16K
```

因为这项优化一个非常重要的特点就是：

> Context 越长，固定 window backward 相对于 full backward 的优势越明显。

---

# 34. 第一版 Online LogitKV 推荐配置

```text
base scorer:
SnapKV

context:
4K

fisher_window:
32

fisher samples:
1

gradient:
window only

prefix:
no-grad

parameters:
frozen

score:
A_i * Fisher-RMS

compression:
CriticalKV-style Stage-1 safeguard

cache:
DynamicCache

batch size:
1
```

---

# 35. 最终建议

不要继续实现：

```text
Full T-token autograd forward
        ↓
Full T-token backward
```

Online LogitKV 应优先改成：

\[
\boxed{
\text{Prefix No-Grad}
\rightarrow
\text{Window Grad Forward}
\rightarrow
\text{One Fisher VJP}
\rightarrow
\text{LogitKV Score}
\rightarrow
\text{KV Eviction}
}
\]

对于：

\[
T=4096,
\qquad W=32,
\]

复杂度层面：

- MLP backward token 数约为 full backward 的 **0.78%**；
- attention backward 主要 query-key interaction 约为 full causal backward 的 **1.56%**。

如果最终希望完全消除 online backward，则进一步采用：

\[
\boxed{
G_l\approx B_lB_l^\top
}
\]

并预计算：

\[
P_l=W_OB_l,
\]

使在线 score 变成：

\[
\boxed{
S_i=A_i\|V_iP_l\|_2
}
\]

从而得到一个真正接近 CriticalKV 在线成本、同时保留 LogitKV downstream Fisher geometry 的高效版本。
