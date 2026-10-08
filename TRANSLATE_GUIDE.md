# `translate.py` 学习指南：从 PyTorch 基础到 Transformer 翻译

这份笔记结合程序真实执行顺序讲解 `translate.py`。阅读时不要求提前学过
PyTorch，但最好知道 Python 的函数、类、列表、字典和切片。

全文分成三层：

1. 第 1～2 节建立 Tensor、参数、梯度和反向传播的数学基础；
2. 第 3～5 节解释数据怎样进入 Transformer，以及模型为什么能并行处理序列；
3. 第 6～14 节对应完整训练、生成、优化和模型保存流程。

其中 2.6～2.7 是数学细节较多的深入章节。第一次阅读如果暂时觉得公式密集，可以
先读到 2.5，然后阅读第 3～6 节建立实际流程，再回来细看 2.6～2.7。

## 1. 先建立整体认识

训练翻译模型，本质上是在反复做下面这件事：

```text
一个 batch 的多对中英文句子
  ↓ SentencePiece 分词并补齐
src [B,S]，tgt [B,T]
  ↓ Encoder + Decoder；训练时提供真实目标前缀
一次并行得到所有句子、所有目标位置的 logits [B,T-1,V]
  ↓ 每个有效位置与正确 token 比较
所有 token loss 取平均，得到一个 batch loss
  ↓ backward 汇总全部有效位置的意见
更新一次模型参数，然后处理下一个 batch
```

推理时没有正确英文答案，不能像训练时那样提前提供整句真实目标，因此流程变成：

```text
BOS → 预测 I → 预测 love → 预测 you → 预测 EOS
```

模型每次只预测“下一个 token”，然后把自己的预测接回输入，继续预测。这叫
**自回归生成**。

因此先记住全文最重要的一组区别：

```text
训练：知道正确目标句 → 所有目标位置并行预测 → 一个 batch 更新一次参数
推理：不知道正确目标句 → 每次生成一个 token → 反复前向，但不反向、不更新参数
```

## 2. PyTorch 中最重要的几个概念

### 2.1 Tensor：可以放在 GPU 上计算的多维数组

`torch.Tensor` 类似 NumPy 数组，但多了两个神经网络需要的能力：

1. 可以放到 GPU 上；
2. 可以记录运算过程并自动求梯度。

例如：

```python
x = torch.tensor([[2, 15, 39, 3]])
```

`x.shape` 是 `[1, 4]`：

- 第 0 维大小为 1，表示 batch 中有 1 句话；
- 第 1 维大小为 4，表示这句话有 4 个 token。

代码中常见方法：

- `tensor.size(1)`：读取第 1 维长度；
- `tensor.to(device)`：把张量移到 CPU 或 GPU；
- `tensor.eq(PAD)`：逐元素判断是否等于 PAD，得到布尔张量；
- `tensor.reshape(...)`：重新组织形状，不改变元素总数；
- `tensor.tolist()`：转换成普通 Python 列表；
- `tensor.item()`：从只有一个元素的张量中取出 Python 数值。

### 2.2 shape：理解模型代码的核心

本项目使用以下字母：

| 字母 | 含义 | 示例 |
|---|---|---:|
| `B` | batch size，一批有多少样本 | 64 |
| `L` | 通用的 sequence length，一条序列补齐后的 token 数；具体到源序列和目标序列时分别写作 `S` 和 `T` | 42 |
| `S` | 源句补齐后的长度 | 35 |
| `T` | 目标句补齐后的长度 | 42 |
| `D` | 每个 token 的向量维度 | 256（当前默认值） |
| `V` | 词表大小 | 8000 |

一次训练中，主要张量的形状是：

```text
src             [B, S]
tgt             [B, T]
src embedding   [B, S, D]
memory          [B, S, D]
decoder hidden  [B, T-1, D]
logits          [B, T-1, V]
labels          [B, T-1]
```

遇到 PyTorch 报 shape 错误时，应先打印这些形状，而不是先猜模型算法有问题。

### 2.3 `nn.Module`：模型和神经网络层的统一基类

`Translator(nn.Module)` 表示 `Translator` 是一个 PyTorch 模型。自定义模型通常
需要两部分：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 3)

    def forward(self, x):
        return self.layer(x)
```

- `__init__`：创建并保存神经网络层；
- `forward`：规定输入怎样流过这些层；
- `model(x)`：PyTorch 会间接调用 `model.forward(x)`，通常不要手动调用
  `forward`；
- `model.parameters()`：取得所有需要训练的参数；
- `model.to(device)`：移动所有参数和 buffer；
- `model.state_dict()`：取得“参数名称 → 参数张量”的字典。

### 2.4 Parameter 与 buffer

Embedding、Linear 和 Transformer 内部的权重都是 `Parameter`：

- 会计算梯度；
- 会被 `optimizer` 更新；
- 会进入 `state_dict`。

位置编码 `pe` 使用 `register_buffer`：

- 不计算、也不更新梯度；
- 会跟随模型移动到 GPU；
- 会进入 `state_dict`。

普通 Python 属性既不会被优化器更新，也不会自动随模型移动设备。

### 2.5 自动求导、计算图和 `backward`

只要梯度记录处于开启状态，并且运算涉及需要梯度的参数，PyTorch 就会记录张量
经过的运算，形成计算图：

```text
参数 → logits → cross entropy → loss
```

这里要区分两个开关：

- `model.train()` 决定 Dropout 等层采用训练行为；
- `torch.set_grad_enabled(...)`、`torch.no_grad()` 决定是否记录计算图。

因此 `model.eval()` 本身并不会禁止求梯度。本项目在 `run_epoch()` 中同时用
`model.train(training)` 和 `torch.set_grad_enabled(training)`，训练时打开二者，
验证时关闭 Dropout 和梯度记录。

调用：

```python
loss.backward()
```

PyTorch 会沿图反向使用链式法则，计算每个参数的梯度，并存放在
`parameter.grad` 中。`backward` 只负责算梯度，不会自动修改参数；真正更新参数
的是：

```python
optimizer.step()
```

可以把 `.grad` 想成每个参数旁边的一张便签，上面写着：“当前 batch 的 loss
对我有多敏感”。例如某个参数的梯度是 `+0.2`，表示在当前位置附近，稍微增大
这个参数会使 loss 增大；为了减小 loss，更新时通常应让参数往负方向移动。

完整顺序通常是：

```python
optimizer.zero_grad()
logits = model(inputs)
loss = loss_function(logits, labels)
loss.backward()
optimizer.step()
```

这四步不能互相替代：前向传播负责得到预测，loss 负责给预测打分，`backward`
负责计算梯度，`step` 才负责修改参数。

### 2.6 导数、偏导数和梯度到底是什么关系

先说结论：**一维时只有一个导数；多维时每个参数都有一个偏导数，把这些偏导数
按顺序放在一起，就叫梯度。** 所以口语里经常说“求导数就是求梯度”，但严格来
说，导数是一个更宽泛的概念，梯度特指“标量函数对一组变量的所有偏导数组成的
向量”。

#### 一维：导数表示曲线此处的斜率

假设模型只有一个参数 `w`：

```text
L(w) = (w - 3)²
```

导数为：

```text
dL/dw = 2(w - 3)
```

导数不是凭空多出来的另一个量。它回答的是：**在当前位置，把 `w` 增加一点点，
`L` 大约会变化多少？**

如果把 `w` 改变一个很小的量 `Δw`，有近似关系：

```text
ΔL ≈ (dL/dw) × Δw
```

例如当前 `w=5`，导数为 `+4`。若让 `w` 增加 `0.01`，那么：

```text
ΔL ≈ 4 × 0.01 = 0.04
```

也就是 loss 大约增加 `0.04`。正导数表示向右走会升高，负导数表示向右走会
降低，导数绝对值越大表示当前位置越陡。

为了降低 loss，要沿导数的反方向更新：

```text
w_new = w_old - learning_rate × dL/dw
      = 5 - 0.1 × 4
      = 4.6
```

若当前 `w=1`，导数是 `-4`：

```text
w_new = 1 - 0.1 × (-4) = 1.4
```

减去负数相当于增大 `w`，所以两种情况都会向最低点 `w=3` 靠近。

#### 多维：每个参数都有自己的偏导数

真实模型不只有一个参数。先看只有两个参数的损失函数：

```text
L(w₁, w₂) = (w₁ - 3)² + (w₂ + 1)²
```

现在不能只问“`L` 的斜率是多少”，因为可以只改变 `w₁`，也可以只改变 `w₂`。
于是分别计算：

```text
∂L/∂w₁ = 2(w₁ - 3)    # 只改变 w₁ 时，loss 怎么变
∂L/∂w₂ = 2(w₂ + 1)    # 只改变 w₂ 时，loss 怎么变
```

这种“固定其他变量，只看一个变量”的导数叫**偏导数**。把所有偏导数排成一个
向量，就是梯度：

```text
∇L = [∂L/∂w₁, ∂L/∂w₂]
```

例如在 `(w₁, w₂)=(5, 2)` 处：

```text
∇L = [4, 6]
```

它表示在当前位置，`w₁` 增加一点会以约 `4` 的速率推高 loss，`w₂` 增加一点会
以约 `6` 的速率推高 loss。若两个参数同时发生小变化，loss 的变化近似为：

```text
ΔL ≈ ∂L/∂w₁ × Δw₁ + ∂L/∂w₂ × Δw₂
   = ∇L · Δw
```

这就是梯度最核心的含义：**它把每个参数对 loss 的局部影响收集起来，因此能
预测参数朝某个方向小幅移动后，loss 会怎样变化。**

在所有长度相同的移动方向中，沿梯度 `∇L` 移动会让 loss 上升最快，沿负梯度
`-∇L` 移动会让 loss 下降最快。因此梯度下降写成：

```text
参数_new = 参数_old - learning_rate × 梯度
```

#### 为什么 `[4, 6]` 要一起移动，而不是只移动影响更大的 `6`

在 `(w₁, w₂)=(5, 2)` 处，梯度为：

```text
∇L = [4, 6]
```

`6` 的绝对值确实比 `4` 大，表示在当前位置，`w₂` 的单位变化对 loss 的影响
更大。但 `w₁` 的影响并不是零：减小 `w₁` 同样可以降低 loss。因此梯度下降会
同时利用两个可以降低 loss 的方向，并通过移动幅度体现影响大小。

假设学习率 `η=0.1`：

```text
Δw = -η∇L
   = -0.1 × [4, 6]
   = [-0.4, -0.6]
```

所以参数更新为：

```text
w₁: 5 → 4.6    # 移动 0.4
w₂: 2 → 1.4    # 移动 0.6
```

`w₂` 已经因为梯度绝对值较大而移动得更多。若只移动 `w₂`，相当于虽然已经知道
减小 `w₁` 也能降低 loss，却暂时不使用这部分信息。

还可以用 `ΔL ≈ ∇L·Δw` 比较不同移动方向。假设参数总共只能移动长度 `0.1`。
如果只减小 `w₂`：

```text
Δw = [0, -0.1]

ΔL ≈ [4, 6] · [0, -0.1]
   = -0.6
```

如果沿负梯度方向移动，先把负梯度转换成长度为 1 的方向：

```text
梯度长度 = √(4² + 6²) = √52 ≈ 7.21

负梯度单位方向
= [-4/7.21, -6/7.21]
≈ [-0.555, -0.832]
```

沿这个方向移动长度 `0.1`：

```text
Δw ≈ [-0.0555, -0.0832]

ΔL ≈ [4, 6] · [-0.0555, -0.0832]
   ≈ -0.721
```

在移动总长度相同的条件下：

```text
只移动 w₂：    loss 约下降 0.600
沿负梯度移动： loss 约下降 0.721
```

因此，负梯度不是要求所有参数移动相同距离，而是让各参数按照各自梯度的大小
成比例移动。它综合利用所有能降低 loss 的分量，是当前位置附近下降最快的方向。

每次只选择一个参数更新也是可行的，这类方法称为**坐标下降**。但神经网络通过
一次反向传播已经得到了所有参数的梯度，同时更新通常可以更充分地利用这些信息。

#### 如果梯度中一个为正、一个为负，该怎样移动

假设某处的梯度是：

```text
∇L = [4, -6]
```

两个分量分别表示：

```text
∂L/∂w₁ = 4
```

`w₁` 增大会使 loss 增大，因此为了降低 loss，应该减小 `w₁`。

```text
∂L/∂w₂ = -6
```

`w₂` 增大会使 loss 减小，因此为了降低 loss，应该增大 `w₂`。

不需要为正负梯度设计两套更新规则，统一使用“减去梯度”即可。假设当前参数为
`[5, 2]`，学习率为 `0.1`：

```text
w_new
= [5, 2] - 0.1 × [4, -6]
= [5, 2] - [0.4, -0.6]
= [4.6, 2.6]
```

也就是：

| 梯度符号 | 参数增大时 loss 怎样变化 | 为降低 loss 的更新方向 |
|---:|---|---|
| 正 | 增大 | 减小参数 |
| 负 | 减小 | 增大参数 |
| 零 | 局部几乎不变 | 暂时不移动 |

回到当前的损失函数：

```text
L(w₁, w₂) = (w₁ - 3)² + (w₂ + 1)²
```

它的最低点是 `(3, -1)`。如果当前位置改为 `(5, -4)`：

```text
∂L/∂w₁ = 2(5 - 3)  = 4
∂L/∂w₂ = 2(-4 + 1) = -6

∇L = [4, -6]
```

此时 `w₁=5` 比最优值 `3` 大，所以应该减小；`w₂=-4` 比最优值 `-1` 小，
所以应该增大。负梯度：

```text
-∇L = [-4, 6]
```

正好同时指向 `w₁` 减小、`w₂` 增大的方向。因此可以这样记忆梯度的每个分量：

> 绝对值表示这个参数当前对 loss 的局部影响有多强；正负号表示为了降低 loss，
> 这个参数应该向哪个方向移动。梯度下降会同时利用所有参数提供的下降机会。

Transformer 有数百万甚至更多参数，但没有出现新的数学概念。梯度只是更长：

```text
∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]
```

PyTorch 不会真的创建一个超长 Python 列表，而是把每个参数对应的偏导数保存在
该参数的 `.grad` 张量中；它的 shape 通常和参数本身完全相同。

### 2.7 loss 到底怎样算，又怎样影响原来的模型参数

本节先用一个简化输出层解释数学链路。这里会提前出现 `hidden`、`W`、`logits` 等
名称；暂时把它们理解成“上一层给出的向量”“输出层权重”和“词表分数”即可，
第 4 节会再放回完整 Transformer，第 6～7 节会放回真实训练循环。

这一节先纠正一个很容易出现、也非常关键的表述：

> 不是“把 loss 的参数转换成输出的参数”，也不是“把 loss 填回模型参数”。
> loss 通常没有可训练参数。真正计算的是 **loss 对输出以及对每个模型参数的
> 偏导数**，优化器再根据这些偏导数修改原来的模型参数。

这里有四类容易混淆的东西：

| 名称 | 本项目中的例子 | 是模型参数吗 | 会被优化器直接更新吗 |
|---|---|---:|---:|
| 模型参数 | 输出层的 `W`、`b`，Attention 权重，Embedding 表 | 是 | 是 |
| 中间结果 | hidden、logits、softmax 概率 | 否 | 否 |
| loss | 由所有有效 token 的交叉熵平均得到的标量 Tensor | 否 | 否 |
| 梯度 | `W.grad`、`b.grad` 等，即 `∂loss/∂参数` | 不是参数，是参数旁边的暂存结果 | 不被更新，而是被优化器读取 |

所以要严格区分：

```text
logits 是模型的输出数值，不是“输出参数”；
W、b 才是产生 logits 的输出层参数；
loss 是评价本次 logits 的一个数值，不是模型参数；
gradient 是 loss 对某个量的变化率，也不是模型参数。
```

#### 本项目的 loss 是怎样一步一步算出来的

代码是：

```python
logits = model(src, decoder_input)
loss = nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    labels.reshape(-1),
    ignore_index=PAD,
)
```

先只看一句话中的一个预测位置。假设词表只有三个 token，模型输出的 logits 是：

```text
z = [-0.1, 1.0, -1.1]
      I    love   you
```

logit 只是任意实数分数，还不是概率。softmax 先计算：

```text
exp(z) = [exp(-0.1), exp(1.0), exp(-1.1)]
       ≈ [0.905, 2.718, 0.333]

总和 ≈ 3.956

p = softmax(z)
  ≈ [0.229, 0.687, 0.084]
```

假设正确 token 是 `you`，标签 id 指向第三项。交叉熵只取正确项的概率并计算
负对数：

```text
L = -log(p_you)
  = -log(0.084)
  ≈ 2.477
```

这个 `2.477` 就是这个位置的 loss。它不是模型输出的某一项，也不是一个准备塞进
模型的参数；它只是根据模型输出和正确答案计算出的“这次预测有多差”的分数。

交叉熵也可直接写成 logits 的函数。若正确类别为 `y`：

```text
L = -log(exp(z_y) / Σ_j exp(z_j))
  = -z_y + log(Σ_j exp(z_j))
```

PyTorch 的 `cross_entropy` 在内部等价于稳定版本的 `log_softmax + NLLLoss`，
因此训练代码不应先手动调用 softmax。

实际 batch 有许多句子和位置。设所有标签不是 PAD 的位置集合为 `M`，有效位置数
为 `N`，本项目使用默认的 `reduction="mean"`，所以最终标量为：

```text
batch_loss = (1/N) × Σ_(k∈M) [-log p(k, 正确 token)]
```

标签为 PAD 的位置被 `ignore_index=PAD` 排除，既不进入求和，也不进入平均分母。
于是 `[B, T-1, V]` 的大量预测最终汇总成一个 shape 为 `[]` 的标量 Tensor。

#### loss 不等于输出，为什么还能求“loss 对输出的梯度”

“两个量不相等”和“一个量由另一个量计算出来”完全是两回事。例如：

```text
y = x²
```

`y` 不等于 `x`，但 `y` 是 `x` 的函数，所以仍然可以计算 `dy/dx = 2x`。同理：

```text
模型参数 θ
   ↓ 模型前向计算
logits z
   ↓ softmax + 交叉熵，并使用正确标签 y
loss L
```

loss 不需要等于 logits。只要 `L` 是由 logits 经过可求导运算算出来的，就可以求：

```text
∂L/∂z
```

先解释这里的符号。假设词表有 `V` 个 token，那么某一个预测位置的完整输出是一个
向量：

```text
z = [z₁, z₂, ..., zᵥ]
```

- `z` 表示完整的 logits 向量；
- `z_i` 表示这个向量的第 `i` 项，也就是模型给第 `i` 个候选 token 的分数；
- 下标 `i` 只是“第几个候选”的编号，不是求导符号；
- `∂L/∂z_i` 才是完整的偏导符号，读作“loss 对第 `i` 个 logit 的偏导数”。

例如：

```text
候选 token：      I      love     you
编号 i：          1       2        3
logits z：      [-0.1,    1.0,    -1.1]

z₁ = -0.1   # I 的 logit
z₂ =  1.0   # love 的 logit
z₃ = -1.1   # you 的 logit
```

所以这里并没有一个单独叫作“`∂z_i`”的量。分母中的 `∂z_i` 是偏导记号的一部分：

```text
      ∂L
     ────
      ∂z_i
```

它的意思不是“把 loss 转换成 logits”，而是：

> 保持其他 logits 不变，只把第 `i` 个 logit `z_i` 增加非常小的一点，loss 会
> 怎样变化？

更具体地说，如果 `∂L/∂z_i = 0.687`，那么把 `z_i` 增加一个很小的量
`Δz_i=0.001`，loss 大约变化：

```text
ΔL ≈ (∂L/∂z_i) × Δz_i
   = 0.687 × 0.001
   = 0.000687
```

#### `p_i - 1(i=y)` 是怎样推导出来的

假设正确 token 的编号是 `y`。softmax 给第 `i` 个 token 的概率是：

```text
               exp(z_i)
p_i = ─────────────────────────
       exp(z₁)+exp(z₂)+...+exp(zᵥ)
```

交叉熵只关心正确 token 的概率 `p_y`：

```text
L = -log(p_y)
```

把 softmax 的 `p_y` 代进去：

```text
               exp(z_y)
L = -log(─────────────────────────)
         Σ_j exp(z_j)
```

使用对数规则 `log(a/b)=log(a)-log(b)`，以及 `log(exp(z_y))=z_y`：

```text
L = -[log(exp(z_y)) - log(Σ_j exp(z_j))]
  = -z_y + log(Σ_j exp(z_j))
```

这个形式很适合求导，因为它明确分成两项：

```text
L = -z_y                  + log(Σ_j exp(z_j))
    └─只直接包含正确项─┘    └─包含所有 logits─┘
```

现在对任意第 `i` 个 logit `z_i` 求偏导，需要分别计算这两项。

第一项 `-z_y` 的导数取决于 `i` 是否等于正确答案 `y`：

```text
∂(-z_y)/∂z_i = -1    当 i=y，即正在对正确 token 的 logit 求导
∂(-z_y)/∂z_i =  0    当 i≠y，即正在对错误 token 的 logit 求导
```

用指示函数可以合并写成：

```text
∂(-z_y)/∂z_i = -1(i=y)
```

这里的 `1(i=y)` 不是数字 1 乘某个式子，而是一个判断函数：

```text
1(i=y) = 1   当 i=y
1(i=y) = 0   当 i≠y
```

再看第二项。为了让式子更清楚，先令：

```text
S = Σ_j exp(z_j)
```

于是第二项是 `log(S)`。使用链式法则：

```text
∂log(S)/∂z_i
= (∂log(S)/∂S) × (∂S/∂z_i)
= (1/S) × exp(z_i)
= exp(z_i) / Σ_j exp(z_j)
= p_i
```

为什么 `∂S/∂z_i = exp(z_i)`？因为：

```text
S = exp(z₁) + ... + exp(z_i) + ... + exp(zᵥ)
```

对 `z_i` 求偏导时，其他 `z_j` 都暂时看作常数，只有 `exp(z_i)` 会变化，而
`exp(z_i)` 的导数仍是 `exp(z_i)`。

最后把两项的导数相加：

```text
∂L/∂z_i
= ∂[-z_y + log(Σ_j exp(z_j))] / ∂z_i
= -1(i=y) + p_i
= p_i - 1(i=y)
```

这就是那个“简洁结果”的来源，并不是突然规定出来的。拆成两种情况会更直观：

```text
如果 i 是正确 token： ∂L/∂z_i = p_i - 1
如果 i 是错误 token： ∂L/∂z_i = p_i
```

因为概率总在 `0` 到 `1` 之间，所以：

- 正确 token：`p_i-1 ≤ 0`，梯度通常为负，梯度下降会倾向于提高它的 logit；
- 错误 token：`p_i ≥ 0`，梯度通常为正，梯度下降会倾向于降低它的 logit。

继续使用前面的数值例子。正确答案是 `you`，即 `y=3`：

```text
正确答案 one-hot = [0, 0, 1]

∂L/∂z
= [0.229, 0.687, 0.084] - [0, 0, 1]
= [0.229, 0.687, -0.916]
```

解释每一项：

- `I` 的梯度为正：提高它的分数会提高 loss，应该压低它；
- `love` 的梯度为较大的正数：模型错误地很相信它，更应该压低它；
- `you` 的梯度为负：提高正确答案的分数会降低 loss，应该提高它。

这里的 `∂L/∂z` 是把所有 `∂L/∂z_i` 排在一起形成的梯度向量：

```text
∂L/∂z = [∂L/∂z₁, ∂L/∂z₂, ∂L/∂z₃]
```

对单个 token 的 loss，上面的公式就是 `p-one_hot`。如果最终 loss 是 `N` 个有效
位置的平均值，那么每个位置传出的梯度还会乘以 `1/N`；PyTorch 会在
`cross_entropy(reduction="mean")` 的反向过程中自动完成这个缩放。

优化器最终做“参数减去梯度方向上的更新”，所以正负号会产生正确的调整方向。
注意优化器不会直接更新 logits：logits 是这一次前向传播的临时结果，下一个 batch
会重新计算。优化器更新的是产生 logits 的 `W`、`b` 以及更前面的参数。

#### `∂loss/∂logits` 怎样继续变成 `∂loss/∂W`

假设最后一层是：

```text
z = Wh + b
```

这里：

- `h` 是 Decoder 输出的隐藏向量；
- `W`、`b` 是模型真正保存的输出层参数；
- `z` 是本次前向临时算出的 logits。

已知上游传来的 `g = ∂L/∂z` 后，根据链式法则：

```text
∂L/∂W = g hᵀ
∂L/∂b = g
∂L/∂h = Wᵀg
```

这三项分别负责不同事情：

```text
∂L/∂W、∂L/∂b  → 保存到输出层参数的 .grad，之后由优化器使用
∂L/∂h          → 继续传给 Decoder，计算 Decoder 更前面参数的梯度
```

因此并不存在一个神秘步骤，把“loss 的参数”转换为“输出参数”。完整说法应该是：

```text
先由交叉熵算出 loss 对 logits 的梯度；
再利用 logits = Wh+b 的局部导数，算出 loss 对 W、b、h 的梯度；
对 h 的梯度继续向 Decoder、Encoder 和 Embedding 反传；
最后每个可训练参数都得到一个与自身 shape 相同的 .grad。
```

#### loss、计算图和梯度在内存里分别存在哪里

执行前向传播后，Python 变量 `loss` 指向一个标量 Tensor。概念上可以观察到：

```python
print(loss.shape)          # torch.Size([])，零维标量 Tensor
print(loss.requires_grad)  # 训练时通常为 True
print(loss.grad_fn)        # 指向产生它的反向运算节点
```

这个 Tensor 大致包含两类信息：

1. 当前 loss 的数值，例如 `2.477`；
2. 通往前面运算的自动求导关系，即它是怎样由 logits 和模型参数算出来的。

第二部分常被笼统地称为**计算图**。图的节点代表诸如 Linear、加法、矩阵乘法、
softmax/交叉熵等运算；边表示 Tensor 之间的依赖关系。为了在反向时计算局部导数，
某些运算还会暂存前向所需的信息，例如 Linear 的输入 `h`。这些都是一次前向传播
期间的临时数据，不会成为 `state_dict` 里的长期模型参数。

调用：

```python
loss.backward()
```

之后，真正需要长期更新的叶子参数会得到 `.grad`：

```python
model.output.weight       # 参数 W，本身的数值
model.output.weight.grad  # ∂loss/∂W，与 W shape 相同

model.output.bias         # 参数 b
model.output.bias.grad    # ∂loss/∂b，与 b shape 相同
```

默认情况下，`loss.grad` 往往是 `None`，logits、hidden 这类非叶子中间 Tensor 的
`.grad` 也默认不保留。并不是没有计算过 `∂L/∂logits`、`∂L/∂h`，而是自动求导在
反传过程中计算并使用它们后，通常不把它们长期存进 `.grad`。如果为了教学或调试
确实想查看，可以在 `backward()` 前对相应 Tensor 调用 `retain_grad()`。

标签 `labels` 是整数 token id，不需要梯度。交叉熵把它当作“正确答案索引”，
而不是需要优化的连续变量。模型参数需要梯度，因为训练的目标正是修改这些参数。

一次训练 step 中，各类数据的生命周期可以概括为：

```text
长期存在：模型参数 θ
    ↓ 前向
临时产生：hidden、logits、loss、计算图及部分保存的中间量
    ↓ backward
参数旁产生或累加：θ.grad
    ↓ optimizer.step
长期参数 θ 被更新
    ↓ 下一批之前 zero_grad
旧的 θ.grad 被清空；旧计算图通常也在不再被引用后释放
```

`loss.item()` 只会取出一个普通 Python 数字用于打印或累计日志。这个数字已经脱离
计算图，不能再拿来调用 `backward()`。真正反向传播使用的是原来的 `loss` Tensor。

#### 参数究竟怎样被改动：一个完整的数值例子

仍用上面的输出层例子，假设隐藏向量为：

```text
h = [2, -1]
```

交叉熵已经给出：

```text
g = ∂L/∂z = [0.229, 0.687, -0.916]
```

那么 `love` 那一行权重的梯度是：

```text
∂L/∂W_love = 0.687 × [2, -1]
            = [1.374, -0.687]
```

假设它原来是：

```text
W_love = [0.4, -0.2]
```

若为了说明原理，先使用最简单的 SGD，学习率 `η=0.1`：

```text
W_new = W_old - η × ∂L/∂W
      = [0.4, -0.2] - 0.1 × [1.374, -0.687]
      = [0.2626, -0.1313]
```

更新后，再把同一个 `h` 送入该行，它给 `love` 的分数会从：

```text
更新前：0.4×2 + (-0.2)×(-1)       = 1.0000
更新后：0.2626×2 + (-0.1313)×(-1) = 0.6565
```

降到约 `0.6565`。这正符合交叉熵给出的意见：错误 token `love` 的 logit 太高，
应该降低。与此同时，正确 token `you` 那一行的梯度为负，按“减去梯度”更新后，
通常会提高 `you` 的 logit。

本项目实际使用的是 `AdamW`，不是最简单的 SGD。AdamW 会结合梯度的一阶、二阶
移动平均以及权重衰减来决定实际步长，所以不能直接用上面的单行公式复现真实更新
值；但输入给 AdamW 的核心信息仍然是每个参数的 `.grad`。优化器额外维护的移动
平均属于 **optimizer state**，不是 loss 的参数，也不是模型的前向输出。

还要注意，本项目使用 `GradScaler`：CUDA 混合精度训练时先临时放大 loss，再反传
得到放大的梯度，随后 `unscale_` 恢复真实尺度，裁剪梯度，最后 `step`。这种缩放
只为防止小梯度在 FP16 中下溢，不改变“通过链式法则求 `∂L/∂参数`”这一含义。

梯度告诉我们“该往哪里更新”，接下来的问题是：模型由成千上万个运算组成，最终
的 loss 怎样追溯到前面的每一个参数？答案是**链式法则**。

先看一个极简计算图：

```text
x --乘 w--> z --平方--> loss
z = wx
loss = z²
```

`w` 并不直接出现在 `loss = z²` 这一层里，但它先影响 `z`，`z` 再影响 loss。
因此 `w` 对 loss 的总影响是两段局部影响的乘积：

```text
∂loss/∂w = ∂loss/∂z × ∂z/∂w
          = 2z × x
```

如果 `x=3、w=2`，则：

```text
z = wx = 6
∂loss/∂z = 2z = 12
∂z/∂w = x = 3
∂loss/∂w = 12 × 3 = 36
```

可以把它理解成两次“敏感程度”的传递：

1. `w` 改一点，`z` 会改多少；
2. `z` 改一点，loss 会改多少；
3. 两者相乘，就得到 `w` 改一点时 loss 最终会改多少。

#### 为什么要从后往前算

设一个稍长的模型是：

```text
w → a → b → c → loss
```

求最前面参数 `w` 的梯度需要：

```text
∂loss/∂w
= ∂loss/∂c × ∂c/∂b × ∂b/∂a × ∂a/∂w
```

从 loss 往回算时，可以不断复用已经算出的结果：先得到 `∂loss/∂c`，再得到
`∂loss/∂b`，接着得到 `∂loss/∂a`，最后得到 `∂loss/∂w`。神经网络中许多参数
共享后半段计算；反向传播会复用这些中间结果，而不是为每个参数重新执行一遍完整
求导，所以能高效地一次算出所有参数的梯度。

如果一个变量通过多条路径影响 loss，各条路径的影响还要相加。例如：

```text
        ┌→ a ─┐
w ──────┤     ├→ loss
        └→ b ─┘

∂loss/∂w
= ∂loss/∂a × ∂a/∂w
+ ∂loss/∂b × ∂b/∂w
```

所以反向传播的基本规则可以概括成：

- 一条路径上的局部导数相乘；
- 多条路径产生的影响相加；
- 从最终的标量 loss 开始，沿计算图反向传递。

在本项目中，大致就是：

```text
loss → output Linear → Decoder → Encoder → Embedding
```

`loss.backward()` 做的是自动应用这些规则，并把结果写入各参数的 `.grad`。它
**只是一种高效计算梯度的方法，不负责选择更新策略，也不负责修改参数**。梯度
算完后，SGD、AdamW 等优化器才根据 `.grad` 和学习率更新参数。

#### 从单个函数过渡到矩阵：矩阵只是许多小函数并排计算

从单个参数过渡到矩阵时，最容易产生的误解是把“矩阵运算”看成一种全新的、
难以拆开的运算。实际上，矩阵只是在整齐地同时执行大量“相乘再相加”的小函数。

以模型最后的输出层为例：

```python
self.output = nn.Linear(d_model, vocab_size)
```

对于某一个输出位置，Decoder 会产生一个 `D` 维隐藏向量，输出层再给词表中的
`V` 个 token 分别打分。为了能完整写出数字，暂时假设：

```text
D = 2
V = 3

隐藏向量 h = [2, -1]
候选 token = [I, love, you]
```

输出层的权重矩阵假设为：

```text
             隐藏维度 1  隐藏维度 2

I          [    0.1         0.3 ]
love       [    0.4        -0.2 ]
you        [   -0.3         0.5 ]
```

也就是：

```text
W = [[ 0.1,  0.3],
     [ 0.4, -0.2],
     [-0.3,  0.5]]
```

矩阵的每一行都是一个 token 的“评分员”。每个评分员读取隐藏向量的两个维度，
给自己的 token 算出一个 logit：

```text
I 的分数    = 0.1×2 + 0.3×(-1)  = -0.1
love 的分数 = 0.4×2 - 0.2×(-1)  =  1.0
you 的分数  = -0.3×2 + 0.5×(-1) = -1.1
```

把三次计算一起写，就是矩阵乘法：

```text
logits = W h = [-0.1, 1.0, -1.1]
```

因此，`W` 有 `V` 行是因为要给 `V` 个 token 打分；每行有 `D` 列是因为每个
评分员都要观察隐藏向量的 `D` 个特征。真实模型中只是数字更多：

```text
h       [D]
W       [V, D]
logits  [V]
```

例如当前默认 `D=256、V=8000` 时，可以想成 8000 个评分员，每个评分员用自己的
一组 256 个权重观察同一个隐藏向量。

#### 一个输出 token 的 loss 怎样变成输出矩阵的每个梯度

继续使用上面的例子。假设这个位置的正确 token 是 `you`，softmax 后模型给出的
概率大约是：

```text
I:     0.229
love:  0.687
you:   0.084
```

模型错误地偏向了 `love`。交叉熵对每个 logit 的偏导有一个重要结论：

```text
∂loss/∂logit_i = 该 token 的预测概率 - 该 token 是否为正确答案
```

正确答案 `you` 的 one-hot 向量是 `[0, 0, 1]`，所以：

```text
                  概率 - 正确答案 = logit 梯度

I:                0.229 - 0        =  0.229
love:             0.687 - 0        =  0.687
you:              0.084 - 1        = -0.916
```

记作：

```text
g = ∂loss/∂logits = [0.229, 0.687, -0.916]
```

可以把 `g` 理解为 loss 对输出层发回来的三条意见：

```text
I 的分数：     有些高，应该降低
love 的分数：  明显太高，应该降低
you 的分数：   明显太低，应该提高
```

梯度为正表示“增加这个 logit 会使 loss 增大”，梯度为负表示“增加这个 logit
会使 loss 减小”。优化器执行“参数减去梯度”，所以最终会倾向于压低错误 token
的分数、提高正确 token 的分数。

接着看一个具体权重 `w₂₁`。它连接隐藏向量的第一个维度和 `love` 的 logit：

```text
love_logit = w₂₁h₁ + w₂₂h₂
```

前向传播时，`w₂₁` 被乘上了 `h₁`，所以：

```text
∂love_logit/∂w₂₁ = h₁
```

根据链式法则：

```text
∂loss/∂w₂₁
= ∂loss/∂love_logit × ∂love_logit/∂w₂₁
= 0.687 × 2
= 1.374
```

另一个权重得到：

```text
∂loss/∂w₂₂ = 0.687 × (-1) = -0.687
```

所以 `love` 对应的整行权重梯度为：

```text
0.687 × [2, -1] = [1.374, -0.687]
```

对另外两个 token 做同样的计算：

```text
I:      0.229 × [2, -1] = [ 0.458, -0.229]
love:   0.687 × [2, -1] = [ 1.374, -0.687]
you:   -0.916 × [2, -1] = [-1.832,  0.916]
```

最终整个权重矩阵的梯度为：

```text
∂loss/∂W = [[ 0.458, -0.229],
             [ 1.374, -0.687],
             [-1.832,  0.916]]
```

它和原权重矩阵形状相同：

```text
W.shape      = [3, 2]
W.grad.shape = [3, 2]
```

矩阵写法只是把刚才所有逐元素计算压缩成一次计算：

```text
∂loss/∂W = g hᵀ
```

这里的 `g hᵀ` 称为外积。与其先记住这个名词，不如先记住它的含义：

> 每个输出 token 收到的误差信号，乘以前向传播时的输入向量，构成权重矩阵中
> 对应一行的梯度。

因此，对任何一条具体连接，都可以用下面的直觉理解：

```text
某个权重的梯度
= 后面传回来的错误程度
× 前向传播时经过这条连接的输入值
```

一个连接应该承担多少“责任”，同时取决于最终预测错得多严重，以及前向传播时
这条连接实际参与了多少。如果输入是零，这个权重即使连接着很大的下游误差，
本次计算得到的梯度也会是零。

#### 输出层怎样把梯度继续传回 Decoder

输出层不仅要计算自己权重的梯度，还需要告诉 Decoder：“为了减小 loss，你给我
的隐藏向量应该怎样变化？”

输出层的前向计算是：

```text
logits = W h
```

已经知道 loss 对 logits 的梯度是 `g`，那么传给隐藏向量的梯度是：

```text
∂loss/∂h = Wᵀg
```

使用上面的数字，结果大约为：

```text
∂loss/∂h = [0.573, -0.527]
```

它表示在当前位置附近：

```text
h₁ 增大会使 loss 增大，应该倾向于减小；
h₂ 增大会使 loss 减小，应该倾向于增大。
```

这里不会直接更新 `h`，因为 `h` 只是本次前向传播产生的临时结果，不是模型参数。
这个梯度会继续传给产生 `h` 的 Decoder 层。Decoder 层再做同样两件事：

1. 根据收到的梯度，计算自己的参数梯度；
2. 计算对上一层输出的梯度，并继续向前传递。

因此，任意一个不考虑 bias 的 Linear 层：

```text
y = Wx
```

收到下游梯度 `g_y = ∂loss/∂y` 后，核心计算都是：

```text
本层权重的梯度： ∂loss/∂W = g_y xᵀ
传给上一层的梯度：∂loss/∂x = Wᵀg_y
```

可以把完整过程想成：

```text
前向：

x ──经过 W──> y ──后续计算──> loss

反向：

x <──上一层意见── W <──下游意见── loss
                    │
                    └──同时计算 W 自己的梯度
```

Transformer 中虽然还有 Attention、LayerNorm、激活函数和残差连接，但没有出现
新的反向传播原则：一条路径上的局部影响相乘，多条路径的影响相加，每个参数都
根据自己收到的下游梯度和前向传播时保存的信息计算梯度。

#### 多个位置和整个 batch 的梯度怎样汇总

实际训练时不只有一个输出 token。例如目标句：

```text
[BOS, I, love, you, EOS]
```

会产生四个预测任务：

```text
BOS              → I
BOS I            → love
BOS I love       → you
BOS I love you   → EOS
```

每个位置都使用同一个 `self.output` 权重矩阵，因此每个位置都会对这个矩阵产生
一份梯度。一个 batch 中还有多句话，所以反向传播最终会把所有有效位置通过同一
参数产生的梯度累加起来。由于本项目的 `cross_entropy` 默认对非 PAD 位置取平均，
最终得到的梯度也是这些有效位置综合后的平均意见。

这也是为什么代码可以把：

```text
logits  [B, T-1, V]
labels  [B, T-1]
```

展平后得到一个标量 loss，再只调用一次：

```python
loss.backward()
```

一个标量 loss 已经汇总了整个 batch 中所有有效输出位置的预测错误。反向传播会
自动把这些位置通过共享参数产生的影响相加，再写入每个参数的 `.grad`。

但这不表示每个 token 都会明显改变每个参数。某个参数的梯度可能很小或为零，
例如：

- 前向输入接近零；
- 下游传回的误差信号接近零；
- 该参数不在影响这个输出的计算路径上；
- 多个 token 产生的正负梯度互相抵消；
- 某条 Attention 路径被 mask 阻断。

Embedding 是一个容易观察的例子：查表时只有当前 batch 实际使用的 token 行参与
计算，因此未使用的 token 行通常不会从这个 batch 获得直接梯度。Transformer
内部的大多数权重矩阵则由许多 token 位置共享，会收到这些位置累积起来的梯度。

最后，可以把从输出 token 到每个权重的过程压缩成下面这幅“脑内动画”：

```text
每个位置的 D 维隐藏向量
        ↓
输出矩阵的 V 行分别给 V 个 token 打分
        ↓
交叉熵发现正确 token 分数太低、错误 token 分数太高
        ↓
产生对 V 个 logits 的梯度，即对各 token 分数的意见
        ↓
每个 token 的意见 × 该位置的隐藏向量
        ↓
得到输出矩阵每个元素的梯度
        ↓
输出矩阵同时把意见转换成对隐藏向量的梯度
        ↓
Decoder、Encoder 和 Embedding 逐层重复同样的链式法则
```

最值得记住的一句话是：

> 前向传播时，每个权重把输入乘进计算；反向传播时，同一个权重根据“当时的
> 输入有多大”和“最终错误传回来有多大”，计算自己应该改变多少。

## 3. 数据与 SentencePiece

### 3.1 为什么不能直接把文字送进模型

神经网络只处理数字，所以要先把文字映射成 token id。例如结果可能类似：

```text
"我喜欢机器学习"
→ [BOS, 128, 931, 55, 847, EOS]
→ [2, 128, 931, 55, 847, 3]
```

这里的数字没有大小语义；token 931 不代表比 token 128 “更大”，只是查表编号。

### 3.2 BPE 子词

SentencePiece 的 BPE 不要求每个完整单词都进入词表，而会学习常见子词。例如：

```text
unbelievable → un + believe + able
```

优点是：

- 词表无需收录所有英文单词；
- 生僻词仍可拆成较小片段；
- 中文和英文可共享一套 token id；
- 词表大小固定，直接决定输出层的大小 `V`。

`encode()` 在 SentencePiece 结果首尾加入 BOS/EOS。BOS 告诉解码器从哪里开始，
EOS 让模型学会何时结束。

### 3.3 为什么需要 PAD

一个 batch 中句子长短不同，但 Tensor 必须是规则矩形，所以短句需要补齐：

```text
[BOS, 你好, EOS, PAD, PAD]
[BOS, 我, 很, 高兴, EOS]
```

PAD 只是占位，不属于句子内容。因此代码在三个地方忽略 PAD：

1. `Embedding(padding_idx=PAD)`：PAD 词向量不正常更新；
2. Transformer 的 padding mask：注意力不读取 PAD；
3. `cross_entropy(ignore_index=PAD)`：PAD 不计入 loss。

### 3.4 `DataLoader` 与 `collate_fn`

`DataLoader` 负责：

- 从数据集中取样本；
- 组成 batch；
- 每轮打乱训练数据；
- 调用 `collate` 补齐不同长度的序列。

这个项目的数据集只是 Python 列表，因此没有自定义 `Dataset` 类。数据量能放入
内存时，这种写法简单而合理。

## 4. 模型结构

### 4.1 Embedding

`nn.Embedding(V, D)` 可以理解成一张 `[V, D]` 的可训练表格。输入 token id 后，
它取出对应行：

```text
token ids [B, L]
       ↓ Embedding
vectors   [B, L, D]
```

这里并不是把 `[B,L]` 和 `[V,D]` 做普通矩阵乘法。`token ids` 中保存的是整数行号，
Embedding 会对其中的每一个 id 分别查表：

```text
Embedding 权重表 E [V,D]

E[0] = token 0 的 D 维向量
E[1] = token 1 的 D 维向量
...
E[V-1] = token V-1 的 D 维向量
```

输入中的一个标量 `token_ids[b,l]` 是某个 token 的 id。假设它等于 `k`，Embedding
就取出权重表的第 `k` 行：

```text
vectors[b,l,:] = E[token_ids[b,l]] = E[k]

token_ids[b,l]       是一个整数，shape 可看作 []
E[k]                 是一个向量，shape 为 [D]
vectors[b,l,:]       因而也是一个 [D] 向量
```

对 `[B,L]` 中的每个位置都做一次查表，再保持原来的 batch 维和位置维排列，就得到
`[B,L,D]`。可以把它理解为：原来每个位置只有一个 token id，查表后这个 id 被替换
成了对应的 `D` 维向量。

例如设 `B=2、L=3、V=5、D=4`，Embedding 权重表为：

```text
E [5,4]

token 0 → [0.1, 0.2, 0.3, 0.4]
token 1 → [1.1, 1.2, 1.3, 1.4]
token 2 → [2.1, 2.2, 2.3, 2.4]
token 3 → [3.1, 3.2, 3.3, 3.4]
token 4 → [4.1, 4.2, 4.3, 4.4]
```

输入 token id：

```text
token_ids [2,3]

[[1, 3, 2],
 [4, 1, 0]]
```

逐个取出第 `1、3、2、4、1、0` 行后，输出为：

```text
vectors [2,3,4]

[[[1.1, 1.2, 1.3, 1.4],    # E[1]
  [3.1, 3.2, 3.3, 3.4],    # E[3]
  [2.1, 2.2, 2.3, 2.4]],   # E[2]

 [[4.1, 4.2, 4.3, 4.4],    # E[4]
  [1.1, 1.2, 1.3, 1.4],    # E[1]
  [0.1, 0.2, 0.3, 0.4]]]   # E[0]
```

所以这几个维度的来源是：

```text
B、L：来自输入 token_ids，保留“第几句话、句中第几个位置”的结构
D：来自 Embedding 表每一行的宽度，即每个 token 的向量维度
V：只决定表中可供查询的行数，不会出现在输出 shape 中
```

从数学上也可以把查表看成 token id 先变成长度为 `V` 的 one-hot 向量，再乘
Embedding 矩阵：

```text
one_hot(token_id) [V] × E [V,D] = token_vector [D]
```

但实际实现通常直接按 id 取行，不需要真的创建巨大的 one-hot 矩阵，这样更节省
内存和计算量。

一开始向量是随机的。训练过程中，有相似用法的 token 往往会逐渐学出具有相关性
的向量。

### 4.2 位置编码

Self-Attention 本身不会自动区分词序。若不加入位置信息，“我喜欢你”和“你喜欢
我”对它而言可能过于相似。

代码使用固定正弦/余弦位置编码：

```python
pe[:, 0::2] = sin(...)
pe[:, 1::2] = cos(...)
```

`0::2` 表示偶数列，`1::2` 表示奇数列。不同维度采用不同频率，使每个位置得到
独特且平滑变化的向量。

### 4.3 Attention 的核心：每个 token 都生成 Q、K、V

Transformer 的特色不是普通矩阵乘法本身，而是让每个 token 根据当前内容，动态
决定应该从哪些位置读取信息。这个过程通过 Query、Key、Value 完成，简称 Q、K、V。

先假设一条序列经过 Embedding 和位置编码后得到：

```text
X [B, L, D]
```

其中每个位置都有一个 `D` 维向量。Attention 使用三组不同的可训练参数，把同一个
输入投影成三种角色：

```text
Q = XW_Q    # Query：我正在寻找什么信息
K = XW_K    # Key：我这里包含什么、可用什么特征被匹配
V = XW_V    # Value：如果关注我，实际取走什么信息
```

这里的 `W_Q`、`W_K`、`W_V` 都是模型参数，会通过反向传播学习；Q、K、V 则是
当前 batch 根据输入和这些参数临时计算出的中间 Tensor，不是长期保存的参数。

若暂时不拆分多头，三组投影参数的形状通常可以写成：

```text
X    [B,L,D]
W_Q  [D,D]
W_K  [D,D]
W_V  [D,D]
```

`X` 中一共有 `B×L` 个 token 向量，每个向量的 shape 都是 `[D]`。这里的 `[D]`
只表示一维向量；为了看清矩阵乘法，可以明确把它写成一个行向量 `[1,D]`。同一个
位置的 `D` 维向量分别经过三次线性变换：

```text
x[b,l,:] [1,D] × W_Q [D,D] = q[b,l,:] [1,D]
x[b,l,:] [1,D] × W_K [D,D] = k[b,l,:] [1,D]
x[b,l,:] [1,D] × W_V [D,D] = v[b,l,:] [1,D]
```

矩阵乘法的规则是中间两个维度相同并被消去：

```text
[1,D] × [D,D]
   ↑       ↑
   └── D ──┘  相乘并求和

结果保留外侧维度：[1,D]
```

因此不是把原来的一个 `D` 维度扩展成 `[D,D]`。`[D,D]` 是线性变换使用的参数表，
经过乘加后，每个 token 仍然只输出一个 `D` 维向量。

如果把 token 写成列向量 `[D,1]`，就必须把参数矩阵写在左边：

```text
W_Q [D,D] × x [D,1] = q [D,1]
```

而下面这种写法不能相乘，因为内侧维度分别是 `1` 和 `D`：

```text
x [D,1] × W_Q [D,D]     # 维度不匹配
```

容易混淆的另一种运算是外积：

```text
x [D,1] × y [1,D] = M [D,D]
```

外积确实会得到 `[D,D]`，但 QKV 的前向投影不是这个运算。QKV 前向投影是
`[1,D] × [D,D] → [1,D]`，或者采用列向量记法时的
`[D,D] × [D,1] → [D,1]`。

PyTorch 的 `nn.Linear(D,D)` 在内部把权重保存为 `[out_features,in_features]`，
即 `[D,D]`，实际计算可写成：

```text
q = x W_Qᵀ + b_Q
```

因为这里输入维和输出维恰好都等于 `D`，转置前后的 shape 都是 `[D,D]`，但元素的
方向仍有区别。为了讲解方便，本文其余位置把概念参数记成适合右乘的 `W_Q`，写作
`q=xW_Q`。

对所有 batch、所有位置并行计算后，可以直接理解为产生了三个形状相同的三维
Tensor：

```text
X [B,L,D]
 ├─× W_Q [D,D]──→ Q [B,L,D]
 ├─× W_K [D,D]──→ K [B,L,D]
 └─× W_V [D,D]──→ V [B,L,D]
```

所以，**可以把它理解成 `[B,L,D]` 的输入乘完 QKV 投影后，得到三个独立的
`[B,L,D]` 张量**。它们的 shape 相同、数值和用途不同。这里的乘法作用于最后一个
维度 `D`，前面的 `B、L` 维只是把大量 token 的计算组织在一起并行执行。

输出中的每一个特征也不是简单地与输入的同编号特征相乘。采用行向量右乘记法，
以 Query 为例：

```text
q[b,l,j] = Σ_i x[b,l,i] × W_Q[i,j]
```

因此一个 token 的每个 Query 特征，通常都是原来全部 `D` 个输入特征的加权组合。
如果线性层带 bias，还会在右侧加上 `b_Q[j]`；K、V 同理。

进入多头计算前，再把最后一个维度拆开。设头数为 `H`，每个头的维度为
`d_head=D/H`：

```text
Q、K、V [B,L,D]
       ↓ D = H × d_head
       [B,L,H,d_head]
       ↓ 调整维度顺序
       [B,H,L,d_head]
```

这一步主要是重新组织已有元素，并没有凭空增加数据。Attention 计算完成后，各头
又会拼接回 `[B,L,D]`。

#### QKV 参数共享到什么范围

同一个 Attention 模块中的所有句子、所有 token 位置共享同一套
`W_Q、W_K、W_V`：

```text
q[0,0] = x[0,0] W_Q
q[0,1] = x[0,1] W_Q
q[1,0] = x[1,0] W_Q
```

输入向量不同，所以算出的 Query 不同；但把输入变成 Query 的规则 `W_Q` 相同，
K、V 也是如此。

不过，**整个 Transformer 通常不只有一套 QKV 参数**。不同 Attention 模块拥有
各自独立的参数。例如：

```text
Encoder 第 1 层 Self-Attention：一套 QKV 参数
Encoder 第 2 层 Self-Attention：另一套 QKV 参数

Decoder 某层 Self-Attention：一套 QKV 参数
Decoder 同层 Cross-Attention：另一套 QKV 参数
```

PyTorch 为了提高计算效率，可能把 `W_Q、W_K、W_V` 合并存储成一个更大的参数，
一次得到 `[B,L,3D]`，再切成 Q、K、V。从概念和数学上仍可按三组投影理解。

#### 所有位置传回来的 QKV 梯度怎样汇总

因为同一个 `W_Q` 被许多位置共享，每个相关输出 loss 都可能通过计算图对它产生
一份梯度。反向传播会针对同一个参数把所有路径的影响相加。若 batch 中共有 `N`
个参与平均 loss 的有效目标位置：

```text
L_batch = (L₁ + L₂ + ... + L_N) / N

∂L_batch/∂W_Q
= (1/N) × (∂L₁/∂W_Q + ∂L₂/∂W_Q + ... + ∂L_N/∂W_Q)
```

`W_K、W_V` 采用完全相同的汇总原则。这里不是把 Q、K、V 三个不同参数矩阵的
梯度互相平均，而是：

```text
W_Q.grad：汇总所有路径对 W_Q 的意见
W_K.grad：汇总所有路径对 W_K 的意见
W_V.grad：汇总所有路径对 W_V 的意见
```

最终每个 `.grad` 都与自己的参数 shape 相同，例如 `W_Q` 和 `W_Q.grad` 都是
`[D,D]`。随后优化器分别根据这三份综合梯度更新三组参数。

可以把一个位置想成正在查资料：

```text
Query：我现在想查什么？
Key：每份资料的索引标签是什么？
Value：每份资料的实际内容是什么？
```

某个位置的 Query 会和所有允许访问位置的 Key 做点积：

```text
score(i,j) = q_i · k_j
```

`score(i,j)` 越大，表示第 `i` 个位置当前越应该关注第 `j` 个位置。一次性写成矩阵：

```text
scores = QKᵀ
```

然后除以 `√d_k`，加入 mask，再做 softmax：

```text
A = softmax(QKᵀ / √d_k + mask)
```

- 除以 `√d_k`：避免维度较大时点积绝对值过大，导致 softmax 过早变得极端；
- mask：把禁止关注的位置变成近似负无穷，softmax 后概率接近 0；
- `A`：注意力权重，每一行表示一个 Query 应怎样分配注意力。

最后用注意力权重对所有 Value 加权求和：

```text
Attention(Q,K,V) = A V
                 = softmax(QKᵀ / √d_k + mask)V
```

因此某个位置的新表示不再只包含自己，而是按当前任务动态混合了其他位置的信息。

#### 一个极简的 QKV 例子

假设某个 Query 对三个位置算出的缩放后分数是：

```text
与位置 1 的匹配分数：1.2
与位置 2 的匹配分数：0.2
与位置 3 的匹配分数：2.0
```

softmax 后可能得到：

```text
注意力权重：[0.28, 0.10, 0.62]
```

如果三个位置的 Value 是 `v₁、v₂、v₃`，那么当前位置读到的信息就是：

```text
context = 0.28v₁ + 0.10v₂ + 0.62v₃
```

注意力并不是硬选一个 token，而通常是把多个位置的信息按不同权重混合。下一层
输入变化后，新的 Q、K、V 和注意力分配也会变化。

### 4.4 Self-Attention 与 Cross-Attention 的区别

Q、K、V 的公式相似，区别在于它们从哪里来。

Self-Attention 中三者来自同一序列：

```text
Q = XW_Q
K = XW_K
V = XW_V
```

它表示序列内部各位置互相读取信息：

- Encoder Self-Attention：源句 token 互相观察；
- Decoder Self-Attention：目标前缀 token 互相观察，并受 causal mask 限制。

Cross-Attention 中，Query 来自 Decoder 当前状态，Key 和 Value 来自 Encoder
输出的 `memory`：

```text
Q = decoder_hidden W_Q
K = memory W_K
V = memory W_V
```

它表达的是：

> Decoder 当前要生成目标句的这个位置时，应该从源句哪些位置取信息？

例如生成英文 `apple` 时，Decoder 的 Query 可能与中文“苹果”位置的 Key 匹配较
强，从对应 Value 中读取较多源句信息。这里的注意力关系不是人工规定，而是
`W_Q、W_K、W_V` 在大量翻译样本中逐渐学习出来的。

### 4.5 多头注意力：同时使用多套 QKV 观察不同关系

如果只有一套 Q、K、V，就只有一种投影空间和一种注意力分配。多头注意力会把
`D` 维隐藏空间分成 `H` 个头，每个头使用自己的一套 QKV 投影：

```text
head_h = Attention(XW_Q^h, XW_K^h, XW_V^h)
```

各头可以学习不同类型的关系，例如某些头可能更关注：

- 临近词和局部短语；
- 主语与谓语；
- 指代关系；
- 源语言与目标语言中的对齐位置；
- 标点、句尾或长距离依赖。

这只是便于理解的可能性，并不表示每个头一定能被稳定命名为某种语法功能。

当前默认配置为（`d_model` 可通过命令行修改，头数在代码中固定为 4）：

```python
d_model = 256
nhead = 4
```

所以每个头处理的维度是：

```text
d_head = D / H = 256 / 4 = 64
```

形状可以想成：

```text
输入 X                         [B, L, 256]
Q、K、V 投影后拆成 4 个头      [B, 4, L, 64]
Self-Attention 分数 QKᵀ         [B, 4, L, L]
softmax 后的注意力权重 A         [B, 4, L, L]
每个头执行 A×V 后的结果          [B, 4, L, 64]
4 个头拼接                      [B, L, 256]
再经过输出投影 W_O              [B, L, 256]
```

`[L,L]` 的含义是：这一层里，每个 Query 位置都对每个 Key 位置产生一个匹配
分数。加上 batch 和多头维度后，就是 `[B,H,L,L]`。

Cross-Attention 的目标长度和源句长度可能不同。若 Decoder 当前长度为 `L`、源句
长度为 `S`：

```text
Decoder 的 Q      [B,4,L,64]
memory 的 K、V    [B,4,S,64]
匹配分数 QKᵀ      [B,4,L,S]
注意力权重 A       [B,4,L,S]
加权结果 A×V       [B,4,L,64]
```

所以 `[B,4,L,S]` 可以直接读作：batch 中每句话、每个头、每个目标位置，都对
源句的 `S` 个位置分配一组注意力权重。

完整形式为：

```text
MultiHead(Q,K,V)
= Concat(head₁, head₂, head₃, head₄) W_O
```

这里“多头”主要是并行关系：同一层的多个头读取同一批位置，但使用不同参数和不同
表示子空间；它们算完后拼接，再混合回一个 `D` 维向量。多头不会把序列长度变成
四倍，也不会让每个 token 变成四个独立 token。

概念上可以说每个头有自己的一套 `W_Q^h、W_K^h、W_V^h`。具体实现为了效率，
PyTorch 可能把多个头的投影参数打包进较大的矩阵一次计算，再 reshape 成多个头；
数学效果仍然等价于各头使用不同的投影分片。

### 4.6 Attention 后为什么还要 FFN、残差连接和 LayerNorm

一个 Transformer 层不只有 Attention。

Attention 主要负责**位置之间交换信息**：一个 token 从其他位置取什么。随后每个
位置还会独立通过前馈网络 FFN，对已经汇总的信息做非线性变换：

```text
FFN(x) = Linear₂(激活函数(Linear₁(x)))
```

当前默认配置中：

```text
D = 256
dim_feedforward = 2D = 512

[B,L,256] → Linear → [B,L,512]
          → 激活和 Dropout
          → Linear → [B,L,256]
```

同一个 FFN 参数会独立应用到所有 batch、所有序列位置。它不会在位置之间传递信息；
位置之间的信息交换已经由 Attention 完成。

每个 Attention 或 FFN 子层外还会有：

```text
残差连接：让子层输出与原输入相加
LayerNorm：稳定各隐藏维度的尺度
Dropout：训练时提供正则化
```

可以概括成：

```text
子层输出 = LayerNorm(输入 + Dropout(子层计算(输入)))
```

某些 Transformer 变体会把 LayerNorm 放在子层计算之前；无论具体先后，残差连接的
核心作用都是为信息和梯度提供直接通路，使很多层堆叠时更容易训练。

### 4.7 一层 Encoder 的计算顺序

Encoder 读取源句。单层 Encoder 可以按下面的逻辑理解：

```text
输入 X [B,S,D]
  ↓
多头 Self-Attention
  Q、K、V 都来自 X
  使用 source padding mask
  ↓
残差连接 + LayerNorm
  ↓
逐位置 FFN：D → 2D → D
  ↓
残差连接 + LayerNorm
  ↓
本层输出 [B,S,D]
```

输出 shape 不变，但含义变了：每个源 token 的表示已经读取了其他源位置的信息，
又经过了非线性特征变换。

### 4.8 一层 Decoder 的计算顺序

单层 Decoder 比 Encoder 多一个 Cross-Attention：

```text
目标输入 Y [B,L,D]
  ↓
1. 多头 masked Self-Attention
   Q、K、V 都来自 Y
   使用 target padding mask 和 causal mask
  ↓
残差连接 + LayerNorm
  ↓
2. 多头 Cross-Attention
   Q 来自 Decoder 当前隐藏状态
   K、V 来自 Encoder memory [B,S,D]
   使用 source padding mask
  ↓
残差连接 + LayerNorm
  ↓
3. 逐位置 FFN：D → 2D → D
  ↓
残差连接 + LayerNorm
  ↓
本层输出 [B,L,D]
```

顺序很重要：Decoder 先让目标前缀内部交流，再拿着更新后的目标表示去源句 memory
中查询相关信息，最后用 FFN 进一步加工每个位置的特征。

### 4.9 多头与多层不是一回事

这两个“多”很容易混淆：

| 名称 | 关系 | 作用 |
|---|---|---|
| 多头 `nhead=4` | 同一 Attention 子层内大体并行 | 同时在 4 个表示子空间中建立注意力关系 |
| 多层 `layers=3` | 上一层输出进入下一层，串行堆叠 | 反复读取、混合和加工信息，逐步形成更深表示 |

当前模型默认有（`layers` 可通过命令行修改）：

```text
3 层 Encoder，每层 4 头 Self-Attention
3 层 Decoder，每层有：
  4 头 masked Self-Attention
  4 头 Cross-Attention
```

不能简单说总共有一个“24 头 Attention”来替代它们，因为不同层收到的输入不同，
参数也不同。第 2 层是在第 1 层已经加工过的表示上继续计算，第 3 层再读取第 2 层
的结果。**头是在层内并行观察，层是在深度方向串行加工。**

### 4.10 整个模型一次前向传播的总顺序

把多头、多层、Encoder、Decoder 和输出层全部串起来，按当前默认参数，一次前向是：

```text
源句 token ids [B,S]
  ↓ 共享 Embedding + 位置编码
源句表示 [B,S,256]
  ↓ Encoder 第 1 层：4 头 Self-Attention → FFN
  ↓ Encoder 第 2 层：4 头 Self-Attention → FFN
  ↓ Encoder 第 3 层：4 头 Self-Attention → FFN
memory [B,S,256]

目标前缀 token ids [B,L]
  ↓ 同一个共享 Embedding + 位置编码
目标表示 [B,L,256]
  ↓ Decoder 第 1 层：
      4 头 masked Self-Attention
      → 4 头 Cross-Attention(memory)
      → FFN
  ↓ Decoder 第 2 层：同样顺序，但使用第 1 层输出
  ↓ Decoder 第 3 层：同样顺序，但使用第 2 层输出
Decoder hidden [B,L,256]
  ↓ 输出 Linear：256 → V
logits [B,L,V]
```

每个子层内部还有残差连接、LayerNorm 和 Dropout，为了突出主线没有在总图中重复
画出。

还要注意 Encoder 和 Decoder 的总体依赖关系：必须先把源句经过全部 Encoder 层
得到 `memory`，Decoder 每一层才能通过 Cross-Attention 读取它。Decoder 不会在
每一层重新运行 Encoder；同一份最终 `memory` 会提供给所有 Decoder 层。

训练时 `L=T-1`，一次得到所有位置的 `[B,T-1,V]`；推理时 `L` 是当前前缀长度，
当前实现会在每一步重新运行这个 Decoder 前缀，然后只取最后位置的 `[V]` 来选择
下一个 token。

### 4.11 多层堆叠时，反向传播怎样返回去

正向传播按层从前往后：

```text
Embedding → Encoder 1 → 2 → 3 → Decoder 1 → 2 → 3 → Output → loss
```

反向传播则沿计算图大体反过来：

```text
loss → Output → Decoder 3 → 2 → 1 → Encoder 3 → 2 → 1 → Embedding
```

“大体”是因为残差连接、Decoder 到 memory 的 Cross-Attention 等会形成分支；
PyTorch 会沿所有有效路径应用链式法则，并把同一参数或中间量收到的多路梯度相加。
每一层都有自己独立的 QKV、输出投影、FFN 和 LayerNorm 参数，所以每层最终都会
得到自己的 `.grad`，再由优化器在同一个 step 中统一更新。

### 4.12 logit、softmax 与概率

logit 是尚未归一化的分数，可以是任意实数：

```text
[2.1, -0.3, 4.8, 1.2]
```

softmax 将它们转成和为 1 的概率。分数最大的 token 仍然拥有最大概率，因此贪心
搜索直接 `argmax(logits)` 即可，不必先算 softmax。

训练时也不要手动 softmax，因为 `cross_entropy` 内部会高效、稳定地完成
`log_softmax + negative log likelihood`。

## 5. mask 为什么重要

### 5.1 Source padding mask

`src.eq(PAD)` 得到 `[B, S]` 布尔张量，PAD 位置为 True。Transformer 中 True
表示禁止关注。

### 5.2 Target padding mask

目标句同样需要屏蔽补齐出来的 PAD，避免解码器把占位符当成真实上下文。

### 5.3 Causal mask

训练时，解码器一次收到整个真实目标前缀。若无遮罩，位置 1 可以直接看到位置
2、3 的正确答案，相当于考试偷看后文。

因果遮罩是上三角布尔矩阵：

```text
False  True  True  True
False False  True  True
False False False  True
False False False False
```

第 0 个位置只能看位置 0；第 1 个位置只能看 0、1；依此类推。

Padding mask 解决“哪些位置只是补齐”，causal mask 解决“哪些真实位置属于未来”。
有了这两类限制，下一节才能安全地把多句话、多个目标位置放在一次前向中并行计算。

## 6. 训练为何整句并行，而推理必须逐 token 生成

这是理解 Transformer 训练循环最关键的一节。最常见的误解是：既然翻译模型预测
的是“下一个 token”，是不是每预测一个 token 就要前向、反向并更新一次参数？

答案是：

```text
推理时确实逐 token 前向生成；
训练时通常把所有已知目标位置并行计算，并按一个 batch 更新一次。
```

### 6.1 推理时：生成一个 token，再把它接回输入

实际翻译时，模型只有中文源句，不知道正确英文答案。假设最终结果是
`I love you`，生成过程只能逐步进行：

```text
第 1 次前向：[BOS]              → 预测 I
第 2 次前向：[BOS, I]           → 预测 love
第 3 次前向：[BOS, I, love]     → 预测 you
第 4 次前向：[BOS, I, love, you]→ 预测 EOS
```

伪代码是：

```python
generated = [BOS]
while True:
    logits = model(src, generated)
    next_token = logits[:, -1].argmax(dim=-1)
    generated.append(next_token)
    if next_token == EOS:
        break
```

每一步都依赖上一步实际选出的 token，所以时间方向上不能一次把未知的未来全部算
出来。推理时通常没有正确目标标签，不计算训练 loss，也不执行 `backward()` 和
`optimizer.step()`；模型只使用已经学好的参数反复前向传播。

### 6.2 训练时：正确目标句已经存在

训练数据同时提供：

```text
源句：我爱你
目标句：I love you
```

加入特殊 token 后，目标序列是：

```text
[BOS, I, love, you, EOS]
```

代码把目标序列错开一位：

```python
decoder_input = tgt[:, :-1]
labels = tgt[:, 1:]
```

得到：

```text
Decoder 输入：[BOS, I,    love, you]
正确标签：    [I,   love, you,  EOS]
```

它实际上组成了四道“根据已有前缀预测下一个 token”的题目：

```text
[BOS]                 → I
[BOS, I]              → love
[BOS, I, love]        → you
[BOS, I, love, you]   → EOS
```

训练时每道题都使用**真实前缀**，而不是模型上一位置刚预测出的结果，这叫
**Teacher Forcing**。正因为正确目标句已经存在，模型不必真的生成 `I` 后才能构造
第二道题；四道题的输入可以提前放进同一个 Tensor。

### 6.3 Causal mask 让并行计算不等于偷看答案

Decoder 虽然一次收到 `[BOS, I, love, you]`，但第 5 节介绍的 causal mask 会限制
每个位置只能读取自己和左边的真实前缀：

```text
位置 1 只能看到：[BOS]
位置 2 只能看到：[BOS, I]
位置 3 只能看到：[BOS, I, love]
位置 4 只能看到：[BOS, I, love, you]
```

因此在 GPU 上是一次矩阵并行计算，在逻辑上仍然等价于四道不能看到未来答案的
“预测下一 token”任务。Transformer 的训练效率很大程度上就来自这种序列位置的
并行能力。

### 6.4 一次前向到底输出什么

模型一次输出的不是一个 token，也不是一个 logits 向量，而是：

> batch 中每句话的每个目标位置，各自得到一个长度为词表大小的 logits 向量。

假设：

```text
B = 2       # batch 中有 2 句话
T-1 = 4     # 每句话有 4 个待预测位置
V = 8000    # 词表中有 8000 个 token
```

那么：

```text
logits.shape = [2, 4, 8000]
```

三个维度依次表示：

```text
[第几句话, 第几个目标位置, 词表中每个 token 的分数]
```

展开理解就是：

```text
句子 1：
  位置 1 → 8000 个 logits，用来预测 I
  位置 2 → 8000 个 logits，用来预测 love
  位置 3 → 8000 个 logits，用来预测 you
  位置 4 → 8000 个 logits，用来预测 EOS

句子 2：
  位置 1 → 8000 个 logits
  位置 2 → 8000 个 logits
  位置 3 → 8000 个 logits
  位置 4 → 8000 个 logits
```

所以“模型预测下一个 token”描述的是**每个位置在做什么**；`[B,T-1,V]` 描述的
则是训练时把大量这样的预测任务怎样一起计算。两种说法并不矛盾。

### 6.5 每个位置有自己的 token loss

每个有效目标位置都是一次从 `V` 个 token 中选择正确 token 的分类。假设某位置
的正确答案是 `love`，模型概率为：

```text
I: 0.10, love: 0.70, you: 0.20
```

该位置的交叉熵为：

```text
token_loss = -log(0.70) ≈ 0.357
```

如果只给正确答案 `0.01` 的概率：

```text
token_loss = -log(0.01) ≈ 4.605
```

虽然交叉熵数值只读取正确 token 的概率，但这个概率是所有 logits 经过 softmax
竞争出来的，所以反向时词表里的每个 logit 都会获得梯度。完整推导见 2.7。

### 6.6 所有有效 token loss 汇总成一个 batch loss

假设两句话各有四个有效预测位置，就会得到八个 token loss：

```text
句子 1：L₁₁  L₁₂  L₁₃  L₁₄
句子 2：L₂₁  L₂₂  L₂₃  L₂₄
```

代码把 shape 展平，只是为了交给交叉熵接口：

```python
logits.reshape(-1, logits.size(-1))  # [B*(T-1), V]
labels.reshape(-1)                   # [B*(T-1)]
```

`ignore_index=PAD` 排除补齐位置，默认的 `reduction="mean"` 对其余有效位置取平均：

```text
batch_loss
= (L₁₁ + L₁₂ + L₁₃ + L₁₄
   + L₂₁ + L₂₂ + L₂₃ + L₂₄) / 8
```

最终得到一个标量 batch loss。它不是说 batch 里只有一道题，而是把所有题的错误
程度汇总成一个可用于反向传播的总目标。

### 6.7 为什么一个 batch 只 backward 和更新一次

如果 batch loss 是 `N` 个有效 token loss 的平均值：

```text
L_batch = (L₁ + L₂ + ... + Lₙ) / N
```

那么对任意模型参数 `w`：

```text
∂L_batch/∂w
= (1/N) × (∂L₁/∂w + ∂L₂/∂w + ... + ∂Lₙ/∂w)
```

因此一次 `batch_loss.backward()` 已经把 batch 中所有句子、所有有效位置对参数
`w` 的意见汇总到 `w.grad`。随后一次 `optimizer.step()` 根据这份综合意见更新
参数。

可以把它想成：

```text
预测 I 的位置认为：    w 应该减小一些
预测 love 的位置认为： w 应该增大一些
预测 you 的位置认为：  w 应该减小更多
其他句子的各位置：     也各自给出意见
                 ↓ 求和并平均
当前 batch 对 w 的最终梯度
```

理论上可以每个 token 更新一次，但通常不这样做，因为它会失去序列并行能力、重复
大量相似计算、不能充分利用 GPU，而且单个 token 的梯度噪声很大。batch 更新通常
更高效、更稳定。

### 6.8 把训练与推理放在一起比较

| 阶段 | 知道正确目标句 | 目标位置怎样计算 | 计算训练 loss | 反向并更新参数 |
|---|---:|---|---:|---:|
| 训练 | 是 | 整句、整个 batch 并行 | 是 | 是，每个 batch 一次 |
| 验证 loss | 是 | 整句、整个 batch 并行 | 是 | 否 |
| 实际翻译 | 否 | 逐 token 自回归前向 | 通常否 | 否 |

最简洁的流程图是：

```text
训练：
一个 batch 的多句话
→ 所有目标位置并行输出 [B,T-1,V]
→ 每个位置计算 token loss
→ 有效 token loss 取平均
→ backward 一次
→ 更新参数一次

推理：
当前前缀
→ 前向预测下一个 token
→ 把预测接回前缀
→ 再次前向
→ 直到 EOS；全程不 backward
```

loss 越低通常越好，但还要结合验证集和实际译文判断：

- train loss 下降、valid loss 上升：可能过拟合；
- train 和 valid 都高：可能尚未学会、模型不足或数据有问题；
- valid loss 较低不保证译文一定自然，仍需 BLEU/COMET 或人工样例检查。

## 7. 一次参数更新发生了什么

第 6 节解释了为什么一个 batch 能同时产生许多 token 预测，并最终只汇总成一个
batch loss。本节不再重复位置级数学，而是把这件事逐行对应到 `run_epoch()`：一个
batch 从清梯度到更新参数究竟执行哪些操作。

先把 `run_epoch()` 中一次训练 step 的真实顺序完整列出来：

```text
1. zero_grad        清掉上一批留下的梯度
2. model(...)       用当前参数进行前向传播
3. cross_entropy    比较预测和答案，得到标量 loss
4. backward         从 loss 反推每个参数的梯度
5. unscale + clip   恢复真实梯度尺度，并防止梯度过大
6. optimizer.step   根据梯度修改参数
7. scheduler.step   调整下一 step 使用的学习率（若启用）
```

第 2 步到第 4 步回答“错在哪里、每个参数对此负多少责任”，第 6 步才真正让模型
发生变化。

### 7.1 `zero_grad`

PyTorch 默认累加梯度。若不清空，当前 batch 的梯度会叠加上一个 batch：

```python
optimizer.zero_grad(set_to_none=True)
```

### 7.2 前向传播

```python
logits = model(src, decoder_input)
loss = cross_entropy(logits, labels)
```

模型根据当前参数做预测，再计算错误程度。

### 7.3 反向传播

```python
scaler.scale(loss).backward()
```

自动求导计算 `loss` 对每个可训练参数的偏导，并写入对应参数的 `.grad`。梯度本身
指向 loss 上升方向，优化器通常沿其反方向更新。

以本项目为例，梯度大致沿下面的反方向传播：

```text
loss
  ← output Linear 的权重和输入
  ← Decoder 各层参数
  ← Encoder memory 与 Encoder 各层参数
  ← token Embedding 参数
```

位置编码 `pe` 是 buffer，不是可训练参数，所以不会得到用于更新的 `.grad`；PAD
对应的 Embedding 行也因 `padding_idx=PAD` 不接受常规梯度更新。

### 7.4 梯度裁剪

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

它限制所有参数梯度的整体范数，减少偶发巨大梯度造成训练发散的风险。它不会把
每个梯度都简单截到 `[-1, 1]`，而是在整体过大时按比例缩小。

### 7.5 `optimizer.step`

AdamW 根据梯度、历史一阶/二阶统计和学习率更新参数。`weight_decay=0.01` 还会
抑制参数无限增大，提供一定正则化。

最基础的 SGD 更新公式是：

```text
parameter = parameter - learning_rate × gradient
```

本项目实际使用 AdamW，所以不是机械地直接套用这一条。AdamW 会参考梯度的移动
平均和平方移动平均，为不同参数自适应地调整步幅，并单独施加 weight decay。
但“利用梯度寻找更低 loss 的参数”这个核心目标没有变化。

### 7.6 一次更新后，模型学到了什么

一次更新通常不会让模型突然“学会一句翻译”。它只会让当前 batch 中正确 token
在类似上下文下变得稍微更容易被预测。例如看到许多“我喜欢……”与“I love …”
的配对后，相关 Embedding、Attention 和输出层参数经过大量小更新，才逐渐形成
稳定对应关系。

这也解释了几个常见现象：

- batch 太小：每次更新受少量样本影响，梯度噪声较大；
- 学习率太大：每步跨得太远，可能越过较好的区域并震荡；
- 学习率太小：方向可能正确，但移动太慢；
- 数据有错误：模型仍会认真降低这些错误样本的 loss，学到错误对应关系；
- 训练轮数过多：模型可能越来越贴合训练集，却不能更好地翻译新句子。

### 7.7 `backward`、`step` 和 `zero_grad` 的常见误区

- `loss.backward()` **不更新参数**，只填充或累加 `.grad`；
- `optimizer.step()` **不重新计算梯度**，只使用当前已有的 `.grad`；
- `optimizer.zero_grad()` **不重置模型权重**，只清理梯度；
- loss 下降不等于每个 batch 都下降，随机 batch 和 Dropout 会造成波动；
- 梯度为负不表示“训练坏了”，它只表示该参数局部应向增大方向更新；
- 参数很多时不能靠观察单个梯度判断模型是否学会，应同时看训练/验证 loss 和
  实际译文。

## 8. 混合精度：`autocast` 与 `GradScaler`

GPU 上使用 FP16/BF16 可以减少显存和加快部分运算，但 FP16 能表示的数值范围比
FP32 小，小梯度可能下溢成 0。

- `autocast`：自动决定哪些运算使用低精度；
- `GradScaler`：暂时放大 loss，从而放大梯度，更新前再还原；
- CPU 路径中二者实际关闭，不影响普通训练。

这也是为什么代码要按以下顺序：

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()
```

## 9. 学习率、warmup 与 cosine decay

学习率决定每次参数更新走多远：

- 太大：loss 可能震荡或发散；
- 太小：训练稳定，但收敛很慢或卡住。

Transformer 训练初期参数和优化器统计都不稳定，因此常用 warmup：

```text
0 ──线性升高──> 3e-4 ──cosine 平滑下降──> 0
```

项目中的 `LambdaLR` 返回的是基础学习率 `args.lr` 的倍率，而不是实际学习率。
并且 `scheduler.step()` 每个 batch 调用一次，所以这里的 step 是优化步骤，不是
epoch。

## 10. 贪心解码

贪心解码每一步选择当前最可能的 token：

```python
next_id = logits.argmax(dim=-1)
```

优点：

- 快；
- 实现简单；
- 结果稳定。

缺点：局部最优不一定组成全局最优句子。例如第一步第二名的词，可能与后续词
组合出整体概率更高、更自然的译文，但贪心搜索第一步就把它丢掉了。

## 11. Beam Search

Beam Search 同时保存 `beam_size` 条候选。若 beam size 为 2：

```text
第 1 步：保留 [I]、[The]
第 2 步：[I] 和 [The] 分别扩展，再从全部候选中保留总体最好的 2 条
第 3 步：继续扩展
```

代码使用 `log_softmax`，因为序列概率原本是乘法：

```text
P(I) × P(love|I) × P(you|I,love)
```

取对数后变成加法：

```text
log P(I) + log P(love|I) + log P(you|I,love)
```

加法更方便，也避免很多很小的概率相乘造成数值下溢。

Beam Search 不保证必然优于 greedy。beam 太大可能：

- 推理明显变慢；
- 更偏向常见、保守的表达；
- 放大模型自身的概率偏差。

你的实验里 beam size 4 只比 size 2 略好，也说明当前主要瓶颈仍在模型与数据。

## 12. 重复 n-gram 屏蔽

生成模型有时会陷入循环，例如：

```text
the company said that the company said that ...
```

`block_repeated_ngram(..., n=3)` 禁止生成已经出现过的三元组。实现思路是：

1. 取当前序列最后两个 token 作为前缀；
2. 查找历史中相同前缀；
3. 找到历史上跟在该前缀后的 token；
4. 将这些 token 的 logit 设置为负无穷。

这是解码约束，不代表模型真正解决了重复问题；约束太强也可能误伤合理重复。

## 13. `train()`、`eval()` 与 `no_grad()` 的区别

这三个概念很容易混淆：

- `model.train()`：启用训练行为，例如 Dropout；
- `model.eval()`：启用推理行为，例如关闭 Dropout；
- `torch.no_grad()`：不记录计算图，降低推理内存；
- `torch.set_grad_enabled(False)`：动态版本的梯度开关。

`eval()` 不会自动关闭梯度，`no_grad()` 也不会自动关闭 Dropout，所以推理时通常
需要两者同时使用。

## 14. 模型保存与恢复

### 14.1 `best.pt`

只用于推理，保存：

- 最佳模型参数；
- 模型结构配置；
- 翻译方向。

加载时先按 `config` 创建模型，再调用 `load_state_dict`。

### 14.2 `last.pt`

用于继续训练，除了模型参数，还保存：

- optimizer 状态；
- GradScaler 状态；
- scheduler 状态；
- 已完成 epoch；
- 当前最佳 valid loss；
- 训练参数。

只恢复模型参数而不恢复 optimizer，不能算严格意义上的无缝续训。

### 14.3 为什么 tokenizer 不能换

模型只认识 token id。例如训练时 128 代表 `coffee`，换词表后 128 可能代表
`airport`。模型结构和权重虽然能加载，语义映射却完全错了。因此 `.pt` 与
`tokenizer.model` 必须成套保存。

## 15. 推荐的阅读和调试顺序

第一次学习时，不建议从 `main()` 第一行一路硬读到底。推荐顺序：

1. 先读第 1 节和 2.1～2.5，区分 Tensor、参数、loss、梯度和更新；
2. 阅读 `encode`、`collate`，理解文字怎样成为 `src [B,S]` 和 `tgt [B,T]`；
3. 阅读第 4 节和 `Translator.embed`、`forward/decode`，沿
   `Embedding → QKV → 多头 Attention → 多层 Encoder/Decoder → logits` 跟踪
   shape 到 `[B,T-1,V]`；
4. 重点阅读第 5～6 节和 `run_epoch`，理解 causal mask、Teacher Forcing、训练
   并行与 batch loss；
5. 回到 2.6～2.7，细看交叉熵怎样得到 logits 梯度，又怎样传到每个参数；
6. 阅读第 7～9 节，把 `zero_grad → forward → loss → backward → step` 与实际
   优化代码对应起来；
7. 阅读 `translate` 的 greedy 分支，理解推理为什么只能逐 token 自回归生成；
8. 最后学习 Beam Search、混合精度、scheduler 和断点恢复。

阅读时始终抓住下面这条 shape 主线：

```text
src [B,S]，tgt [B,T]
→ decoder_input / labels [B,T-1]
→ Embedding + 位置编码
→ Encoder 多层 Self-Attention，得到 memory [B,S,D]
→ Decoder 多层 masked Self-Attention + Cross-Attention
→ hidden [B,T-1,D]
→ logits [B,T-1,V]
→ 每个 [V] 对应一个 token 分类任务
→ 所有有效位置的 token loss 取平均
→ 标量 batch loss
→ 每个参数各自得到同 shape 的 .grad
```

调试时可以临时打印：

```python
print("src", src.shape)
print("decoder_input", decoder_input.shape)
print("logits", logits.shape)
print("labels", labels.shape)
```

理想输出可能是：

```text
src torch.Size([64, 31])
decoder_input torch.Size([64, 36])
logits torch.Size([64, 36, 8000])
labels torch.Size([64, 36])
```

## 16. 可以亲手做的学习实验

### 实验一：观察分词

在创建 `sp` 后打印：

```python
sentence = "我喜欢学习 PyTorch。"
print(sp.encode(sentence, out_type=str))
print(encode(sp, sentence))
```

观察子词字符串与 token id 的对应关系。

### 实验二：观察 padding mask

在 `run_epoch` 中打印第一批：

```python
print(src[:2])
print(src[:2].eq(PAD))
```

理解为什么 mask 的 True 位正好对应补齐区域。

### 实验三：比较 greedy 和 beam

```bash
python translate.py --out runs/clean_data_fin --text "我明天去机场。" --decode greedy
python translate.py --out runs/clean_data_fin --text "我明天去机场。" --decode beam --beam-size 2
python translate.py --out runs/clean_data_fin --text "我明天去机场。" --decode beam --beam-size 4
```

比较质量和速度，不要只凭单句下结论，应使用固定测试集。

### 实验四：故意移除 causal mask

这个实验只在临时副本中做。训练 loss 可能异常好看，但推理效果会很差，因为训练
时 Decoder 偷看了未来正确答案，而推理时未来答案不存在。这能直观说明“训练
loss 低”不一定代表流程正确。

## 17. 当前代码的定位与限制

这个文件非常适合学习完整流程，但它仍是小型教学/实验实现：

- 每个生成步骤都会重新运行整个 Decoder 前缀，没有 KV cache；
- 只有单句推理，没有批量推理；
- 训练和验证划分是随机句对级划分，未处理近似重复句；
- 没有 BLEU、chrF 或 COMET 等翻译指标；
- 没有 label smoothing、梯度累积和分布式训练；
- Encoder 与 Decoder 使用同一个共享词表，但没有共享输入/输出权重；
- n-gram 屏蔽和长度惩罚是人工解码策略，需要通过验证集调参。

这些不妨碍学习。相反，当前代码把关键步骤都直接写出来，比大型框架更容易看清
数据、张量、loss 和生成之间的关系。
