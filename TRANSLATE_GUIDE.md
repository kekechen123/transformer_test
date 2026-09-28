# `translate.py` 学习指南：从 PyTorch 基础到 Transformer 翻译

这份笔记按程序真实执行顺序讲解 `translate.py`。阅读时不要求提前学过
PyTorch，但最好知道 Python 的函数、类、列表、字典和切片。

## 1. 先建立整体认识

训练翻译模型，本质上是在反复做下面这件事：

```text
中文字符串
  ↓ SentencePiece 分词
中文 token id
  ↓ Encoder
源句的上下文表示 memory
  ↓ Decoder + 已知英文前缀
下一个英文 token 的预测分数
  ↓ 与正确答案比较，计算 loss
  ↓ backward
更新模型参数
```

推理时没有正确英文答案，流程变成：

```text
BOS → 预测 I → 预测 love → 预测 you → 预测 EOS
```

模型每次只预测“下一个 token”，然后把自己的预测接回输入，继续预测。这叫
**自回归生成**。

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
| `S` | 源句补齐后的长度 | 35 |
| `T` | 目标句补齐后的长度 | 42 |
| `D` | 每个 token 的向量维度 | 512 |
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

### 2.6 什么是梯度：先从一维斜率理解

假设模型暂时只有一个参数 `w`，损失函数是：

```text
L(w) = (w - 3)²
```

它在 `w=3` 时最小。导数为：

```text
dL/dw = 2(w - 3)
```

若当前 `w=5`，梯度为 `+4`。正梯度说明“增大 w 会让 loss 上升”，所以应向
相反方向更新：

```text
w_new = w_old - learning_rate × gradient
      = 5 - 0.1 × 4
      = 4.6
```

若当前 `w=1`，梯度为 `-4`：

```text
w_new = 1 - 0.1 × (-4) = 1.4
```

两种情况都向最优点 3 靠近。这就是“梯度下降”中“下降”的含义：**梯度指向
局部上升最快的方向，所以沿负梯度方向走**。

真实 Transformer 有数百万个参数，不再是一条曲线，而是一个极高维的 loss
曲面。梯度也不再是单个数，而是每个参数各自的偏导数组成的集合：

```text
gradient = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]
```

直觉仍然相同：它描述当前位置附近，怎样微调所有参数会使 loss 变化。

### 2.7 反向传播为什么能算出所有梯度

神经网络是许多简单运算的串联。用一个极简例子表示：

```text
x --乘 w--> z --平方--> loss
z = wx
loss = z²
```

要知道 `w` 对最终 loss 的影响，可以用链式法则：

```text
∂loss/∂w = ∂loss/∂z × ∂z/∂w
          = 2z × x
```

如果 `x=3、w=2`，那么 `z=6`，最终梯度是 `2 × 6 × 3 = 36`。反向传播所做的
就是从 loss 开始，沿计算图从后向前，把每个局部导数按链式法则组合起来：

```text
loss → output Linear → Decoder → Encoder → Embedding
```

它不是另一种优化算法，而是一种**高效计算梯度的方法**。梯度算完以后，SGD、
AdamW 等优化器才决定如何利用梯度更新参数。

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

### 4.3 Encoder

Encoder 读取整条源句，并通过多层 Self-Attention 建立各 token 之间的关系。

```text
源句 embedding [B, S, D]
       ↓ Encoder
memory         [B, S, D]
```

`memory` 仍然为源句每个位置保存一个 D 维表示，但每个表示已经融合上下文。

### 4.4 Decoder

Decoder 中有两类关键注意力：

1. 目标句 Self-Attention：观察已经出现的目标 token；
2. Cross-Attention：观察 Encoder 生成的 `memory`，从源句提取信息。

Decoder 最后输出 `[B, T, D]`，再经 `nn.Linear(D, V)` 得到 `[B, T, V]`。
每个位置的 V 个数是所有词表 token 的 logits。

### 4.5 logit、softmax 与概率

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

## 6. Teacher Forcing 与 loss

假设目标句 token 是：

```text
[BOS, I, love, you, EOS]
```

训练代码进行切片：

```python
decoder_input = tgt[:, :-1]
labels = tgt[:, 1:]
```

得到：

```text
输入：[BOS, I,    love, you]
标签：[I,   love, you,  EOS]
```

这叫 Teacher Forcing：训练时每个位置使用真实历史作为前缀，而不是使用模型刚才
可能预测错误的结果。配合 causal mask，四个位置可以在一次前向中并行训练。

`cross_entropy` 衡量正确 token 得到的概率是否足够高。loss 越低通常越好，但：

- train loss 下降、valid loss 上升：可能过拟合；
- train 和 valid 都高：可能尚未学会、模型不足或数据有问题；
- valid loss 较低不保证译文一定自然，仍需 BLEU/COMET 或人工样例检查。

### 6.1 交叉熵究竟在惩罚什么

假设某个位置的正确 token 是 `love`，模型给三个候选的概率为：

```text
I: 0.10, love: 0.70, you: 0.20
```

这个位置的交叉熵是 `-log(0.70)`，约为 `0.357`。如果模型只给正确答案 0.01，
loss 就是 `-log(0.01)`，约为 `4.605`。因此它会强烈惩罚“对正确答案非常没信心”
的预测。

对 logits 求导后，有一个很有用的结论：

```text
某候选 token 的梯度 = 模型概率 - 是否为正确答案
```

正确 token 对应的值是 `p - 1`，通常为负；其他 token 是 `p - 0`，通常为正。
优化器沿负梯度更新后，会倾向于提高正确 token 的 logit、压低错误 token 的
logit。这个信号再通过反向传播一路传回 Decoder、Encoder 和 Embedding。

### 6.2 一个 batch 为什么只有一个 loss

`logits` 的形状是 `[B, T-1, V]`，也就是说 batch 中每句话的每个有效目标位置，
都是一次“从 V 个 token 中选正确 token”的分类。代码先展平：

```python
logits.reshape(-1, logits.size(-1))  # [B*(T-1), V]
labels.reshape(-1)                   # [B*(T-1)]
```

`cross_entropy` 再忽略标签为 PAD 的位置，并默认对其余位置取平均，最终得到一个
标量 loss。`backward()` 通常从这个标量出发，把所有有效 token 对参数的影响汇总
起来。因此一个训练 step 是“根据整个 batch 的平均意见更新一次”，不是每个句子
各更新一次。

## 7. 一次参数更新发生了什么

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

1. `encode` 和 `collate`：先理解文字如何成为 `[B, L]`；
2. `Translator.embed`：理解 `[B, L] → [B, L, D]`；
3. `Translator.forward/decode`：跟踪到 `[B, T, V]`；
4. `run_epoch`：理解 Teacher Forcing、loss 和参数更新；
5. `translate` 的 greedy 分支；
6. 最后学习 Beam Search、混合精度、scheduler 和断点恢复。

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
