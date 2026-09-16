# 用 PyTorch 学一个小型翻译 Transformer

主代码是 `translate.py`，只有 PyTorch 和 SentencePiece 两个依赖。从随机权重开始训练，默认中译英；用 `--reverse` 可另训一个英译中模型。没有下载预训练模型。

## 1. 安装（在有 4060 的电脑上）

建议 Python 3.10～3.12，新建虚拟环境。在 PyTorch 官网 https://pytorch.org/get-started/locally/ 选择你的系统、Pip 和支持的 CUDA 版本，执行它给出的安装命令，然后：

```bash
pip install sentencepiece
python -c "import torch; print(torch.__version__); print('CUDA 可用:', torch.cuda.is_available())"
```

最后应显示 `True`。若为 `False`，检查 NVIDIA 驱动和安装的 PyTorch 是否支持 CUDA，否则代码会走 CPU，训练慢很多。`requirements_translation.txt` 是依赖清单；GPU 机器优先按官网命令安装 PyTorch。

## 2. 数据格式

把数据保存为 UTF-8 编码的 `parallel.tsv`，无表头，每行两列：**中文、英文，中间一个真正的 Tab**。例如下面三行（仅展示格式）：

```text
我喜欢学习。	I like learning.
你今天怎么样？	How are you today?
这本书很有趣。	This book is interesting.
```

一行只放一个句对，句子内部不要包含 Tab 或换行。默认随机抽取 3 万条，并从中留出最多 1000 条验证；词表只从训练部分学习。抽样和数据划分使用固定随机种子，但不同 GPU/软件版本的结果不保证完全一致。

## 3. 先短跑，再增加数据

在代码所在目录打开终端，先用 5000 条、2 轮检查整个流程（此时翻译通常还很差）：

```bash
python translate.py --data parallel.tsv --limit 5000 --epochs 2 --out runs/quick
```

正式做一次学习实验，用默认 3 万条、5 轮：

```bash
python translate.py --data parallel.tsv

or python translate.py --data parallel.tsv --limit 200000 --out runs/zh_en_200k
```

默认配置是编码器 3 层、解码器 3 层，隐藏维度 256、4 个注意力头、前馈维度 512、8000 子词词表、最大长度 96、batch 64。参数量约 8 百万，CUDA 上启用 FP16 混合精度，面向 4060 级别显卡。显存消耗还取决于句长与软件环境；如果显存不足，加 `--batch-size 32` 或 `--batch-size 16`。

这里不承诺固定耗时：第一轮会打印秒数，用它乘以轮数即可粗估同配置的总训练时间。想更快，可用 `--d-model 128 --layers 2 --batch-size 32`；想改善效果，再逐渐增加数据和轮数：

```bash
python translate.py --data parallel.tsv --limit 0 --epochs 10 --out runs/full
```

`--limit 0` 表示用全部数据。默认只处理较短句子，超过 96 个 token（含起止符）的句对会丢弃，实际数量会打印；需要长句时增大 `--max-len`，注意注意力计算和显存开销随长度增加很快。每次新训练请指定新的 `--out`，防止覆盖旧实验。

## 4. 看结果、单独翻译

训练每 100 个 batch 打印一次 loss；每轮打印训练 loss、验证 loss、耗时，以及固定 3 条验证句子的「原文 / 参考译文 / 当前模型预测」。loss 是平均每个非 PAD token 的交叉熵，越低越好，但不是翻译准确率。训练 loss 下降而验证 loss 上升通常说明过拟合。

输出目录包含 `tokenizer.model`、`tokenizer.vocab`、`best.pt` 和 `loss.csv`。`best.pt` 保存验证 loss 最低的一轮；每轮显示的例句来自当轮模型，最终单独翻译则使用保存的最佳模型。`loss.csv` 可以用 Excel 打开画训练曲线。

```bash
python translate.py --out runs/zh_en --text "我喜欢学习。"
python translate.py --out runs/zh_en --text "这本书很有趣。"
```

快速实验请改成 `--out runs/quick`。另训英译中：

```bash
python translate.py --data parallel.tsv --reverse --out runs/en_zh
python translate.py --out runs/en_zh --text "I like learning."
```

复制模型到另一台电脑时，保留同目录的 `best.pt` 和 `tokenizer.model`。超长输入会明确报错；生成达到长度上限时会停止，所以可能出现未完成的句子。

## 5. 如何读代码

按文件中的 1～4 节顺序读，再看 `main()` 如何把它们连起来：

1. **数据与张量**：SentencePiece 用 BPE 把句子切成子词 ID；`collate()` 把变长列表补齐成 `[batch, length]`，展示 DataLoader 的用法。
2. **模型与位置编码**：`nn.Module` 管理层和参数；Embedding 输出 `[batch, length, d_model]`，加正弦位置编码。编码器读取原文，解码器通过交叉注意力读取编码器输出。
3. **逐词生成**：推理只知道原文和已生成的前缀，从 BOS 起步，通过 `argmax` 一个 token 接一个 token 地生成，直到 EOS。
4. **训练循环**：`train/eval` 控制 dropout，`zero_grad → backward → step` 完成优化；验证关闭梯度，保存最佳权重。

最关键的是训练时目标句的右移。例如真实译文是 `[BOS, I, love, you, EOS]`，解码器输入为 `[BOS, I, love, you]`，标签为 `[I, love, you, EOS]`。因果遮罩禁止每个位置看到后续词，所以可以一次并行训练所有位置，又不会偷看答案。原文的编码器没有这个限制，可以双向读取整个原句。交叉熵直接吃 logits，代码不先做 softmax。

`nn.Transformer` 封装了多头注意力、前馈网络、残差连接和 LayerNorm；这份代码重点是理解整体 Transformer 和 PyTorch 训练流程，并未手写这些内部算子。之后可以尝试自己实现一个 EncoderLayer，替换封装层。

## 预期与取舍

3 万条、5 轮主要用来理解流程和观察学习；20 万条干净、领域一致的数据有机会学会常见短句，但泛化质量受数据影响很大，从零小模型不能期待通用翻译产品的效果。前几轮出现重复词、空译文、漏译并不罕见。请同时查看未见过的句子，别只凭训练句判断效果。

为便于阅读，这里采用固定学习率 AdamW 和贪心解码，没有实现原论文的学习率日程、beam search、BLEU 或断点续训。架构是标准 Encoder–Decoder Transformer，训练配方是简化学习版。保存的是推理权重，重新运行训练会从头开始。
