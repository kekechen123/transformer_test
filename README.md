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

`--limit 0` 表示用全部数据。默认只处理较短句子，超过 96 个 token（含起止符）的句对会丢弃，实际数量会打印；需要长句时增大 `--max-len`，注意注意力计算和显存开销随长度增加很快。每次新训练请指定新的 `--out`，防止覆盖旧实验。训练被中断时不要换目录，使用下一节的 `--resume` 从最近完成的一轮继续。

## 4. 断点续训

每轮训练完成后，程序会把最近的完整训练状态保存到输出目录中的 `last.pt`。它包含模型、AdamW 优化器、混合精度缩放器、学习率调度器、最佳验证 loss 和已完成的 epoch。保存时先写临时文件再原子替换，降低保存过程中进程被 kill 导致断点损坏的风险。

例如计划总共训练 30 轮：

```bash
python translate.py \
  --data parallel_wmt_600k_clean.tsv \
  --limit 600000 \
  --epochs 30 \
  --batch-size 64 \
  --d-model 512 \
  --layers 6 \
  --lr 3e-4 \
  --lr-warmup \
  --warmup-ratio 0.1 \
  --out runs/clean_data
```

如果训练被 kill，使用相同参数并增加 `--resume`：

```bash
python translate.py \
  --data parallel_wmt_600k_clean.tsv \
  --limit 600000 \
  --epochs 30 \
  --batch-size 64 \
  --d-model 512 \
  --layers 6 \
  --lr 3e-4 \
  --lr-warmup \
  --warmup-ratio 0.1 \
  --out runs/clean_data \
  --resume
```

`--epochs` 表示目标总轮数，不是续训时额外增加的轮数。例如断点已经完成第 12 轮，仍指定 `--epochs 30`，程序会从第 13 轮训练到第 30 轮。断点按 epoch 保存，因此如果在某轮中途被 kill，该轮会从头重跑。

续训时必须保持 `--data`、`--limit`、`--batch-size`、`--epochs`、`--lr`、warmup 设置、翻译方向和模型结构参数不变；程序会检查这些配置。数据文件内容也不应改变。`loss.csv` 会继续追加，不会覆盖已有记录。

## 5. 看结果、单独翻译

训练每 100 个 batch 打印一次 loss；每轮打印训练 loss、验证 loss 和耗时；每五轮还会用测试文件中的句子做一次 CPU 推理。loss 是平均每个非 PAD token 的交叉熵，越低越好，但不是翻译准确率。训练 loss 下降而验证 loss 上升通常说明过拟合。

输出目录包含 `tokenizer.model`、`tokenizer.vocab`、`best.pt`、`last.pt` 和 `loss.csv`。`best.pt` 保存验证 loss 最低的一轮，用于单独翻译；`last.pt` 保存最近完成的一轮及完整训练状态，用于断点续训。`loss.csv` 可以用 Excel 打开画训练曲线。

```bash
python translate.py --out runs/zh_en --text "我喜欢学习。"
python translate.py --out runs/zh_en --text "这本书很有趣。"
```

快速实验请改成 `--out runs/quick`。另训英译中：

```bash
python translate.py --data parallel.tsv --reverse --out runs/en_zh
python translate.py --out runs/en_zh --text "I like learning."
```

复制模型到另一台电脑推理时，保留同目录的 `best.pt` 和 `tokenizer.model`；如果还要继续训练，还需保留 `last.pt`，并使用相同的数据和训练参数。超长输入会明确报错；生成达到长度上限时会停止，所以可能出现未完成的句子。

## 6. 如何读代码

按文件中的 1～4 节顺序读，再看 `main()` 如何把它们连起来：

1. **数据与张量**：SentencePiece 用 BPE 把句子切成子词 ID；`collate()` 把变长列表补齐成 `[batch, length]`，展示 DataLoader 的用法。
2. **模型与位置编码**：`nn.Module` 管理层和参数；Embedding 输出 `[batch, length, d_model]`，加正弦位置编码。编码器读取原文，解码器通过交叉注意力读取编码器输出。
3. **逐词生成**：推理只知道原文和已生成的前缀，从 BOS 起步，通过 `argmax` 一个 token 接一个 token 地生成，直到 EOS。
4. **训练循环**：`train/eval` 控制 dropout，`zero_grad → backward → step` 完成优化；验证关闭梯度，保存最佳权重。

最关键的是训练时目标句的右移。例如真实译文是 `[BOS, I, love, you, EOS]`，解码器输入为 `[BOS, I, love, you]`，标签为 `[I, love, you, EOS]`。因果遮罩禁止每个位置看到后续词，所以可以一次并行训练所有位置，又不会偷看答案。原文的编码器没有这个限制，可以双向读取整个原句。交叉熵直接吃 logits，代码不先做 softmax。

`nn.Transformer` 封装了多头注意力、前馈网络、残差连接和 LayerNorm；这份代码重点是理解整体 Transformer 和 PyTorch 训练流程，并未手写这些内部算子。之后可以尝试自己实现一个 EncoderLayer，替换封装层。

## 预期与取舍

3 万条、5 轮主要用来理解流程和观察学习；20 万条干净、领域一致的数据有机会学会常见短句，但泛化质量受数据影响很大，从零小模型不能期待通用翻译产品的效果。前几轮出现重复词、空译文、漏译并不罕见。请同时查看未见过的句子，别只凭训练句判断效果。

为便于阅读，这里采用 AdamW、可选的线性 warmup + cosine decay 和贪心解码，没有实现 beam search 或 BLEU。架构是标准 Encoder–Decoder Transformer，训练配方是简化学习版。程序同时保存最佳推理权重和最近一轮的完整训练断点。
