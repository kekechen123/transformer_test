"""训练并使用一个小型中英 Transformer 翻译模型。

这个文件包含一条完整的机器翻译流水线：

1. 读取中英平行语料；
2. 使用 SentencePiece 训练共享的 BPE 子词词表；
3. 将不同长度的句子补齐并组成 batch；
4. 使用 PyTorch ``nn.Transformer`` 训练编码器—解码器模型；
5. 使用贪心搜索或 Beam Search 逐 token 生成译文；
6. 保存最佳模型和可继续训练的断点。

本文统一使用以下张量形状记号：

- ``B``：batch size，一批样本的数量；
- ``S``：source length，源句 token 数；
- ``T``：target length，目标句 token 数；
- ``D``：d_model，每个 token 的隐藏向量维度；
- ``V``：vocab size，词表大小。
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader

# 模型不能直接计算文字，而是计算 token 对应的整数编号。
# 这四个编号同时传给 SentencePiece，因此分词器和模型对特殊 token 的约定一致。
PAD, UNK, BOS, EOS = 0, 1, 2, 3
# PAD（padding）          ：把同一 batch 内的短句补到相同长度。
# UNK（unknown）          ：表示词表无法覆盖的内容。
# BOS（begin of sentence）：表示目标句生成的起点。
# EOS（end of sentence）  ：表示一句话已经生成完毕。


# ============================================================================
# 1. 数据读取、分词与 batch 整理
# ============================================================================
def read_pairs(path, limit, reverse):
    """读取、检查、去重并随机打乱平行句对。

    数据文件每行必须是 ``中文<TAB>英文``。默认返回中译英句对；当
    ``reverse=True`` 时交换两侧，得到英译中句对。

    ``limit`` 在打乱后生效，所以表示随机抽样，而不是只取文件前几行。
    返回值形如 ``[(source_text, target_text), ...]``。
    """
    pairs = []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            # utf-8-sig 兼容普通 UTF-8，也会自动去掉文件开头可能存在的 BOM。
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) != 2 or not all(s.strip() for s in cols):
                raise ValueError(f"第 {line_no} 行应为两个非空句子，用一个 Tab 分隔")
            a, b = (s.strip() for s in cols)
            pairs.append((b, a) if reverse else (a, b))

    # dict 的键具有唯一性且保持插入顺序，可用来去除完全重复的句对。
    # 在划分训练集/验证集之前去重，也能避免同一句对同时出现在两边。
    pairs = list(dict.fromkeys(pairs))
    random.shuffle(pairs)
    return pairs[:limit] if limit else pairs


def encode(sp, sentence):
    """把字符串编码为 ``[BOS, 子词 id..., EOS]``。"""
    return [BOS] + sp.encode(sentence, out_type=int) + [EOS]


def collate(batch):
    """把一批不同长度的句对补齐为两个二维张量。

    ``DataLoader`` 每次收集若干 ``(src_ids, tgt_ids)`` 后调用本函数。
    例如两条源句长度分别为 3 和 5，补齐后源句张量形状为 ``[2, 5]``。

    返回：
        src：``[B, S]``，源句 token id；
        tgt：``[B, T]``，目标句 token id。
    """
    src, tgt = zip(*batch)
    return tuple(nn.utils.rnn.pad_sequence(
        [torch.tensor(ids) for ids in side], batch_first=True, padding_value=PAD
    ) for side in (src, tgt))


# ============================================================================
# 2. 模型：Embedding → 位置编码 → Encoder/Decoder → 词表 logits
# ============================================================================
class Translator(nn.Module):
    """基于 ``nn.Transformer`` 的编码器—解码器翻译模型。

    ``nn.Module`` 是 PyTorch 神经网络组件的基类。继承它以后，PyTorch 会自动
    追踪子模块和可训练参数，模型也会获得 ``to``、``train``、``eval``、
    ``state_dict`` 等统一接口。
    """

    def __init__(self, vocab_size, d_model=256, layers=3, max_len=96):
        super().__init__()

        # Embedding 是可学习的查找表：token id [B, L] → 向量 [B, L, D]。
        # padding_idx=PAD 使 PAD 对应的向量不接受常规梯度更新。
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            # 这是手动初始化，不属于模型学习过程，因此不需要记录计算图。
            self.embedding.weight[PAD].zero_()

        # Dropout 训练时随机将部分元素置零，以降低过拟合；eval 模式下自动关闭。
        self.dropout = nn.Dropout(0.1)

        # Transformer 本身只看 token 向量，不天然知道 token 的先后顺序。
        # 正弦位置编码为第 0、1、2... 个位置提供不同向量，使模型能够区分顺序。
        # pos: [max_len, 1]；freq: [D/2]；二者相乘时 PyTorch 自动广播。
        pos = torch.arange(max_len).unsqueeze(1)
        freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)

        # buffer 是模型状态的一部分，会随 model.to(device) 移动，也会被保存；
        # 但它不是 Parameter，优化器不会更新它。
        self.register_buffer("pe", pe)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=4, num_encoder_layers=layers,
            num_decoder_layers=layers, dim_feedforward=2 * d_model,
            dropout=0.1, batch_first=True,
        )

        # 将每个位置的 D 维隐藏向量映射为 V 个候选 token 的原始分数 logits。
        # 这里不做 softmax：训练时 cross_entropy 内部会处理；推理时按需处理。
        self.output = nn.Linear(d_model, vocab_size)

    def embed(self, ids):
        """将 token id ``[B, L]`` 转换为带位置信息的向量 ``[B, L, D]``。"""
        # 乘 sqrt(D) 是原始 Transformer 使用的缩放方式，可平衡词向量与位置编码。
        x = self.embedding(ids) * math.sqrt(self.embedding.embedding_dim)
        return self.dropout(x + self.pe[:ids.size(1)])

    def decode(self, tgt, memory, src_pad):
        """根据目标前缀和编码器输出，计算每个目标位置的词表 logits。

        参数形状：
            tgt：``[B, T]``，解码器已经看到的目标序列；
            memory：``[B, S, D]``，编码器对源句的表示；
            src_pad：``[B, S]``，源句 PAD 位置为 True。

        返回 ``[B, T, V]``。训练时 T 通常是完整目标句减去最后一个 token；
        推理时 T 从 1 开始，每生成一个 token 就增加 1。
        """
        # causal 是因果遮罩。True 的上三角位置禁止注意，确保第 t 个位置
        # 只能看见自己和之前的 token，不能偷看尚未生成的未来答案。
        causal = torch.triu(torch.ones(tgt.size(1), tgt.size(1),
                                      device=tgt.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer.decoder(
            self.embed(tgt), memory, tgt_mask=causal,
            tgt_key_padding_mask=tgt.eq(PAD), memory_key_padding_mask=src_pad,
        )
        return self.output(hidden)

    def forward(self, src, tgt):
        """执行一次完整前向传播：源句编码，再根据目标前缀解码。

        ``model(src, tgt)`` 会自动调用本方法。输入形状分别是 ``[B, S]`` 和
        ``[B, T]``，输出形状是 ``[B, T, V]``。
        """
        src_pad = src.eq(PAD)
        memory = self.transformer.encoder(self.embed(src), src_key_padding_mask=src_pad)
        return self.decode(tgt, memory, src_pad)


# ============================================================================
# 3. 推理：自回归生成、贪心搜索与 Beam Search
# ============================================================================
@torch.no_grad()
def translate(model, sp, sentence, device, decode="greedy", beam_size=2):
    """翻译单个句子。

    推理是“自回归”的：先给解码器 BOS，再把刚生成的 token 接回输入，继续
    预测下一个 token，直到生成 EOS 或达到 ``max_len``。

    ``greedy`` 每一步只保留当前概率最高的 token；``beam`` 同时保留若干条
    总体得分较高的候选序列，计算更多，但有机会得到更好的完整句子。
    """
    # eval() 切换模块行为（主要是关闭 Dropout）；@torch.no_grad() 关闭梯度记录。
    # 两者作用不同，推理时通常同时使用。
    model.eval()
    ids = encode(sp, sentence)
    max_len = model.pe.size(0)
    if len(ids) > max_len:
        raise ValueError(f"输入有 {len(ids)} 个 token，超过 max_len={max_len}")

    # 即使只翻译一句，模型仍要求 batch 维，因此外层再套一层列表：[1, S]。
    src = torch.tensor([ids], device=device)
    src_pad = src.eq(PAD)
    # 源句只需编码一次；后续每一步生成都复用同一份 memory。
    memory = model.transformer.encoder(model.embed(src), src_key_padding_mask=src_pad)

    def block_repeated_ngram(logits, token_ids, n=3):
        """禁止形成重复 n-gram，缓解短语反复循环的问题。

        例如 n=3、已有 ``A B C A B`` 时，当前前缀是 ``A B``，若下一步再次
        生成 C，就会重复三元组 ``A B C``，因此把 C 的 logit 设为负无穷。
        """
        if len(token_ids) < n - 1:
            return
        prefix = token_ids[-(n - 1):]
        blocked = {
            token_ids[i + n - 1]
            for i in range(len(token_ids) - n + 1)
            if token_ids[i:i + n - 1] == prefix
        }
        if blocked:
            logits[..., list(blocked)] = -float("inf")

    if decode == "greedy":
        # 初始目标前缀只有 BOS，形状为 [1, 1]。
        tgt = torch.tensor([[BOS]], device=device)
        for _ in range(max_len - 1):
            # decode 会返回所有目标位置的预测，但只有最后一个位置是“下一个词”。
            logits = model.decode(tgt, memory, src_pad)[:, -1, :]
            # -inf 经 argmax/softmax 后不可能被选中，因此可用于屏蔽非法候选。
            logits[:, [PAD, BOS]] = -float("inf")
            block_repeated_ngram(logits, tgt[0].tolist())
            # argmax 返回 V 个候选中 logit 最大者；keepdim 保留 [1, 1] 形状。
            next_id = logits.argmax(dim=-1, keepdim=True)
            if next_id.item() == EOS:
                break
            # dim=1 是句长维，将新 token 接到目标前缀末尾。
            tgt = torch.cat([tgt, next_id], dim=1)
        return sp.decode(tgt[0, 1:].tolist())

    if decode != "beam":
        raise ValueError(f"不支持的解码方式：{decode}")
    if beam_size < 1:
        raise ValueError("beam_size 必须大于等于 1")

    # 每条 beam 保存：(token id 序列, 累计对数概率, 是否生成 EOS)。
    def beam_score(item):
        """使用长度惩罚后的分数比较不同长度的候选译文。

        序列概率是每一步条件概率的乘积。取对数后，乘积变成加法，数值也更
        稳定；但 log probability 通常为负，序列越长累计值越低，因此需要长度
        归一化，避免搜索过度偏爱很短的译文。
        """
        token_ids, score, _ = item
        length_penalty = ((5 + max(1, len(token_ids) - 1)) / 6) ** 0.6
        return score / length_penalty

    beams = [([BOS], 0.0, False)]
    for _ in range(max_len - 1):
        candidates = []
        for token_ids, score, finished in beams:
            if finished:
                candidates.append((token_ids, score, True))
                continue

            tgt = torch.tensor([token_ids], device=device)
            logits = model.decode(tgt, memory, src_pad)[:, -1, :]
            logits[:, [PAD, BOS]] = -float("inf")
            block_repeated_ngram(logits, token_ids)
            # log_softmax 把 logits 转成对数概率；topk 取本轮最好的 beam_size 个词。
            log_probs = torch.log_softmax(logits, dim=-1)[0]
            top_scores, top_ids = torch.topk(log_probs, beam_size)
            for token_score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                candidates.append((
                    token_ids + [token_id],
                    score + token_score,
                    token_id == EOS,
                ))

        # 所有旧 beam 各自扩展出若干候选，再全局保留得分最高的 beam_size 条。
        candidates.sort(key=beam_score, reverse=True)
        beams = candidates[:beam_size]
        if all(finished for _, _, finished in beams):
            break

    token_ids, _, _ = max(beams, key=beam_score)
    if EOS in token_ids:
        token_ids = token_ids[1:token_ids.index(EOS)]
    else:
        token_ids = token_ids[1:]
    return sp.decode(token_ids)


# ============================================================================
# 4. 单个 epoch：Teacher Forcing、损失、反向传播与参数更新
# ============================================================================
def run_epoch(model, loader, optimizer, scaler, device, scheduler=None):
    """运行一轮训练或验证，并返回按非 PAD token 加权的平均 loss。

    ``optimizer`` 不为 None 时执行训练；为 None 时只验证。两种模式共享前向和
    loss 计算，区别在于训练模式会记录梯度、反向传播并更新参数。
    """
    training = optimizer is not None
    # 等价于 training 为 True 时 model.train()，否则 model.eval()。
    model.train(training)
    total_loss = total_tokens = 0
    for step, (src, tgt) in enumerate(loader, 1):
        # DataLoader 默认创建 CPU 张量；模型和输入必须位于同一个 device。
        src, tgt = src.to(device), tgt.to(device)

        # Teacher Forcing 的右移对齐：
        # 原目标：[BOS, I, love, you, EOS]
        # 输入：  [BOS, I, love, you]
        # 标签：  [I,   love, you,  EOS]
        # 一次前向即可并行学习每个位置的“根据真实前缀预测下一个 token”。
        decoder_input, labels = tgt[:, :-1], tgt[:, 1:]

        if training:
            # PyTorch 默认把新梯度累加到旧梯度上，所以每个 batch 前必须清空。
            # set_to_none=True 通常比填充为 0 更省内存。
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            # autocast 在 CUDA 上自动为合适的运算选择较低精度，通常更快、更省显存。
            # CPU 路径关闭，因此仍按普通精度运行。
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(src, decoder_input)
                # 每个目标位置都是一次 V 分类。将 [B, T, V] 展平为 [B*T, V]，
                # labels 从 [B, T] 展平为 [B*T]，即可交给 cross_entropy。
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                    ignore_index=PAD,
                )
            if training:
                # backward 沿计算图反向计算每个参数对 loss 的梯度。
                # GradScaler 放大 loss，缓解 FP16 中很小的梯度下溢为 0 的问题。
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # 裁剪前先 unscale，确保这里检查的是梯度的真实尺度。
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    # 当前学习率调度按 batch/step 更新，而不是按 epoch 更新。
                    scheduler.step()

        # 各 batch 的真实 token 数不同，因此按非 PAD token 数加权平均 loss。
        tokens = labels.ne(PAD).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        if training and step % 100 == 0:
            print(f"  step {step}/{len(loader)} | loss {total_loss / total_tokens:.3f}", flush=True)
    return total_loss / total_tokens


def main():
    """解析命令行参数，并根据 ``--text`` 选择推理流程或训练流程。"""

    # ------------------------------------------------------------------
    # 5.1 命令行参数
    # ------------------------------------------------------------------
    # argparse 将命令行字符串转换为带属性的 args 对象。
    # 例如 ``python translate.py --epochs 10`` 会得到 args.epochs == 10。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="parallel.tsv", help="无表头的中文 TAB 英文文件")
    p.add_argument("--out", default="runs/zh_en", help="模型和词表的保存目录")
    p.add_argument("--text", help="提供句子时加载模型进行翻译，不训练")
    p.add_argument("--decode", choices=("greedy", "beam"), default="greedy",
                   help="推理解码方式（默认 greedy）")
    p.add_argument("--beam-size", type=int, default=2,
                   help="beam search 保留的候选数量（默认 2）")
    p.add_argument("--resume", action="store_true", help="从 --out/last.pt 继续训练")
    p.add_argument("--reverse", action="store_true", help="训练英译中（默认中译英）")
    p.add_argument("--limit", type=int, default=30000, help="随机抽样条数；0 为全量")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=96, help="含 BOS/EOS 的最大 token 数")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=3, help="编码器、解码器各自的层数")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-warmup", action="store_true",
                   help="启用按 step 线性 warmup，达到 --lr 后再余弦衰减")
    p.add_argument("--warmup-ratio", type=float, default=0.1,
                   help="warmup 占总训练 step 的比例（默认 0.1）")
    args = p.parse_args()

    # 固定随机种子，使数据打乱和参数初始化尽可能可复现。
    random.seed(42)
    torch.manual_seed(42)

    # 如果机器有可用 NVIDIA GPU，就使用 CUDA；否则使用 CPU。
    # 模型和输入张量必须放在同一个 device 上，否则 PyTorch 会报错。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    print(f"设备：{device}", flush=True)

    # ------------------------------------------------------------------
    # 5.2 纯推理流程：加载 best.pt 和与它配套的 tokenizer
    # ------------------------------------------------------------------
    if args.text is not None:
        ckpt = torch.load(out / "best.pt", map_location=device, weights_only=True)
        sp = spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))
        # checkpoint 中的 config 保存了构建模型所需的结构参数。
        # 必须先创建相同结构，再用 load_state_dict 填入训练好的权重。
        model = Translator(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"方向：{ckpt['direction']}")
        if args.beam_size < 1:
            p.error("--beam-size 必须大于等于 1")
        print(translate(model, sp, args.text, device, args.decode, args.beam_size))
        return

    # ------------------------------------------------------------------
    # 5.3 训练前检查、读取数据和训练 SentencePiece
    # ------------------------------------------------------------------
    if (args.d_model < 4 or args.d_model % 4 or args.layers < 1 or args.max_len < 4
            or args.batch_size < 1 or args.epochs < 1 or args.limit < 0 or args.lr <= 0
            or not 0 < args.warmup_ratio < 1):
        p.error("d-model 须为 4 的正倍数；layers/batch-size/epochs/lr > 0；"
                "max-len ≥ 4；limit ≥ 0；warmup-ratio 须在 0 和 1 之间")
    # 每次实验使用独立目录。模型权重与 tokenizer 必须配套，否则相同 token id
    # 可能被解释成不同子词，模型即使能加载也无法正常翻译。
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not (out / "last.pt").is_file():
            p.error(f"找不到断点文件：{out / 'last.pt'}")
        if not (out / "tokenizer.model").is_file():
            p.error(f"找不到断点配套词表：{out / 'tokenizer.model'}")
    elif any(out.iterdir()):
        p.error("输出目录非空，请用 --out 指定一个新目录，或用 --resume 继续训练")
    pairs = read_pairs(args.data, args.limit, args.reverse)
    if len(pairs) < 20:
        p.error("至少需要 20 条去重后的句对；学习训练建议数万条")

    # 最多使用 1000 条作为验证集；数据较少时大约取总量的 5%。
    n_valid = min(1000, max(1, len(pairs) // 20))
    valid_pairs, train_pairs = pairs[:n_valid], pairs[n_valid:]

    # 先划分验证集，再仅用训练文本学习 BPE，避免验证文本参与词表训练。
    if not args.resume:
        spm.SentencePieceTrainer.train(
            sentence_iterator=(s for pair in train_pairs for s in pair),
            model_prefix=str(out / "tokenizer"), model_type="bpe",
            vocab_size=args.vocab_size, character_coverage=0.9995,
            pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
            hard_vocab_limit=False, minloglevel=2,
        )
    sp = spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))
    cpu_test_sentences = [line.strip() for line in Path(
        "./test_data/test_data/test.md"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cpu_test_sentences:
        p.error("CPU 测试文本为空")

    def prepare(raw):
        """编码句对，并丢弃任一侧超过 max_len 的样本。"""
        encoded = [(encode(sp, a), encode(sp, b)) for a, b in raw]
        # 这里选择整条丢弃而不是截断，避免源句与目标句语义变得不完整。
        return [(a, b) for a, b in encoded if max(len(a), len(b)) <= args.max_len]

    train_data, valid_data = prepare(train_pairs), prepare(valid_pairs)
    if not train_data or not valid_data:
        p.error("长度过滤后训练集或验证集为空，请增大 --max-len，并指定新的 --out")
    # DataLoader 管理分批和打乱；collate_fn 负责将变长序列补齐为张量。
    # 训练集每轮打乱可减少样本顺序偏差，验证集不更新参数，因此无需打乱。
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, collate_fn=collate)

    # ------------------------------------------------------------------
    # 5.4 创建模型、优化器、学习率调度器和混合精度工具
    # ------------------------------------------------------------------
    config = dict(vocab_size=sp.get_piece_size(), d_model=args.d_model,
                  layers=args.layers, max_len=args.max_len)
    # **config 将字典展开为关键字参数；to(device) 会移动参数和 buffer。
    model = Translator(**config).to(device)

    # 优化器保存参数引用及其更新状态。AdamW 使用自适应学习率，并将
    # weight decay 与梯度更新解耦，是 Transformer 中常见的选择。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = None
    if args.lr_warmup:
        total_steps = args.epochs * len(train_loader)
        warmup_steps = max(1, round(total_steps * args.warmup_ratio))

        def lr_scale(step):
            """返回基础学习率的倍率：先线性 warmup，再 cosine 衰减到 0。"""
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, (step - warmup_steps) / decay_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        print(f"学习率：warmup {warmup_steps}/{total_steps} steps "
              f"({args.warmup_ratio:.0%}) → 峰值 {args.lr:g} → cosine decay", flush=True)

    # GradScaler 只在 CUDA 混合精度下启用；CPU 上对象仍可调用，但不进行缩放。
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    direction = "英 → 中" if args.reverse else "中 → 英"
    training_config = {
        "data": str(Path(args.data).resolve()),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_warmup": args.lr_warmup,
        "warmup_ratio": args.warmup_ratio,
    }
    start_epoch = 1
    best = float("inf")

    # ------------------------------------------------------------------
    # 5.5 可选的断点恢复
    # ------------------------------------------------------------------
    if args.resume:
        ckpt = torch.load(out / "last.pt", map_location=device, weights_only=True)
        # 续训必须恢复的不只是模型权重，还包括优化器、混合精度和学习率状态。
        # 否则虽然参数接上了，训练轨迹却会突然改变。
        if ckpt["config"] != config:
            p.error(f"断点模型配置 {ckpt['config']} 与当前参数 {config} 不一致")
        if ckpt["direction"] != direction:
            p.error(f"断点方向为 {ckpt['direction']}，与当前方向 {direction} 不一致")
        if ckpt.get("training_config") != training_config:
            p.error("断点与当前 data/limit/batch-size/epochs/lr/warmup 参数不一致")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        saved_scheduler = ckpt.get("scheduler")
        if (scheduler is None) != (saved_scheduler is None):
            p.error("断点与当前 --lr-warmup 设置不一致")
        if scheduler is not None:
            scheduler.load_state_dict(saved_scheduler)
        best = ckpt["best"]
        start_epoch = ckpt["epoch"] + 1
        if start_epoch > args.epochs:
            p.error(f"断点已完成 {ckpt['epoch']} 轮，--epochs 必须更大")
        print(f"已恢复断点：完成 {ckpt['epoch']} 轮，将从第 {start_epoch} 轮继续", flush=True)

    print(f"{direction} | 训练 {len(train_data)} / 验证 {len(valid_data)} 句对 | "
          f"过滤 {len(pairs) - len(train_data) - len(valid_data)} 条超长句对 | "
          f"参数 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    loss_path = out / "loss.csv"
    write_header = not args.resume or not loss_path.exists() or loss_path.stat().st_size == 0

    # ------------------------------------------------------------------
    # 5.6 epoch 循环、日志与模型保存
    # ------------------------------------------------------------------
    with open(loss_path, "a" if args.resume else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "valid_loss", "seconds"])
        for epoch in range(start_epoch, args.epochs + 1):
            start = time.perf_counter()
            # 一个 epoch 表示模型完整遍历一次训练集。
            train_loss = run_epoch(model, train_loader, optimizer, scaler, device, scheduler)
            # optimizer=None：切换到验证模式，不建立梯度，也不更新参数。
            valid_loss = run_epoch(model, valid_loader, None, scaler, device)
            seconds = time.perf_counter() - start
            writer.writerow([epoch, train_loss, valid_loss, round(seconds, 1)])
            f.flush()
            print(f"Epoch {epoch}/{args.epochs} | train {train_loss:.3f} | "
                  f"valid {valid_loss:.3f} | {seconds:.1f}s", flush=True)
            if valid_loss < best:
                # best.pt 面向最终推理，只保存最佳权重和重建模型所需的信息。
                best = valid_loss
                torch.save({"model": model.state_dict(), "config": config,
                            "direction": direction}, out / "best.pt")

            # last.pt 面向续训，需保存最近 epoch 的全部训练状态。
            # 先写临时文件，再用同目录原子替换，避免保存中断损坏旧断点。
            last_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best": best,
                "config": config,
                "direction": direction,
                "training_config": training_config,
            }
            last_tmp = out / "last.pt.tmp"
            torch.save(last_state, last_tmp)
            last_tmp.replace(out / "last.pt")

            # 每 5 轮临时把模型移到 CPU，对固定句子做一次直观质量检查。
            # finally 保证即使测试翻译报错，模型也会被移回原训练设备。
            if epoch % 5 == 0:
                cpu = torch.device("cpu")
                model.to(cpu)
                try:
                    for source in cpu_test_sentences:
                        infer_start = time.perf_counter()
                        prediction = translate(model, sp, source, cpu)
                        infer_seconds = time.perf_counter() - infer_start
                        print(f"  CPU 推理 {infer_seconds * 1000:.1f}ms\n  原文：{source}\n"
                              f"  预测：{prediction}", flush=True)
                finally:
                    model.to(device)
    print(f"完成！最佳验证 loss={best:.3f}，模型保存在 {out / 'best.pt'}")


if __name__ == "__main__":
    main()
