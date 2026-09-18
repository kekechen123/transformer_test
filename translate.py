"""从零训练一个小型中英翻译 Transformer。先读 README_translation.md。"""
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

PAD, UNK, BOS, EOS = 0, 1, 2, 3
# 这四个整数是特殊 token（特殊标记）的编号：
# PAD：padding，补齐句子长度时使用；模型应当忽略它。
# UNK：unknown，词表中没有收录的内容会用它表示。
# BOS：beginning of sentence，句子开始。
# EOS：end of sentence，句子结束。
#
# 模型实际处理的不是汉字或英文单词本身，而是这些 token 对应的整数编号。


# 1. 数据：中文、英文共用一套子词词表，避免英文按字母切分、中文词表过大。
def read_pairs(path, limit, reverse):
    # 读取形如“中文<TAB>英文”的平行语料。
    # 一行就是一个训练样本：前一句是输入（source），后一句是目标（target）。
    pairs = []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            # utf-8-sig 也能自动处理某些文件开头附带的 BOM 标记。
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) != 2 or not all(s.strip() for s in cols):
                raise ValueError(f"第 {line_no} 行应为两个非空句子，用一个 Tab 分隔")
            a, b = (s.strip() for s in cols)
            # reverse=True 时交换方向，用同一份数据训练英译中。
            pairs.append((b, a) if reverse else (a, b))
    # dict 保持原顺序，同时利用键的唯一性去掉完全重复的句对。
    # 去重可以减少重复样本，也避免同一个句对同时落入训练集和验证集。
    pairs = list(dict.fromkeys(pairs))  # 完全重复的句对去重，减少训练/验证泄漏。
    # 打乱后再切 limit，表示随机抽取一部分数据，而不是总取文件开头。
    random.shuffle(pairs)
    return pairs[:limit] if limit else pairs


def encode(sp, sentence):
    # SentencePiece 把原始字符串切成子词，并返回每个子词的整数 id。
    # 在首尾加 BOS/EOS 后，模型知道“从哪里开始生成”和“什么时候结束”。
    return [BOS] + sp.encode(sentence, out_type=int) + [EOS]


def collate(batch):
    # 不同句长补 PAD 到当前 batch 的最长长度；返回 [batch, length]。
    # DataLoader 每次取出若干样本后，会调用这个函数把它们拼成一个 batch。
    # 例如长度分别为 3、5 的两句话，会变成形状 [2, 5] 的二维张量。
    src, tgt = zip(*batch)
    return tuple(nn.utils.rnn.pad_sequence(
        [torch.tensor(ids) for ids in side], batch_first=True, padding_value=PAD
    ) for side in (src, tgt))


# 2. 模型：Embedding → 正弦位置编码 → Encoder/Decoder → 词表概率。
class Translator(nn.Module):
    # nn.Module 是 PyTorch 中所有神经网络模型的基类。
    # 继承它后，模型参数会被自动收集，能够使用 .to(device)、state_dict() 等功能。
    def __init__(self, vocab_size, d_model=256, layers=3, max_len=96):
        super().__init__()
        # Embedding 是一个可学习的查表：token id -> d_model 维向量。
        # 输入形状通常是 [batch, length]，输出是 [batch, length, d_model]。
        # padding_idx=PAD 让 PAD 的向量不参与正常的梯度更新。
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            # 明确把 PAD 的向量设为全 0；no_grad 表示这次修改不建立计算图。
            self.embedding.weight[PAD].zero_()
        # 训练时随机丢弃一小部分数值，用于减少过拟合；推理时自动关闭。
        self.dropout = nn.Dropout(0.1)
        # Transformer 本身只看 token 向量，不天然知道 token 的先后顺序。
        # 位置编码 pe 为第 0、1、2... 个位置提供不同的向量，让模型能分辨顺序。
        # pos: [length, 1]，freq: [d_model/2]，广播后得到各位置/频率的角度。
        pos = torch.arange(max_len).unsqueeze(1)
        freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        self.register_buffer("pe", pe)  # 随模型搬到 GPU、保存，但不参与梯度更新。
        # register_buffer 与普通属性不同：它会随 model.to(device) 一起移动，
        # 也会进入模型保存内容，但不是需要优化器学习的 parameter。
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=4, num_encoder_layers=layers,
            num_decoder_layers=layers, dim_feedforward=2 * d_model,
            dropout=0.1, batch_first=True,
        )
        # 把每个位置的隐藏向量转换成词表大小的分数。
        # 输出的每个分数叫 logit，分数越大代表模型越倾向于该 token。
        # 这里故意不做 softmax，因为交叉熵损失函数内部会完成相关计算。
        self.output = nn.Linear(d_model, vocab_size)

    def embed(self, ids):
        # 先查词向量，再乘 sqrt(d_model) 是 Transformer 的常见缩放方式，
        # 最后加上对应长度的正弦位置编码。
        x = self.embedding(ids) * math.sqrt(self.embedding.embedding_dim)
        return self.dropout(x + self.pe[:ids.size(1)])

    def decode(self, tgt, memory, src_pad):
        # True 表示禁止关注：上三角遮住未来词；PAD 遮罩忽略补齐部分。
        # tgt 是解码器已经看到的目标前缀；memory 是编码器读完源句后的结果。
        # 训练时 tgt 可能包含整批目标句子的前缀，推理时则逐步变长。
        causal = torch.triu(torch.ones(tgt.size(1), tgt.size(1),
                                      device=tgt.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer.decoder(
            self.embed(tgt), memory, tgt_mask=causal,
            tgt_key_padding_mask=tgt.eq(PAD), memory_key_padding_mask=src_pad,
        )
        # hidden 的形状是 [batch, target_length, d_model]；
        # 线性层把最后一维映射成 [batch, target_length, vocab_size]。
        return self.output(hidden)  # [batch, target_length, vocab_size]，未做 softmax。

    def forward(self, src, tgt):
        # 编码器先理解源句，得到 memory；解码器再结合 memory 生成目标句。
        # src_pad 的形状为 [batch, source_length]，True 的位置是补齐的 PAD。
        src_pad = src.eq(PAD)
        memory = self.transformer.encoder(self.embed(src), src_key_padding_mask=src_pad)
        return self.decode(tgt, memory, src_pad)


# 3. 推理：从 BOS 开始，每次预测下一个词，遇到 EOS 停止。
@torch.no_grad()
def translate(model, sp, sentence, device):
    model.eval()  # 关闭 dropout；no_grad 则关闭梯度记录，两者用途不同。
    # 推理阶段只需要前向计算，不需要反向传播和更新参数，因此不保存梯度。
    ids = encode(sp, sentence)
    max_len = model.pe.size(0)
    if len(ids) > max_len:
        raise ValueError(f"输入有 {len(ids)} 个 token，超过 max_len={max_len}")
    # 外层列表表示 batch size=1；即使只翻译一句，也要保留 batch 这一维。
    src = torch.tensor([ids], device=device)
    src_pad = src.eq(PAD)
    memory = model.transformer.encoder(model.embed(src), src_key_padding_mask=src_pad)
    # 目标句目前只有 BOS，后面每轮把模型新预测出的 token 接到末尾。
    tgt = torch.tensor([[BOS]], device=device)
    for _ in range(max_len - 1):
        # 只取最后一个位置的预测，因为前面位置已经生成过了。
        logits = model.decode(tgt, memory, src_pad)[:, -1, :]
        # 生成过程中不希望再次生成 PAD 或 BOS，所以把它们的分数设为负无穷。
        logits[:, [PAD, BOS]] = -float("inf")
        next_id = logits.argmax(dim=-1, keepdim=True)  # 贪心解码，便于理解。
        if next_id.item() == EOS:
            break
        # 沿句子长度这一维拼接新 token；下一轮会把更长的 tgt 再交给解码器。
        tgt = torch.cat([tgt, next_id], dim=1)
    # 去掉开头的 BOS，再把 token id 还原为可读字符串。
    return sp.decode(tgt[0, 1:].tolist())


# 4. 一轮训练/验证：Teacher Forcing，用真实前缀预测下一个真实词。
def run_epoch(model, loader, optimizer, scaler, device, scheduler=None):
    # optimizer 不为空表示训练；传 None 表示验证。
    # 训练和验证使用同一套前向/损失计算，但验证不反向传播、不更新参数。
    training = optimizer is not None
    model.train(training)
    total_loss = total_tokens = 0
    for step, (src, tgt) in enumerate(loader, 1):
        # 把 CPU 上的 batch 搬到 CPU/GPU 中实际运行模型的设备。
        src, tgt = src.to(device), tgt.to(device)
        # 例如 [BOS, I, love, you, EOS]：输入去尾，标签去头（右移对齐）。
        decoder_input, labels = tgt[:, :-1], tgt[:, 1:]
        # Teacher Forcing：每个位置都用真实的历史前缀作为输入。
        # 例如目标为 [BOS, I, love, EOS]：输入是 [BOS, I, love]，标签是 [I, love, EOS]。
        # 所以模型在位置 i 学的是“根据前面的词预测下一个词”。
        if training:
            # 清除上一批次留下的梯度；PyTorch 默认会累加梯度。
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            # CUDA 上用 FP16 混合精度节省显存；CPU 上仍用 FP32。
            # autocast 在 CUDA 上使用混合精度，通常可以节省显存并加速；CPU 上关闭。
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(src, decoder_input)
                # cross_entropy 接收 logits 和正确答案 labels，衡量预测有多不准确。
                # reshape 把 batch 和时间位置合并，变成“很多个分类问题”。
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                    ignore_index=PAD,  # PAD 不算进损失。
                )
            if training:
                # backward 根据 loss 计算每个参数的梯度；scaler 用于混合精度训练的数值稳定。
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # 梯度裁剪，防止某次更新的梯度过大导致训练发散。
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 根据梯度更新参数，然后更新混合精度的缩放因子。
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    # 按 batch（step）更新学习率，而不是等到 epoch 结束。
                    scheduler.step()
        # 只统计真实 token，不统计补齐用的 PAD；loss 也使用了相同规则。
        tokens = labels.ne(PAD).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        if training and step % 100 == 0:
            print(f"  step {step}/{len(loader)} | loss {total_loss / total_tokens:.3f}", flush=True)
    return total_loss / total_tokens


def main():
    # argparse 负责读取命令行参数，例如 --epochs 10 或 --text "你好"。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="parallel.tsv", help="无表头的中文 TAB 英文文件")
    p.add_argument("--out", default="runs/zh_en", help="模型和词表的保存目录")
    p.add_argument("--text", help="提供句子时加载模型进行翻译，不训练")
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
    random.seed(42)
    torch.manual_seed(42)
    # 如果机器有可用 NVIDIA GPU，就使用 CUDA；否则使用 CPU。
    # 模型和输入张量必须放在同一个 device 上，否则 PyTorch 会报错。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    print(f"设备：{device}", flush=True)

    if args.text is not None:
        # 有 --text 时走纯推理流程：读取之前保存的最佳模型，不重新训练。
        ckpt = torch.load(out / "best.pt", map_location=device, weights_only=True)
        sp = spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))
        model = Translator(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"方向：{ckpt['direction']}")
        print(translate(model, sp, args.text, device))
        return

    if (args.d_model < 4 or args.d_model % 4 or args.layers < 1 or args.max_len < 4
            or args.batch_size < 1 or args.epochs < 1 or args.limit < 0 or args.lr <= 0
            or not 0 < args.warmup_ratio < 1):
        p.error("d-model 须为 4 的正倍数；layers/batch-size/epochs/lr > 0；"
                "max-len ≥ 4；limit ≥ 0；warmup-ratio 须在 0 和 1 之间")
    # 每次实验单独保存，避免新词表覆盖旧模型所依赖的词表。
    # Transformer 的参数和 tokenizer 必须配套，否则同一个 id 可能代表不同子词。
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error("输出目录非空，请用 --out 指定一个新目录")
    pairs = read_pairs(args.data, args.limit, args.reverse)
    if len(pairs) < 20:
        p.error("至少需要 20 条去重后的句对；学习训练建议数万条")
    n_valid = min(1000, max(1, len(pairs) // 20))
    # 验证集只用来检查泛化效果，不参与参数更新。
    valid_pairs, train_pairs = pairs[:n_valid], pairs[n_valid:]
    # 先划分验证集，再仅用训练文本学习 BPE，避免验证文本参与词表训练。
    spm.SentencePieceTrainer.train(
        sentence_iterator=(s for pair in train_pairs for s in pair),
        model_prefix=str(out / "tokenizer"), model_type="bpe",
        vocab_size=args.vocab_size, character_coverage=0.9995,
        pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        hard_vocab_limit=False, minloglevel=2,
    )
    sp = spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))

    def prepare(raw):
        # 把字符串句对转换成 token id 句对，供 DataLoader 和模型使用。
        encoded = [(encode(sp, a), encode(sp, b)) for a, b in raw]
        # 丢掉超长句对，避免截断后留下语义不对应的训练数据。
        return [(a, b) for a, b in encoded if max(len(a), len(b)) <= args.max_len]

    train_data, valid_data = prepare(train_pairs), prepare(valid_pairs)
    if not train_data or not valid_data:
        p.error("长度过滤后训练集或验证集为空，请增大 --max-len，并指定新的 --out")
    # DataLoader 负责按 batch 取数据；训练集打乱顺序，验证集不需要打乱。
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, collate_fn=collate)
    config = dict(vocab_size=sp.get_piece_size(), d_model=args.d_model,
                  layers=args.layers, max_len=args.max_len)
    # **config 把字典中的键值作为关键字参数传给 Translator。
    # .to(device) 会把模型的可学习参数和 buffer 一起搬到目标设备。
    model = Translator(**config).to(device)
    # AdamW 是常用的优化器：它根据梯度调整模型参数，使 loss 逐步下降。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = None
    if args.lr_warmup:
        total_steps = args.epochs * len(train_loader)
        warmup_steps = max(1, round(total_steps * args.warmup_ratio))

        def lr_scale(step):
            # 先线性升到 --lr，再用 cosine 平滑降到 0。
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, (step - warmup_steps) / decay_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        print(f"学习率：warmup {warmup_steps}/{total_steps} steps "
              f"({args.warmup_ratio:.0%}) → 峰值 {args.lr:g} → cosine decay", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    direction = "英 → 中" if args.reverse else "中 → 英"
    print(f"{direction} | 训练 {len(train_data)} / 验证 {len(valid_data)} 句对 | "
          f"过滤 {len(pairs) - len(train_data) - len(valid_data)} 条超长句对 | "
          f"参数 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    best = float("inf")
    with open(out / "loss.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss", "seconds"])
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            # 一轮（epoch）就是把整个训练集完整看一遍。
            train_loss = run_epoch(model, train_loader, optimizer, scaler, device, scheduler)
            # optimizer=None，因此这一轮只评估，不更新模型。
            valid_loss = run_epoch(model, valid_loader, None, scaler, device)
            seconds = time.perf_counter() - start
            writer.writerow([epoch, train_loss, valid_loss, round(seconds, 1)])
            f.flush()
            print(f"Epoch {epoch}/{args.epochs} | train {train_loss:.3f} | "
                  f"valid {valid_loss:.3f} | {seconds:.1f}s", flush=True)
            if valid_loss < best:
                # 保存验证集 loss 最低的版本，而不是盲目保存最后一轮。
                # state_dict 是“参数名 -> 参数张量”的字典，也是 PyTorch 常用保存方式。
                best = valid_loss
                torch.save({"model": model.state_dict(), "config": config,
                            "direction": direction}, out / "best.pt")
            # 每轮只抽一条，并明确在 CPU 上推理，顺便观察单条翻译耗时。
            src_ids, tgt_ids = valid_data[0]
            source, reference = sp.decode(src_ids[1:-1]), sp.decode(tgt_ids[1:-1])
            cpu = torch.device("cpu")
            model.to(cpu)
            try:
                infer_start = time.perf_counter()
                prediction = translate(model, sp, source, cpu)
                infer_seconds = time.perf_counter() - infer_start
            finally:
                model.to(device)
            print(f"  CPU 推理 {infer_seconds * 1000:.1f}ms\n  原文：{source}\n"
                  f"  参考：{reference}\n  预测：{prediction}", flush=True)
    print(f"完成！最佳验证 loss={best:.3f}，模型保存在 {out / 'best.pt'}")


if __name__ == "__main__":
    main()
