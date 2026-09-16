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


# 1. 数据：中文、英文共用一套子词词表，避免英文按字母切分、中文词表过大。
def read_pairs(path, limit, reverse):
    pairs = []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) != 2 or not all(s.strip() for s in cols):
                raise ValueError(f"第 {line_no} 行应为两个非空句子，用一个 Tab 分隔")
            a, b = (s.strip() for s in cols)
            pairs.append((b, a) if reverse else (a, b))
    pairs = list(dict.fromkeys(pairs))  # 完全重复的句对去重，减少训练/验证泄漏。
    random.shuffle(pairs)
    return pairs[:limit] if limit else pairs


def encode(sp, sentence):
    return [BOS] + sp.encode(sentence, out_type=int) + [EOS]


def collate(batch):
    # 不同句长补 PAD 到当前 batch 的最长长度；返回 [batch, length]。
    src, tgt = zip(*batch)
    return tuple(nn.utils.rnn.pad_sequence(
        [torch.tensor(ids) for ids in side], batch_first=True, padding_value=PAD
    ) for side in (src, tgt))


# 2. 模型：Embedding → 正弦位置编码 → Encoder/Decoder → 词表概率。
class Translator(nn.Module):
    def __init__(self, vocab_size, d_model=256, layers=3, max_len=96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()
        self.dropout = nn.Dropout(0.1)
        # pos: [length, 1]，freq: [d_model/2]，广播后得到各位置/频率的角度。
        pos = torch.arange(max_len).unsqueeze(1)
        freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        self.register_buffer("pe", pe)  # 随模型搬到 GPU、保存，但不参与梯度更新。
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=4, num_encoder_layers=layers,
            num_decoder_layers=layers, dim_feedforward=2 * d_model,
            dropout=0.1, batch_first=True,
        )
        self.output = nn.Linear(d_model, vocab_size)

    def embed(self, ids):
        x = self.embedding(ids) * math.sqrt(self.embedding.embedding_dim)
        return self.dropout(x + self.pe[:ids.size(1)])

    def decode(self, tgt, memory, src_pad):
        # True 表示禁止关注：上三角遮住未来词；PAD 遮罩忽略补齐部分。
        causal = torch.triu(torch.ones(tgt.size(1), tgt.size(1),
                                      device=tgt.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer.decoder(
            self.embed(tgt), memory, tgt_mask=causal,
            tgt_key_padding_mask=tgt.eq(PAD), memory_key_padding_mask=src_pad,
        )
        return self.output(hidden)  # [batch, target_length, vocab_size]，未做 softmax。

    def forward(self, src, tgt):
        src_pad = src.eq(PAD)
        memory = self.transformer.encoder(self.embed(src), src_key_padding_mask=src_pad)
        return self.decode(tgt, memory, src_pad)


# 3. 推理：从 BOS 开始，每次预测下一个词，遇到 EOS 停止。
@torch.no_grad()
def translate(model, sp, sentence, device):
    model.eval()  # 关闭 dropout；no_grad 则关闭梯度记录，两者用途不同。
    ids = encode(sp, sentence)
    max_len = model.pe.size(0)
    if len(ids) > max_len:
        raise ValueError(f"输入有 {len(ids)} 个 token，超过 max_len={max_len}")
    src = torch.tensor([ids], device=device)
    src_pad = src.eq(PAD)
    memory = model.transformer.encoder(model.embed(src), src_key_padding_mask=src_pad)
    tgt = torch.tensor([[BOS]], device=device)
    for _ in range(max_len - 1):
        logits = model.decode(tgt, memory, src_pad)[:, -1, :]
        logits[:, [PAD, BOS]] = -float("inf")
        next_id = logits.argmax(dim=-1, keepdim=True)  # 贪心解码，便于理解。
        if next_id.item() == EOS:
            break
        tgt = torch.cat([tgt, next_id], dim=1)
    return sp.decode(tgt[0, 1:].tolist())


# 4. 一轮训练/验证：Teacher Forcing，用真实前缀预测下一个真实词。
def run_epoch(model, loader, optimizer, scaler, device):
    training = optimizer is not None
    model.train(training)
    total_loss = total_tokens = 0
    for step, (src, tgt) in enumerate(loader, 1):
        src, tgt = src.to(device), tgt.to(device)
        # 例如 [BOS, I, love, you, EOS]：输入去尾，标签去头（右移对齐）。
        decoder_input, labels = tgt[:, :-1], tgt[:, 1:]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            # CUDA 上用 FP16 混合精度节省显存；CPU 上仍用 FP32。
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(src, decoder_input)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                    ignore_index=PAD,  # PAD 不算进损失。
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        tokens = labels.ne(PAD).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        if training and step % 100 == 0:
            print(f"  step {step}/{len(loader)} | loss {total_loss / total_tokens:.3f}", flush=True)
    return total_loss / total_tokens


def main():
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
    args = p.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    print(f"设备：{device}", flush=True)

    if args.text is not None:
        ckpt = torch.load(out / "best.pt", map_location=device, weights_only=True)
        sp = spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))
        model = Translator(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"方向：{ckpt['direction']}")
        print(translate(model, sp, args.text, device))
        return

    if (args.d_model < 4 or args.d_model % 4 or args.layers < 1 or args.max_len < 4
            or args.batch_size < 1 or args.epochs < 1 or args.limit < 0 or args.lr <= 0):
        p.error("d-model 须为 4 的正倍数；layers/batch-size/epochs/lr > 0；max-len ≥ 4；limit ≥ 0")
    # 每次实验单独保存，避免新词表覆盖旧模型所依赖的词表。
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error("输出目录非空，请用 --out 指定一个新目录")
    pairs = read_pairs(args.data, args.limit, args.reverse)
    if len(pairs) < 20:
        p.error("至少需要 20 条去重后的句对；学习训练建议数万条")
    n_valid = min(1000, max(1, len(pairs) // 20))
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
        encoded = [(encode(sp, a), encode(sp, b)) for a, b in raw]
        # 丢掉超长句对，避免截断后留下语义不对应的训练数据。
        return [(a, b) for a, b in encoded if max(len(a), len(b)) <= args.max_len]

    train_data, valid_data = prepare(train_pairs), prepare(valid_pairs)
    if not train_data or not valid_data:
        p.error("长度过滤后训练集或验证集为空，请增大 --max-len，并指定新的 --out")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, collate_fn=collate)
    config = dict(vocab_size=sp.get_piece_size(), d_model=args.d_model,
                  layers=args.layers, max_len=args.max_len)
    model = Translator(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
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
            train_loss = run_epoch(model, train_loader, optimizer, scaler, device)
            valid_loss = run_epoch(model, valid_loader, None, scaler, device)
            seconds = time.perf_counter() - start
            writer.writerow([epoch, train_loss, valid_loss, round(seconds, 1)])
            f.flush()
            print(f"Epoch {epoch}/{args.epochs} | train {train_loss:.3f} | "
                  f"valid {valid_loss:.3f} | {seconds:.1f}s", flush=True)
            if valid_loss < best:
                best = valid_loss
                torch.save({"model": model.state_dict(), "config": config,
                            "direction": direction}, out / "best.pt")
            for src_ids, tgt_ids in valid_data[:3]:
                source, reference = sp.decode(src_ids[1:-1]), sp.decode(tgt_ids[1:-1])
                print(f"  原文：{source}\n  参考：{reference}\n"
                      f"  预测：{translate(model, sp, source, device)}", flush=True)
    print(f"完成！最佳验证 loss={best:.3f}，模型保存在 {out / 'best.pt'}")


if __name__ == "__main__":
    main()
