# 用 PyTorch 学一个小型翻译 Transformer

本人自己的transformer学习项目，transformer_test/test_results2.md 是跑出来的效果，transformer_test/test_data/test_config.md 是一些试验记录，最后能跑出来一个大概差强人意的小翻译模型。

主代码是 `translate.py`，只有 PyTorch 和 SentencePiece 两个依赖。从随机权重开始训练，默认中译英；用 `--reverse` 可另训一个英译中模型。没有下载预训练模型。

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

数据默认中文-tab-英文的tsv格式，我自己用的600k，但个别有bad case，例如花生→flower，因为8k vocab里花生是分开的，且数据中花生比较少。

其他情况还凑合，只能说果然数据是最重要的，希望你的数据比我的分布更合理，我抽样wmt显然不太行（笑

然后transformer_test/TRANSLATE_GUIDE.md 里是一个学习文档，和Astra几轮交互生成的，也是一些搭配的入门知识。