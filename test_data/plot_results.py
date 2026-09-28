"""Generate a complete experiment summary and loss comparison SVG.

The experiment list mirrors test_config.md.  The script intentionally keeps the
metadata here explicit so that the chart remains reproducible even when the
training CSVs only contain loss values.
"""
import csv
import html
from pathlib import Path


ROOT = Path(__file__).parent
EXPERIMENTS = [
    ("testA", "test_a_loss.csv", "200k", "3e-4", 64, 256, 3,
     "train / valid loss 在 1.5 附近震荡", "固定 lr", "原始数据"),
    ("testB", "test_b_loss.csv", "200k", "3e-4", 64, 384, 4,
     "train 继续下降，valid loss 在 1.4–1.5 附近震荡", "固定 lr", "原始数据"),
    ("testC", "test_c_loss.csv", "200k", "3e-4", 64, 512, 6,
     "valid loss 飙升到 10，训练不稳定", "固定 lr", "原始数据"),
    ("testD", "test_d_loss.csv", "200k", "1e-4", 64, 512, 6,
     "train 降至约 1.1，valid loss 在 1.5 附近震荡，疑似泛化性缺失", "固定 lr", "原始数据"),
    ("testE", "test_e_loss.csv", "600k", "1e-4", 64, 512, 6,
     "loss 到 2 左右后下降变慢，怀疑学习率不够", "固定 lr", "含垃圾数据"),
    ("testF", "test_f_loss.csv", "600k", "3e-4", 64, 512, 6,
     "warmup 后 valid loss 稳定下降到约 1.39，随后震荡", "warmup 0.1 + cosine", "含垃圾数据"),
    ("testG", "test_G_loss.csv", "600k", "3e-4", 64, 512, 6,
     "取消 cosine 衰减后 valid loss 下降较慢，在约 1.52 卡住", "warmup 0.1，无 cosine", "含垃圾数据"),
    ("testF+", "test_f+_loss.csv", "600k干净", "3e-4", 64, 512, 6,
     "清洗垃圾数据后复刻 F，valid loss 在第 16 轮达到约 1.33；下降变慢后提前终止",
     "warmup 0.1 + cosine", "干净数据"),
    ("testF+ clean", "test_f+_clean_loss.csv", "600k干净", "3e-4", 64, 512, 6,
     "重复 F+ 并训练到第 28 轮，valid loss 停在约 1.26，长句效果仍不理想",
     "warmup 0.1 + cosine", "干净数据"),
    ("f+clean fin", "f+clean_fin.csv", "600k最终清洗", "3e-4", 64, 512, 6,
     "进一步清洗数据后重跑 30 轮，valid loss 最低约 1.31，末轮约 1.31",
     "warmup 0.1 + cosine", "最终清洗数据"),
]
COLORS = {"testA": "#2563eb", "testB": "#dc2626", "testC": "#9333ea",
          "testD": "#059669", "testE": "#ea580c", "testF": "#0891b2",
          "testG": "#7c3aed", "testF+": "#16a34a", "testF+ clean": "#65a30d",
          "f+clean fin": "#be123c"}

# Add schedule/data annotations to the entries above that predate the extra
# metadata fields.  This keeps the configuration table easy to edit.
def load_data(filename):
    with (ROOT / filename).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value):
    return f"{float(value):.3f}"


def summary():
    result = []
    for name, filename, dataset, lr, batch, d_model, layers, note, schedule, data_status in EXPERIMENTS:
        rows = load_data(filename)
        train = [float(r["train_loss"]) for r in rows]
        valid = [float(r["valid_loss"]) for r in rows]
        result.append({
            "name": name, "dataset": dataset, "lr": lr, "batch": batch,
            "d_model": d_model, "layers": layers, "epochs": len(rows),
            "train_first": train[0], "train_last": train[-1],
            "valid_best": min(valid), "valid_last": valid[-1],
            "seconds_avg": sum(float(r["seconds"]) for r in rows) / len(rows),
            "note": note, "schedule": schedule, "data_status": data_status,
        })
    return result


def write_summary(rows):
    lines = [
        "# Transformer 实验结果汇总",
        "",
        "纳入 test_config.md 中全部 10 组实验：testA–testG、testF+、testF+ clean、f+clean fin。",
        "",
        "| 实验 | 数据量 | d-model | layers | lr | batch-size | epochs | schedule | 数据状态 | train 首/末 | valid 最优 | valid 末轮 | 平均每 epoch(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['dataset']} | {r['d_model']} | {r['layers']} | "
            f"{r['lr']} | {r['batch']} | {r['epochs']} | {r['schedule']} | {r['data_status']} | "
            f"{fmt(r['train_first'])} / {fmt(r['train_last'])} | "
            f"{fmt(r['valid_best'])} | {fmt(r['valid_last'])} | {r['seconds_avg']:.1f} |"
        )
    lines += ["", "## 配置与结论", ""]
    for r in rows:
        lines.append(f"- **{r['name']}**：数据集规模 `{r['dataset']}`，`--d-model {r['d_model']}`，"
                     f"`--layers {r['layers']}`，`--lr {r['lr']}`，`--batch-size {r['batch']}`。"
                     f"调度：{r['schedule']}；数据：{r['data_status']}。{r['note']}。")
    lines += ["", "## 读图说明", "", "图中只绘制 valid loss；横轴为按每个 epoch 的 seconds 累加得到的训练时间（分钟），纵轴采用分段等高显示：`1–2`、`2–11` 两段各占一半。"]
    (ROOT / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def points(rows, key, x0, y0, width, height, x_max):
    def scale(value):
        # 两段纵轴等高：1–2、2–11 各占 1/2。
        value = max(1.0, min(11.0, float(value)))
        if value <= 2:
            return (value - 1) / 2
        return 0.5 + (value - 2) / 18

    def xy(row):
        x = x0 + float(row["time_min"]) / max(1.0, x_max) * width
        y = y0 + height - scale(row[key]) * height
        return f"{x:.1f},{y:.1f}"
    return " ".join(xy(row) for row in rows)


def chart():
    datasets = []
    for name, filename, *_ in EXPERIMENTS:
        data_rows = load_data(filename)
        elapsed = 0.0
        for row in data_rows:
            elapsed += float(row["seconds"]) / 60.0
            row["time_min"] = elapsed
        datasets.append((name, data_rows))
    rows = summary()
    max_time = max(float(row["time_min"]) for _, data_rows in datasets for row in data_rows)
    width, height = 2000, 1250
    left, top, panel_w, panel_h = 70, 125, 1320, 650
    right, side_w = 1430, 500
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Arial,"Noto Sans SC","Microsoft YaHei",sans-serif;fill:#0f172a}.title{font-size:30px;font-weight:700}.subtitle{font-size:16px;fill:#64748b}.panel{font-size:21px;font-weight:700}.axis{font-size:15px;fill:#334155}.tick{font-size:13px;fill:#64748b}.legend{font-size:15px;fill:#334155}.table{font-size:14px;fill:#334155}.tablehead{font-size:14px;font-weight:700;fill:#0f172a}.note{font-size:14px;fill:#475569}.small{font-size:13px;fill:#64748b}</style>',
        '<text x="70" y="55" class="title">Transformer Loss 实验对比与参数汇总（全部 10 组）</text>',
        '<text x="70" y="86" class="subtitle">横轴：累计训练时间（分钟） · 分段等高纵轴：1–2、2–11 · 曲线：valid loss</text>',
    ]
    px, py, pw, ph = left + 82, top + 78, panel_w - 120, panel_h - 140
    parts.extend([f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" rx="12" fill="#fff" stroke="#dbe3ee"/>',
                  f'<text x="{left + 25}" y="{top + 38}" class="panel">Loss 曲线（分段等高纵轴）</text>'])
    # 纵轴刻度：1–2 区间细分，2–11 区间给出主要量级刻度。
    ticks = [1, 1.5, 2, 3, 5, 7, 9, 11]
    def y_for(value):
        value = max(1.0, min(11.0, value))
        if value <= 2:
            scaled = (value - 1) / 2
        else:
            scaled = 0.5 + (value - 2) / 18
        return py + ph - scaled * ph
    for tick in ticks:
        ty = y_for(tick)
        parts.append(f'<line x1="{px}" y1="{ty:.1f}" x2="{px + pw}" y2="{ty:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{px - 12}" y="{ty + 4:.1f}" text-anchor="end" class="tick">{tick:g}</text>')
    for label, value in [("1–2", 1.5), ("2–11", 6.5)]:
        parts.append(f'<text x="{px + pw - 8}" y="{y_for(value) + 4:.1f}" text-anchor="end" class="tick">{label}</text>')
    for boundary in [1, 2]:
        ty = y_for(boundary)
        parts.append(f'<line x1="{px}" y1="{ty:.1f}" x2="{px + pw}" y2="{ty:.1f}" stroke="#94a3b8" stroke-width="1.5"/>')
    tick_step = 50
    time_ticks = list(range(0, int(max_time) + 1, tick_step))
    if not time_ticks or time_ticks[-1] < max_time:
        time_ticks.append(max_time)
    for tick in time_ticks:
        tx = px + tick / max_time * pw
        parts.append(f'<line x1="{tx:.1f}" y1="{py}" x2="{tx:.1f}" y2="{py + ph}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{tx:.1f}" y="{py + ph + 26}" text-anchor="middle" class="tick">{tick:g}</text>')
    parts.extend([f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" stroke="#94a3b8"/>',
                  f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" stroke="#94a3b8"/>',
                  f'<text x="{px + pw / 2:.1f}" y="{py + ph + 58}" text-anchor="middle" class="axis">累计训练时间（分钟）</text>',
                  f'<text x="{left + 26}" y="{py + ph / 2:.1f}" transform="rotate(-90 {left + 26} {py + ph / 2:.1f})" text-anchor="middle" class="axis">Loss</text>'])
    for name, data_rows in datasets:
        color = COLORS[name]
        parts.append(f'<polyline points="{points(data_rows, "valid_loss", px, py, pw, ph, max_time)}" fill="none" stroke="{color}" stroke-width="3"/>')
    # 右侧独立放图例与结论，避免和参数表、曲线互相挤压。
    parts.extend([f'<rect x="{right}" y="{top}" width="{side_w}" height="{panel_h}" rx="12" fill="#fff" stroke="#dbe3ee"/>',
                  f'<text x="{right + 28}" y="{top + 40}" class="panel">实验图例</text>'])
    for i, (name, _) in enumerate(datasets):
        lx = right + 30 + (i % 2) * 225
        ly = top + 88 + (i // 2) * 48
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 34}" y2="{ly}" stroke="{COLORS[name]}" stroke-width="4"/>')
        parts.append(f'<text x="{lx + 45}" y="{ly + 5}" class="legend">{html.escape(name)}</text>')
    note_y = top + 370
    parts.extend([
        f'<line x1="{right + 25}" y1="{note_y - 28}" x2="{right + side_w - 25}" y2="{note_y - 28}" stroke="#e2e8f0"/>',
        f'<text x="{right + 28}" y="{note_y}" class="panel">关键观察</text>',
        f'<text x="{right + 28}" y="{note_y + 42}" class="note">• testC 在 512×6、lr 3e-4 下明显失稳</text>',
        f'<text x="{right + 28}" y="{note_y + 78}" class="note">• testF 比 testG 收敛更快，cosine 衰减有效</text>',
        f'<text x="{right + 28}" y="{note_y + 114}" class="note">• testF+ clean 最优 valid loss：1.268</text>',
        f'<text x="{right + 28}" y="{note_y + 150}" class="note">• f+clean fin 最优 valid loss：1.309</text>',
        f'<text x="{right + 28}" y="{note_y + 201}" class="small">注：test_f_clean.csv 与 test_f+_clean_loss.csv</text>',
        f'<text x="{right + 28}" y="{note_y + 224}" class="small">数据相同，不作为额外实验重复绘制。</text>',
    ])

    # 参数表改为底部通栏；各列留足固定空间，消除文字重叠。
    table_x, table_y, table_w, table_h = 70, 815, 1860, 365
    parts.extend([f'<rect x="{table_x}" y="{table_y}" width="{table_w}" height="{table_h}" rx="12" fill="#fff" stroke="#dbe3ee"/>',
                  f'<text x="{table_x + 25}" y="{table_y + 38}" class="panel">实验参数与结果</text>'])
    cols = [(table_x + 25, "实验"), (table_x + 190, "数据"), (table_x + 390, "d-model"),
            (table_x + 510, "层数"), (table_x + 600, "lr"), (table_x + 720, "batch"),
            (table_x + 820, "epoch"), (table_x + 930, "valid 最优"),
            (table_x + 1075, "valid 末轮"), (table_x + 1225, "训练策略"),
            (table_x + 1510, "数据状态")]
    header_y = table_y + 76
    parts.append(f'<line x1="{table_x + 20}" y1="{header_y + 13}" x2="{table_x + table_w - 20}" y2="{header_y + 13}" stroke="#cbd5e1"/>')
    for x, label in cols:
        parts.append(f'<text x="{x}" y="{header_y}" class="tablehead">{label}</text>')
    for i, row in enumerate(rows):
        y = header_y + 39 + i * 26
        if i % 2:
            parts.append(f'<rect x="{table_x + 18}" y="{y - 18}" width="{table_w - 36}" height="26" fill="#f8fafc"/>')
        values = [row["name"], row["dataset"], str(row["d_model"]), str(row["layers"]),
                  row["lr"], str(row["batch"]), str(row["epochs"]), fmt(row["valid_best"]),
                  fmt(row["valid_last"]), row["schedule"], row["data_status"]]
        for (x, _), value in zip(cols, values):
            parts.append(f'<text x="{x}" y="{y}" class="table">{html.escape(value)}</text>')
    parts += ['<text x="70" y="1220" class="small">数据来源：test_config.md 与 test_data/*.csv；时间按各 epoch seconds 累加。</text>',
              '</svg>']
    (ROOT / "training_loss_comparison.svg").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    rows = summary()
    write_summary(rows)
    chart()
    print("generated experiment_summary.md and training_loss_comparison.svg")
