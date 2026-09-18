"""Generate ordered experiment summary and loss comparison SVG for test_data."""
import csv
import html
from pathlib import Path


ROOT = Path(__file__).parent
EXPERIMENTS = [
    ("testA", "test_a_loss.csv", "200k", "3e-4", 64, 256, 3,
     "train / valid loss 在 1.5 附近震荡"),
    ("testB", "test_b_loss.csv", "200k", "3e-4", 64, 384, 4,
     "train 继续下降，valid loss 在 1.4–1.5 附近震荡"),
    ("testC", "test_c_loss.csv", "200k", "3e-4", 64, 512, 6,
     "valid loss 飙升到 10，训练不稳定"),
    ("testD", "test_d_loss.csv", "200k", "1e-4", 64, 512, 6,
     "train 降至约 1.1，valid loss 在 1.5 附近震荡，泛化性缺失"),
    ("testE", "test_e_loss.csv", "600k", "1e-4", 64, 512, 6,
     "loss 到 2 左右后下降变慢，可能需要更长训练或调整学习率"),
]
COLORS = {"testA": "#2563eb", "testB": "#dc2626", "testC": "#9333ea",
          "testD": "#059669", "testE": "#ea580c"}


def load_data(filename):
    with (ROOT / filename).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value):
    return f"{float(value):.3f}"


def summary():
    result = []
    for name, filename, dataset, lr, batch, d_model, layers, note in EXPERIMENTS:
        rows = load_data(filename)
        train = [float(r["train_loss"]) for r in rows]
        valid = [float(r["valid_loss"]) for r in rows]
        result.append({
            "name": name, "dataset": dataset, "lr": lr, "batch": batch,
            "d_model": d_model, "layers": layers, "epochs": len(rows),
            "train_first": train[0], "train_last": train[-1],
            "valid_best": min(valid), "valid_last": valid[-1],
            "seconds_avg": sum(float(r["seconds"]) for r in rows) / len(rows),
            "note": note,
        })
    return result


def write_summary(rows):
    lines = [
        "# Transformer 实验结果汇总",
        "",
        "仅纳入已有 CSV 结果的 testA–testE；testF 尚未开始，未纳入表格和图。",
        "",
        "| 实验 | 数据量 | d-model | layers | lr | batch-size | epochs | train 首/末 | valid 最优 | valid 末轮 | 平均每 epoch(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['dataset']} | {r['d_model']} | {r['layers']} | "
            f"{r['lr']} | {r['batch']} | {r['epochs']} | "
            f"{fmt(r['train_first'])} / {fmt(r['train_last'])} | "
            f"{fmt(r['valid_best'])} | {fmt(r['valid_last'])} | {r['seconds_avg']:.1f} |"
        )
    lines += ["", "## 配置与结论", ""]
    for r in rows:
        lines.append(f"- **{r['name']}**：数据集规模 `{r['dataset']}`，`--d-model {r['d_model']}`，"
                     f"`--layers {r['layers']}`，`--lr {r['lr']}`，`--batch-size {r['batch']}`。"
                     f" {r['note']}。")
    lines += ["", "## 读图说明", "", "图中只绘制 valid loss，横轴为按每个 epoch 的 seconds 累加得到的训练时间（分钟），纵轴采用分段等高显示：`1–2`、`2–11` 两段各占一半。"]
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
    width, height = 1700, 930
    left, top, panel_w, panel_h = 70, 125, 1050, 690
    right = 1160
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1700" height="930" viewBox="0 0 1700 930">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Arial,"Noto Sans SC",sans-serif;fill:#0f172a}.title{font-size:30px;font-weight:700}.subtitle{font-size:15px;fill:#64748b}.panel{font-size:20px;font-weight:700}.axis{font-size:14px;fill:#334155}.tick{font-size:12px;fill:#64748b}.legend{font-size:14px;fill:#334155}.table{font-size:13px;fill:#334155}.tablehead{font-size:13px;font-weight:700;fill:#0f172a}.note{font-size:14px;fill:#475569}</style>',
        '<text x="70" y="55" class="title">Transformer Loss 实验对比与参数汇总（testA–testE）</text>',
        '<text x="70" y="84" class="subtitle">横轴：累计训练时间（分钟）　·　分段等高纵轴：1–2、2–11　·　仅显示 valid loss　·　F 尚未开始</text>',
    ]
    px, py, pw, ph = left + 78, top + 78, panel_w - 115, panel_h - 150
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
    # 右侧参数表。
    table_x, table_y, table_w = right, top, width - right - 55
    parts.extend([f'<rect x="{table_x}" y="{table_y}" width="{table_w}" height="{panel_h}" rx="12" fill="#fff" stroke="#dbe3ee"/>',
                  f'<text x="{table_x + 25}" y="{table_y + 38}" class="panel">实验参数与结果</text>'])
    cols = [(table_x + 22, "实验"), (table_x + 76, "数据"), (table_x + 132, "d-model"),
            (table_x + 202, "层数"), (table_x + 245, "lr"), (table_x + 300, "batch"),
            (table_x + 355, "epoch"), (table_x + 410, "valid 最优")]
    header_y = table_y + 75
    parts.append(f'<line x1="{table_x + 18}" y1="{header_y + 10}" x2="{table_x + table_w - 18}" y2="{header_y + 10}" stroke="#cbd5e1"/>')
    for x, label in cols:
        parts.append(f'<text x="{x}" y="{header_y}" class="tablehead">{label}</text>')
    for i, row in enumerate(rows):
        y = header_y + 48 + i * 48
        parts.append(f'<line x1="{table_x + 18}" y1="{y + 17}" x2="{table_x + table_w - 18}" y2="{y + 17}" stroke="#eef2f7"/>')
        values = [row["name"], row["dataset"], str(row["d_model"]), str(row["layers"]), row["lr"], str(row["batch"]), str(row["epochs"]), fmt(row["valid_best"])]
        for (x, _), value in zip(cols, values):
            parts.append(f'<text x="{x}" y="{y}" class="table">{html.escape(value)}</text>')
    legend_y = top + panel_h + 38
    for i, (name, _) in enumerate(datasets):
        lx = left + i * 135
        parts.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 30}" y2="{legend_y}" stroke="{COLORS[name]}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 38}" y="{legend_y + 5}" class="legend">{name}</text>')
    parts += [#f'<line x1="{left + 720}" y1="{legend_y}" x2="{left + 750}" y2="{legend_y}" stroke="#475569" stroke-width="3"/>',
              #f'<text x="{left + 758}" y="{legend_y + 5}" class="legend">valid loss</text>',
              '<text x="70" y="875" class="note">testC：512×6、3e-4 的 valid loss 明显失稳；testD：降低 lr 后恢复到约 1.5。</text>',
              '</svg>']
    (ROOT / "training_loss_comparison.svg").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    rows = summary()
    write_summary(rows)
    chart()
    print("generated experiment_summary.md and training_loss_comparison.svg")
