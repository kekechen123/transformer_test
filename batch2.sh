MODEL_OUT="runs/clean_data2"
TEST_FILE="test_data/test_data/test2.md"
OUTPUT_FILE="test_results2.md"
BEAM_SIZE=4

{
    printf '# 翻译测试结果\n\n'
    printf '模型目录：`%s`\n\n' "$MODEL_OUT"
    printf '测试集：`%s`\n\n' "$TEST_FILE"
    printf '推理设备：CPU\n\n'
    printf '解码方式：Beam search（beam size = %d）\n\n' "$BEAM_SIZE"
    printf '| 序号 | 原文 | Beam 翻译 |\n'
    printf '|---:|---|---|\n'
} > "$OUTPUT_FILE"

escape_md_cell() {
    local text="$1"
    text=${text//\\/\\\\}
    text=${text//|/\\|}
    text=${text//$'\r'/}
    text=${text//$'\n'/<br>}
    printf '%s' "$text"
}

sentence_no=0

while IFS= read -r sentence || [[ -n "$sentence" ]]; do
    # 跳过空行
    [[ -z "$sentence" ]] && continue

    sentence_no=$((sentence_no + 1))

    translation=$(
        CUDA_VISIBLE_DEVICES="" python3 translate.py \
            --out "$MODEL_OUT" \
            --text "$sentence" \
            --decode beam \
            --beam-size "$BEAM_SIZE" \
        | tail -n 1
    )

    printf '| %d | %s | %s |\n' \
        "$sentence_no" \
        "$(escape_md_cell "$sentence")" \
        "$(escape_md_cell "$translation")" \
        >> "$OUTPUT_FILE"
done < "$TEST_FILE"

printf '测试完成，结果已保存到：%s\n' "$OUTPUT_FILE"
