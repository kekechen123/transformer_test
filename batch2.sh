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
} > "$OUTPUT_FILE"

sentence_no=0

while IFS= read -r sentence || [[ -n "$sentence" ]]; do
    # 跳过空行
    [[ -z "$sentence" ]] && continue

    sentence_no=$((sentence_no + 1))

    {
        printf '## 第 %d 句\n\n' "$sentence_no"
        printf '**原文：** %s\n\n' "$sentence"
        printf '### Beam size = %d\n\n' "$BEAM_SIZE"

        CUDA_VISIBLE_DEVICES="" python3 translate.py \
            --out "$MODEL_OUT" \
            --text "$sentence" \
            --decode beam \
            --beam-size "$BEAM_SIZE" \
        | tail -n 1

        printf '\n\n%s\n\n' '---'
    } >> "$OUTPUT_FILE"
done < "$TEST_FILE"

printf '测试完成，结果已保存到：%s\n' "$OUTPUT_FILE"
