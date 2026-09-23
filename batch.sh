
MODEL_OUT="runs/clean_data2"
TEST_FILE="test_data/test_data/test.md"
OUTPUT_FILE="test_results.md"

{
    printf '# 翻译测试结果\n\n'
    printf '模型目录：`%s`\n\n' "$MODEL_OUT"
    printf '推理设备：CPU\n\n'
} > "$OUTPUT_FILE"

sentence_no=0

while IFS= read -r sentence || [[ -n "$sentence" ]]; do
    # 跳过空行
    [[ -z "$sentence" ]] && continue

    sentence_no=$((sentence_no + 1))

    {
        printf '## 第 %d 句\n\n' "$sentence_no"
        printf '**原文：** %s\n\n' "$sentence"

        printf '### Greedy\n\n'
        CUDA_VISIBLE_DEVICES="" python3 translate.py \
            --out "$MODEL_OUT" \
            --text "$sentence" \
            --decode greedy \
        | tail -n 1
        printf '\n\n'

        for beam_size in 2 4 8; do
            printf '### Beam size = %d\n\n' "$beam_size"

            CUDA_VISIBLE_DEVICES="" python3 translate.py \
                --out "$MODEL_OUT" \
                --text "$sentence" \
                --decode beam \
                --beam-size "$beam_size" \
            | tail -n 1

            printf '\n\n'
        done

        printf '%s\n\n' '---'
    } >> "$OUTPUT_FILE"
done < "$TEST_FILE"

printf '测试完成，结果已保存到：%s\n' "$OUTPUT_FILE"