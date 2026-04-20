#!/bin/bash

# ===============================
# 批量评测脚本：evaluate_all.sh
# ===============================

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

# 输出目录
RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

# 遍历每个模型文件夹
for MODEL_DIR in ../models/*; do
    if [ -d "$MODEL_DIR" ]; then
        MODEL_NAME=$(basename "$MODEL_DIR")
        CHECKPOINT="$MODEL_DIR/checkpoint_best.pt"

        echo "🚀 Evaluating model: $MODEL_NAME"

        # 遍历 train/dev/test
        for SPLIT in train dev test; do
            CONLLU="$MODEL_DIR/${SPLIT}.conllu"
            OUTPUT_FILE="$RESULTS_DIR/${MODEL_NAME}_${SPLIT}_results.json"

            if [ -f "$CONLLU" ]; then
                echo "  ▶ Evaluating $SPLIT ..."
                python evaluate.py \
                    --checkpoint "$CHECKPOINT" \
                    --test_file "$CONLLU" \
                    --output_file "$OUTPUT_FILE" \
                    --detailed
            else
                echo "  ⚠️  Missing file: $CONLLU"
            fi
        done
        echo ""
    fi
done

echo "✅ All evaluations finished. Results saved in: $RESULTS_DIR"
