#!/bin/bash

# ===============================
# Evaluate Dependency Parser
# ===============================

# ====== 基本配置 ======
CHECKPOINT=/mnt/ssd/weicheng/data_interns/yahan/syntactic-parsing/models/last4_en2de/best_model.pt
TEST_FILE=/mnt/ssd/weicheng/data_interns/yahan/syntactic-parsing/data/UD_German-GSD/de_gsd-ud-test.conllu
BATCH_SIZE=32
OUTPUT_FILE=/mnt/ssd/weicheng/data_interns/yahan/syntactic-parsing/models/last4_en2de/test_results_relation_type.json

# ====== 可选：指定GPU ======
export CUDA_VISIBLE_DEVICES=0

# ====== 运行评估 ======
python eval_parser.py \
    --checkpoint ${CHECKPOINT} \
    --test_file ${TEST_FILE} \
    --batch_size ${BATCH_SIZE} \
    --output_file ${OUTPUT_FILE} \
    --detailed