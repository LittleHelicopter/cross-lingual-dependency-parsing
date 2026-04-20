#!/bin/bash

# ----------------------------
# 模型总层数（根据使用的模型修改）
# ----------------------------
TOTAL_LAYERS=12

# ----------------------------
# 可配置任务列表
# ----------------------------
declare -A TASKS
# TASKS[language]="GPU,train_file,dev_file,test_file,model_name"
TASKS[ar]="0,data/UD_Arabic-PADT/ar_padt-ud-train.conllu,data/UD_Arabic-PADT/ar_padt-ud-dev.conllu,data/UD_Arabic-PADT/ar_padt-ud-test.conllu,xlm-roberta-base"
TASKS[he]="1,data/UD_Hebrew-HTB/he_htb-ud-train.conllu,data/UD_Hebrew-HTB/he_htb-ud-dev.conllu,data/UD_Hebrew-HTB/he_htb-ud-test.conllu,xlm-roberta-base"
TASKS[zh]="2,data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-train.conllu,data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-dev.conllu,data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-test.conllu,xlm-roberta-base"
TASKS[de]="3,data/UD_German-GSD/de_gsd-ud-train.conllu,data/UD_German-GSD/de_gsd-ud-dev.conllu,data/UD_German-GSD/de_gsd-ud-test.conllu,xlm-roberta-base"
TASKS[en]="4,data/UD_English-EWT/en_ewt-ud-train.conllu,data/UD_English-EWT/en_ewt-ud-dev.conllu,data/UD_English-EWT/en_ewt-ud-test.conllu,xlm-roberta-base"

# ----------------------------
# 可修改的训练参数
# ----------------------------
FREEZE_UNTIL=11       # 要冻结的层数上界（只训练后面的层）
BATCH_SIZE=32
LR=5e-4
EPOCHS=50
OUTPUT_DIR=models

# ----------------------------
# 循环启动任务
# ----------------------------
for LANG in "${!TASKS[@]}"; do
    IFS=',' read -r GPU TRAIN DEV TEST MODEL <<< "${TASKS[$LANG]}"
    
    # 根据 freeze_until 自动计算 last k 层
    LAST_K=$((TOTAL_LAYERS - FREEZE_UNTIL))
    EXP_NAME="last${LAST_K}_${LANG}"
    
    echo "Starting training for $EXP_NAME on GPU $GPU ..."
    CUDA_VISIBLE_DEVICES=$GPU python src/train_parser.py \
        --train_file $TRAIN \
        --dev_file $DEV \
        --test_file $TEST \
        --model_name $MODEL \
        --freeze_until $FREEZE_UNTIL \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --output_dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --exp_name $EXP_NAME &
done

# 等待所有后台任务完成
wait
echo "All trainings have finished!"
