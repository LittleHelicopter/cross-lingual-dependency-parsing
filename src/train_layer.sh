#!/bin/bash

# 中文任务 GPU 1
CUDA_VISIBLE_DEVICES=0 python src/train_parser.py \
  --train_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-train.conllu \
  --dev_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-dev.conllu \
  --test_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-test.conllu \
  --model_name xlm-roberta-base \
  --freeze_until 8 \
  --batch_size 32 \
  --lr 5e-4 \
  --output_dir models \
  --epochs 50 \
  --exp_name last4_zh &

# 德文任务 GPU 2
CUDA_VISIBLE_DEVICES=2 python src/train_parser.py \
  --train_file data/UD_German-GSD/de_gsd-ud-train.conllu \
  --dev_file data/UD_German-GSD/de_gsd-ud-dev.conllu \
  --test_file data/UD_German-GSD/de_gsd-ud-test.conllu \
  --model_name xlm-roberta-base \
  --freeze_until 8 \
  --batch_size 32 \
  --lr 5e-4 \
  --output_dir models \
  --epochs 50 \
  --exp_name last4_de &

# English
CUDA_VISIBLE_DEVICES=3 python src/train_parser.py \
  --train_file data/UD_English-EWT/en_ewt-ud-train.conllu \
  --dev_file data/UD_English-EWT/en_ewt-ud-dev.conllu \
  --test_file data/UD_English-EWT/en_ewt-ud-test.conllu \
  --model_name xlm-roberta-base \
  --freeze_until 8 \
  --batch_size 32 \
  --lr 5e-4 \
  --output_dir models \
  --epochs 50 \
  --exp_name last4_en &

# 等待所有任务完成
wait

echo "All three trainings are running in parallel!"
