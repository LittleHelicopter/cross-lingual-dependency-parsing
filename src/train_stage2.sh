#!/bin/bash

# first stage: full
# 中文任务 GPU 1
CUDA_VISIBLE_DEVICES=0 python src/train_parser.py \
  --train_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-train.conllu \
  --dev_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-dev.conllu \
  --test_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-test.conllu \
  --model_name xlm-roberta-base \
  --load_encoder models/baseline_en_full/best_model.pt \
  --freeze_encoder \
  --batch_size 32 \
  --epochs 50 \
  --lr 5e-4 \
  --output_dir models \
  --exp_name full_en2zh &

# 德文任务 GPU 2
CUDA_VISIBLE_DEVICES=1 python src/train_parser.py \
  --train_file data/UD_German-GSD/de_gsd-ud-train.conllu \
  --dev_file data/UD_German-GSD/de_gsd-ud-dev.conllu \
  --test_file data/UD_German-GSD/de_gsd-ud-test.conllu \
  --model_name xlm-roberta-base \
  --load_encoder models/baseline_en_full/best_model.pt \
  --freeze_encoder \
  --batch_size 32 \
  --epochs 50 \
  --lr 5e-4 \
  --output_dir models \
  --exp_name full_en2de &

# first stage: last 4 layers
# 中文任务 GPU 1
CUDA_VISIBLE_DEVICES=2 python src/train_parser.py \
  --train_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-train.conllu \
  --dev_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-dev.conllu \
  --test_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-test.conllu \
  --model_name xlm-roberta-base \
  --load_encoder models/last4_en/best_model.pt \
  --freeze_encoder \
  --batch_size 32 \
  --epochs 50 \
  --lr 5e-4 \
  --output_dir models \
  --exp_name last4_en2zh &

# 德文任务 GPU 2
CUDA_VISIBLE_DEVICES=3 python src/train_parser.py \
  --train_file data/UD_German-GSD/de_gsd-ud-train.conllu \
  --dev_file data/UD_German-GSD/de_gsd-ud-dev.conllu \
  --test_file data/UD_German-GSD/de_gsd-ud-test.conllu \
  --model_name xlm-roberta-base \
  --load_encoder models/last4_en/best_model.pt \
  --freeze_encoder \
  --batch_size 32 \
  --epochs 50 \
  --lr 5e-4 \
  --output_dir models \
  --exp_name last4_en2de &

# 等待所有任务完成
wait

echo "All three trainings are running in parallel!"
