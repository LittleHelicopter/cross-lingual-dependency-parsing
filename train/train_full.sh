#!/bin/bash

# 希伯来语 GPU 0
CUDA_VISIBLE_DEVICES=0 python src/train_parser.py \
  --train_file data/UD_Arabic-PADT/ar_padt-ud-train.conllu \
  --dev_file data/UD_Arabic-PADT/ar_padt-ud-dev.conllu \
  --test_file data/UD_Arabic-PADT/ar_padt-ud-test.conllu \
  --model_name xlm-roberta-base \
  --batch_size 32 \
  --epochs 50 \
  --lr 5e-4 \
  --output_dir models \
  --exp_name baseline_ar_full &

# 阿拉伯语 GPU 1
CUDA_VISIBLE_DEVICES=1 python src/train_parser.py \
  --train_file data/UD_Hebrew-HTB/he_htb-ud-train.conllu \
  --dev_file data/UD_Hebrew-HTB/he_htb-ud-dev.conllu \
  --test_file data/UD_Hebrew-HTB/he_htb-ud-test.conllu \
  --model_name xlm-roberta-base \
  --batch_size 32 \
  --epochs 50 \
  --lr 5e-4 \
  --output_dir models \
  --exp_name baseline_he_full &

# 英文任务
# CUDA_VISIBLE_DEVICES=2 python src/train_parser.py \
#   --train_file data/UD_English-EWT/en_ewt-ud-train.conllu \
#   --dev_file data/UD_English-EWT/en_ewt-ud-dev.conllu \
#   --test_file data/UD_English-EWT/en_ewt-ud-test.conllu \
#   --model_name xlm-roberta-base \
#   --batch_size 32 \
#   --epochs 50 \
#   --lr 2e-5 \
#   --output_dir models \
#   --exp_name baseline_en_full &

# # 中文任务 GPU 1
# CUDA_VISIBLE_DEVICES=3 python src/train_parser.py \
#   --train_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-train.conllu \
#   --dev_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-dev.conllu \
#   --test_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-test.conllu \
#   --model_name xlm-roberta-base \
#   --batch_size 32 \
#   --epochs 50 \
#   --lr 5e-4 \
#   --output_dir models \
#   --exp_name baseline_zh_full &

# # 德文任务 GPU 2
CUDA_VISIBLE_DEVICES=2 python src/train_parser.py \
  --train_file data/UD_German-GSD/de_gsd-ud-train.conllu \
  --dev_file data/UD_German-GSD/de_gsd-ud-dev.conllu \
  --test_file data/UD_German-GSD/de_gsd-ud-test.conllu \
  --model_name xlm-roberta-base \
  --batch_size 32 \
  --epochs 50 \
  --lr 5e-4 \
  --output_dir models \
  --exp_name baseline_de_full &



# 等待所有任务完成
wait

echo "All three trainings are running in parallel!"
