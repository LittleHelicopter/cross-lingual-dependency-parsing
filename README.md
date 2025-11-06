# Cross-lingual Dependency Parsing
## Introduction

Languages: 

English & German (same family), English & Chinese (different families)
*optional: Arabic & Hebrew(same family), Chinese & Tibetan/Cantonese/Burmese (to check availability)
Most of these languages can also be extended to constituency parsing experiments.
Experimental setup:

Stage 1: Train XLM-R + parsing head(full finetune) on Ls and save the encoder weights.
Stage 2: Load the encoder from Stage 1, freeze it, randomly initialize a new parsing head, and train only the head on Lt.
Baseline: Use pretrained XLM-R with frozen encoder, train only the head on Lt.
Upper bound: Full fine-tuning (encoder + head) on Lt.
Evaluation: 

Compare the model’s performance on Lt under different settings.


## Setup Environment

conda create -n xlmr_parse python=3.10 -y

conda activate xlmr_parse

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install conllu

pip install transformers



## Example Training

### Stage 1

unfrozen encoder
ls: english

```bash
python src/train_parser.py \
  --train_file data/UD_English-EWT/en_ewt-ud-train.conllu \
  --dev_file data/UD_English-EWT/en_ewt-ud-dev.conllu \
  --test_file data/UD_English-EWT/en_ewt-ud-test.conllu \
  --model_name xlm-roberta-base \
  --batch_size 32 \
  --epochs 20 \
  --lr 2e-5 \
  --output_dir models \
  --exp_name baseline_en_full
```

### Stage 2

quick run:
```
chmod +x src/train_stage2.sh
src/train_stage2.sh
```
include:


frozen encoder
lt: chinese(different group)

```bash
python src/train_parser.py \
  --train_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-train.conllu \
  --dev_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-dev.conllu \
  --test_file data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-test.conllu \
  --model_name xlm-roberta-base \
  --load_encoder models/baseline_en_full/best_model.pt \
  --freeze_encoder \
  --batch_size 32 \
  --epochs 20 \
  --lr 5e-4 \
  --output_dir models \
  --exp_name transfer_en2zh
```

lt: german(same group)

```bash
python src/train_parser.py \
  --train_file data/UD_German-GSD/de_gsd-ud-train.conllu \
  --dev_file data/UD_German-GSD/de_gsd-ud-dev.conllu \
  --test_file data/UD_German-GSD/de_gsd-ud-test.conllu \
  --model_name xlm-roberta-base \
  --load_encoder models/baseline_en_full/best_model.pt \
  --freeze_encoder \
  --batch_size 32 \
  --epochs 20 \
  --lr 5e-4 \
  --output_dir models \
  --exp_name transfer_en2de
```



### Upperbound

quick run:
```
chmod +x src/train_upperbound.sh
src/train_upperbound.sh
```





# xxx




check num_label:
grep -v '^#' data/UD_English-EWT/en_ewt-ud-train.conllu | awk '{print $8}' | sort | uniq | wc -l

grep -v '^#' data/UD_German-GSD/de_gsd-ud-train.conllu | awk '{print $8}' | sort | uniq | wc -l


## note：

德语的依存关系不需要更新英语的label，而中文需要