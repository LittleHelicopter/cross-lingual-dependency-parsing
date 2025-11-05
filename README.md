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

### Baseline English

(unfrozen encoder)
```bash
python src/train_parser.py \
  --train_file data/UD_English-EWT/en_ewt-ud-train.conllu \
  --dev_file data/UD_English-EWT/en_ewt-ud-dev.conllu \
  --test_file data/UD_English-EWT/en_ewt-ud-test.conllu \
  --model_name xlm-roberta-base \
  --batch_size 8 \
  --epochs 20 \
  --lr 2e-5 \
  --output_dir models \
  --exp_name baseline_en_full
```

check num_label:
grep -v '^#' data/UD_English-EWT/en_ewt-ud-train.conllu | awk '{print $8}' | sort | uniq | wc -l


