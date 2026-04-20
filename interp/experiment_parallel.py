# ============================================================
# experiment_parallel_comparison.py
# Cross-lingual Parallel Comparison
# 完整流程：Localization → Necessity → Circuit
#           与单语言（experiment_de.py）逐步对标
#
# 核心设计：
#   - Patching metric  = logit boost on DE corr sentence
#                        （来源激活从 EN clean → DE corr，subj 位置对齐）
#   - Ablation metric  = logit drop on DE clean sentence
#                        （mean ablation，与单语言完全一致）
#   - Joint Score      = normalized patch boost + normalized abl drop
#                        （与单语言完全一致）
#   - Positive Control : EN clean 激活 → DE corr，subj 位置
#   - Negative Control : EN corr  激活 → DE corr，subj 位置
#   - Position Control : EN clean 激活 → DE corr，random 位置
# ============================================================


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import itertools

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer

sys.path.append("/mnt/ssd/weicheng/data_interns/yahan/syntactic-parsing/train/")
from train_parser import BiaffineDependencyParser

from utils import (
    resolve_token_indices,
    apply_first_subword_mask,
    get_arc_logit,
    get_agreement_delta,
    get_activations,
    compute_mean_activations,
    load_jsonl,
)


# =============================================
# 1. 环境配置与模型加载
# =============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DE_CHECKPOINT_PATH = "/mnt/ssd/weicheng/data_interns/yahan/syntactic-parsing/models/last4_de/best_model.pt"
EN_CHECKPOINT_PATH = "/mnt/ssd/weicheng/data_interns/yahan/syntactic-parsing/models/last4_en/best_model.pt"
EN_JSONL_PATH = "../data/en_100_cleaned_data.jsonl"
DE_JSONL_PATH = "../data/de_100_cleaned_data.jsonl"

# ── DE 模型（主模型，用于所有 patching / ablation）──
print(f"正在从 {DE_CHECKPOINT_PATH} 加载 DE 模型...")
de_checkpoint = torch.load(DE_CHECKPOINT_PATH, map_location=device)
de_args = de_checkpoint["args"]

tokenizer  = AutoTokenizer.from_pretrained(de_args["model_name"])
de_encoder = AutoModel.from_pretrained(de_args["model_name"])
de_num_labels = de_checkpoint["model_state_dict"]["rel_biaffine.weight"].shape[0]

model = BiaffineDependencyParser(
    encoder=de_encoder,
    num_labels=de_num_labels,
    hidden_size=de_encoder.config.hidden_size,
    mlp_size=de_args["mlp_size"],
    dropout=de_args["dropout"],
)
model.load_state_dict(de_checkpoint["model_state_dict"])
model.to(device).eval()

n_layers = model.encoder.config.num_hidden_layers
n_heads  = model.encoder.config.num_attention_heads
head_dim = model.encoder.config.hidden_size // n_heads

print(f"DE 模型加载成功！设备: {device} | layers={n_layers}, heads={n_heads}, head_dim={head_dim}")

# ── EN 模型（用于 EN 侧 baseline 和 EN cross-ablation）──
print(f"正在从 {EN_CHECKPOINT_PATH} 加载 EN 模型...")
en_checkpoint = torch.load(EN_CHECKPOINT_PATH, map_location=device)
en_args = en_checkpoint["args"]

en_encoder    = AutoModel.from_pretrained(en_args["model_name"])
en_num_labels = en_checkpoint["model_state_dict"]["rel_biaffine.weight"].shape[0]

model_en = BiaffineDependencyParser(
    encoder=en_encoder,
    num_labels=en_num_labels,
    hidden_size=en_encoder.config.hidden_size,
    mlp_size=en_args["mlp_size"],
    dropout=en_args["dropout"],
)
model_en.load_state_dict(en_checkpoint["model_state_dict"])
model_en.to(device).eval()

# XLM-R 共享 tokenizer，直接复用
print(f"EN 模型加载成功！")
print(f"  DE model backbone: {de_args['model_name']}")
print(f"  EN model backbone: {en_args['model_name']}")


# =============================================
# 2. 数据加载 + Token Index 解析
# =============================================

en_data = load_jsonl(EN_JSONL_PATH)
de_data = load_jsonl(DE_JSONL_PATH)

common_ids = sorted(set(en_data.keys()) & set(de_data.keys()))
dataset = []
for id_ in common_ids:
    item = {
        "id":         id_,
        "en_clean":   en_data[id_]["en_clean"],
        "en_corr":    en_data[id_]["en_corr"],
        "en_indices": en_data[id_]["en_indices"],
        "en_meta":    en_data[id_].get("metadata", {}),
        "de_clean":   de_data[id_]["de_clean"],
        "de_corr":    de_data[id_]["de_corr"],
        "de_indices": de_data[id_]["de_indices"],
        "de_meta":    de_data[id_].get("metadata", {}),
    }
    dataset.append(item)

print(f"✅ 加载 EN {len(en_data)} 条 | DE {len(de_data)} 条 | 配对成功: {len(dataset)} 条")

# ── Token Index 解析（EN 和 DE 各自用对应语言的 meta）──
errors = []
for item in dataset:
    item["_tok_en"] = resolve_token_indices(item, tokenizer, "en")
    item["_tok_de"] = resolve_token_indices(item, tokenizer, "de")
    for lang in ["en", "de"]:
        tok = item[f"_tok_{lang}"]
        if any(v is None for v in tok.values()):
            errors.append(f"id={item['id']} lang={lang} | {tok}")

print(f"✅ Token Index 解析完成，{len(errors)} 条有问题：")
for e in errors[:10]:
    print(" ", e)

# 快速抽样校验（与单语言对标）
for item in dataset[:3]:
    for lang in ["en", "de"]:
        tok = item[f"_tok_{lang}"]
        cw  = item[f"{lang}_clean"].split()
        rw  = item[f"{lang}_corr"].split()
        sw, vw = tok["subj_word_idx"], tok["verb_word_idx"]
        print(f"\nid={item['id']} [{lang.upper()}]")
        print(f"  subj word_idx={sw} -> '{cw[sw] if sw is not None else None}' "
              f"| subj_tok_clean={tok['subj_tok_clean']}")
        print(f"  verb word_idx={vw} -> "
              f"clean='{cw[vw] if vw is not None else None}' "
              f"corr='{rw[vw] if vw is not None else None}'")
        print(f"  verb_tok_clean={tok['verb_tok_clean']}, verb_tok_corr={tok['verb_tok_corr']}")


# =============================================
# 3. Baseline 评估
# =============================================
#
# 对标单语言 Cell 4：
#   单语言：get_agreement_delta(model, item, lang="de")
#   跨语言：同时对 DE 模型算 DE 句 delta，对 EN 模型算 EN 句 delta
#   EN delta 用于后续 EN 侧 ablation 的归一化基准

print("📊 计算 baseline agreement delta...")

results_baseline = []
for item in dataset:
    de_cl, de_co, de_d = get_agreement_delta(
        model, tokenizer, item, lang="de", device=device, tok_key="_tok_de"
    )
    en_cl, en_co, en_d = get_agreement_delta(
        model_en, tokenizer, item, lang="en", device=device, tok_key="_tok_en"
    )
    results_baseline.append({
        "item":           item,
        "de_clean_logit": de_cl,
        "de_corr_logit":  de_co,
        "de_delta":       de_d,
        "en_clean_logit": en_cl,
        "en_corr_logit":  en_co,
        "en_delta":       en_d,
    })

valid    = [r for r in results_baseline if r["de_delta"] is not None]
de_deltas = [r["de_delta"] for r in valid]

print(f"\n总配对: {len(dataset)} | DE delta 可计算（有效）: {len(valid)}")

print(f"\n--- DE Baseline Agreement Delta (DE model) ---")
print(f"Mean clean logit : {np.mean([r['de_clean_logit'] for r in valid]):.4f}")
print(f"Mean corr  logit : {np.mean([r['de_corr_logit']  for r in valid]):.4f}")
print(f"Mean delta       : {np.mean(de_deltas):.4f}")
print(f"delta > 0 比例   : {np.mean([d > 0 for d in de_deltas]):.2%}")

en_valid  = [r for r in results_baseline if r["en_delta"] is not None]
en_deltas = [r["en_delta"] for r in en_valid]
print(f"\n--- EN Baseline Agreement Delta (EN model) ---")
print(f"Mean clean logit : {np.mean([r['en_clean_logit'] for r in en_valid]):.4f}")
print(f"Mean corr  logit : {np.mean([r['en_corr_logit']  for r in en_valid]):.4f}")
print(f"Mean delta       : {np.mean(en_deltas):.4f}")
print(f"delta > 0 比例   : {np.mean([d > 0 for d in en_deltas]):.2%}")

# 分布图（对标单语言 baseline_delta_de.png，补充 EN 对照）
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(de_deltas, bins=30, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="red", linestyle="--", label="delta=0")
axes[0].set_xlabel("Agreement Delta (clean_logit - corr_logit)")
axes[0].set_ylabel("Count")
axes[0].set_title("Baseline Agreement Delta Distribution (DE model, DE sentences)")
axes[0].legend()
axes[1].hist(en_deltas, bins=30, color="darkorange", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--", label="delta=0")
axes[1].set_xlabel("Agreement Delta (clean_logit - corr_logit)")
axes[1].set_ylabel("Count")
axes[1].set_title("Baseline Agreement Delta Distribution (EN model, EN sentences)")
axes[1].legend()
plt.tight_layout()
plt.savefig("cl_baseline_delta_distribution.png", dpi=150)
plt.show()

# ── 过滤策略：与单语言一致，使用全部 DE delta 可计算的样本 ──
filtered         = [r for r in valid]
filtered_dataset = [r["item"] for r in filtered]

print(f"\n有效样本（DE delta 可计算）: {len(filtered_dataset)} 条")
print(f"Mean DE delta (filtered): {np.mean([r['de_delta'] for r in filtered]):.4f}")

# clean logit 分层分析（与单语言 Cell 4 对标）
print("\n--- DE Clean Logit 分层 ---")
for t in [-5, -3, -2, -1, 0]:
    sub = [r for r in filtered if r["de_clean_logit"] > t]
    if sub:
        print(f"  de_clean_logit > {t:2d}: {len(sub):3d} 条 | "
              f"Mean delta = {np.mean([r['de_delta'] for r in sub]):.4f}")

# EN 侧有效样本（用于后续 EN cross-ablation 归一化）
filtered_en      = [r for r in filtered if r["en_delta"] is not None]
print(f"\nEN 模型有效样本（用于 EN cross-ablation）: {len(filtered_en)} 条")
print(f"Mean EN delta: {np.mean([r['en_delta'] for r in filtered_en]):.4f}")


# =============================================
# 4. 预计算激活与基准值
# =============================================
#
# 对标单语言 Cell 5：
#   单语言：预计算 de_clean 激活 + de_corr inputs
#   跨语言：额外预计算 en_clean / en_corr 激活（作为注入源）
#           de_corr inputs 作为注入目标

print("📦 预计算 filtered 数据集的激活与基准值...")

for r in filtered:
    item = r["item"]
    # EN clean 激活（用于 Positive Control）
    r["en_clean_acts"] = get_activations(
        model, tokenizer(item["en_clean"], return_tensors="pt").to(device)
    )
    # EN corr 激活（用于 Negative Control）
    r["en_corr_acts"] = get_activations(
        model, tokenizer(item["en_corr"], return_tensors="pt").to(device)
    )
    # DE corr inputs（注入目标）
    r["de_corr_inputs"] = tokenizer(item["de_corr"], return_tensors="pt").to(device)

all_base_corr_logits = [r["de_corr_logit"] for r in filtered]

print(f"✅ 预计算完成，共 {len(filtered_dataset)} 条")
print(f"Base DE corr logit mean: {np.mean(all_base_corr_logits):.4f}")


# =============================================
# 5. Cross-lingual Activation Patching — 逐 Head 扫描
# =============================================
#
# 对标单语言 Cell 6：
#   单语言：patch DE clean → DE corr，全序列对齐
#   跨语言：patch EN clean → DE corr，仅在 syntactic role 位置对齐
#           subj 位置：EN clean 的 subj_tok_clean → DE corr 的 subj_tok_corr
#
# 三组条件（对标单语言的 Positive/Negative/Position Control）：
#   Positive : EN clean acts → DE corr，subj 位置
#   Negative : EN corr  acts → DE corr，subj 位置
#   Position : EN clean acts → DE corr，random 位置

def make_cl_patch_hook(l_idx, h_idx, src_acts, pos_pairs, n_heads, head_dim):
    """
    跨语言 single-head, single-position patching hook。
    对标单语言 patch_hook，但只替换 pos_pairs 指定的位置。
    pos_pairs: list of (src_pos, tgt_pos)
    """
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        B, T_tgt, D = hidden.shape
        src_h = src_acts[f"layer_{l_idx}"].reshape(1, -1, n_heads, head_dim)
        out_h = hidden.reshape(B, T_tgt, n_heads, head_dim).clone()
        for sp, tp in pos_pairs:
            if sp is not None and tp is not None \
                    and sp < src_h.shape[1] and tp < T_tgt:
                out_h[:, tp, h_idx, :] = src_h[:, sp, h_idx, :]
        new_out = out_h.reshape(B, T_tgt, D)
        return (new_out,) + output[1:] if isinstance(output, tuple) else new_out
    return hook


def run_patching_scan(filtered, filtered_dataset, src_acts_key, patch_positions, label):
    """
    对标单语言 Section 5 的双层 for loop，
    返回 logit_boost_heatmap [n_layers, n_heads]。

    patch_positions: "subj" | "verb" | "both" | "random"
    src_acts_key   : "en_clean_acts" | "en_corr_acts"
    """
    print(f"\n🚀 开始 Cross-lingual Patching 扫描 [{label}] ({n_layers}x{n_heads} heads)...")
    boost_heatmap = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        for h in range(n_heads):
            total_boost = 0.0

            for r, item in zip(filtered, filtered_dataset):
                src_tok  = item["_tok_en"]
                tgt_tok  = item["_tok_de"]
                src_acts = r[src_acts_key]
                tgt_inp  = r["de_corr_inputs"]
                tgt_text = item["de_corr"]
                baseline = r["de_corr_logit"]

                if baseline is None:
                    continue

                # 构建位置对
                if patch_positions == "subj":
                    pos_pairs = [(src_tok["subj_tok_clean"], tgt_tok["subj_tok_corr"])]
                elif patch_positions == "verb":
                    pos_pairs = [(src_tok["verb_tok_clean"], tgt_tok["verb_tok_corr"])]
                elif patch_positions == "both":
                    pos_pairs = [
                        (src_tok["subj_tok_clean"], tgt_tok["subj_tok_corr"]),
                        (src_tok["verb_tok_clean"], tgt_tok["verb_tok_corr"]),
                    ]
                elif patch_positions == "random":
                    forbidden  = {tgt_tok["subj_tok_corr"], tgt_tok["verb_tok_corr"]}
                    tgt_len    = tgt_inp["input_ids"].shape[1]
                    candidates = [i for i in range(1, tgt_len - 1) if i not in forbidden]
                    rp = candidates[len(candidates) // 2] if candidates else 1
                    pos_pairs  = [(rp, rp)]
                else:
                    raise ValueError(f"Unknown patch_positions: {patch_positions}")

                handle = model.encoder.encoder.layer[l].attention.self \
                    .register_forward_hook(
                        make_cl_patch_hook(l, h, src_acts, pos_pairs, n_heads, head_dim)
                    )
                try:
                    with torch.no_grad():
                        out = model(**tgt_inp)
                        arc = out[0] if isinstance(out, (tuple, list)) else out.arc_scores
                    arc = apply_first_subword_mask(arc, tokenizer, tgt_text, device)
                    patched_logit = get_arc_logit(
                        arc, tgt_tok["subj_tok_corr"], tgt_tok["verb_tok_corr"]
                    )
                    if patched_logit is not None:
                        total_boost += (patched_logit - baseline)
                finally:
                    handle.remove()

            boost_heatmap[l, h] = total_boost / len(filtered_dataset)

        print(f"  Layer {l:2d} 完成 | Max Logit Boost: {np.max(boost_heatmap[l]):+.4f}")

    print(f"\n✅ [{label}] Patching 扫描完成")
    return boost_heatmap


# 三组扫描

# =============================================
# Subj-anchored Cross-lingual Patching
# =============================================

# boost_pos  = run_patching_scan(filtered, filtered_dataset,
#                                "en_clean_acts", "subj",   "Positive (EN_clean→DE, subj)")
# boost_neg  = run_patching_scan(filtered, filtered_dataset,
#                                "en_corr_acts",  "subj",   "Negative (EN_corr→DE, subj)")
# boost_ctrl = run_patching_scan(filtered, filtered_dataset,
#                                "en_clean_acts", "random", "Position Control (EN_clean→DE, random)")

# specificity   = boost_pos - boost_neg
# pos_ctrl_diff = boost_pos - boost_ctrl

# =============================================
# Verb-anchored Cross-lingual Patching
# =============================================

boost_pos  = run_patching_scan(
    filtered, filtered_dataset,
    "en_clean_acts", "verb",
    "Positive (EN_clean→DE, verb)"
)

boost_neg  = run_patching_scan(
    filtered, filtered_dataset,
    "en_corr_acts", "verb",
    "Negative (EN_corr→DE, verb)"
)

boost_ctrl = run_patching_scan(
    filtered, filtered_dataset,
    "en_clean_acts", "random",
    "Position Control (EN_clean→DE, random, verb baseline)"
)

# 对应指标（完全对齐 subj）
specificity   = boost_pos - boost_neg
pos_ctrl_diff = boost_pos - boost_ctrl


# =============================================
# 6. Patching 热力图 + Top-K Head 识别
# =============================================
#
# 对标单语言 Cell 7：
#   单语言：1 张 patching heatmap
#   跨语言：3 张（Positive / Specificity / Pos-Ctrl diff）

# 主 heatmap（Positive）
fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(
    boost_pos, annot=True, fmt=".2f",
    cmap="YlGnBu", ax=ax,
    cbar_kws={"label": "Avg Logit Boost (EN_clean→DE, subj pos)"}
)
ax.set_title("Cross-lingual Activation Patching: Positive Control\n"
             "(EN clean → DE corr, subj position, attention.self)")
ax.set_xlabel("Attention Head Index")
ax.set_ylabel("Layer Index")
plt.tight_layout()
plt.savefig("cl_patching_heatmap_positive.png", dpi=150)
plt.show()

# Negative Control heatmap
fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(
    boost_neg, annot=True, fmt=".2f",
    cmap="YlOrRd", ax=ax,
    cbar_kws={"label": "Avg Logit Boost (EN_corr→DE, subj pos)"}
)
ax.set_title("Cross-lingual Activation Patching: Negative Control\n"
             "(EN corr → DE corr, subj position, attention.self)")
ax.set_xlabel("Attention Head Index")
ax.set_ylabel("Layer Index")
plt.tight_layout()
plt.savefig("cl_patching_heatmap_negative.png", dpi=150)
plt.show()

# Specificity heatmap（对标单语言没有此步，但跨语言必须有控制）
fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(
    specificity, annot=True, fmt=".2f",
    cmap="RdBu", ax=ax,
    cbar_kws={"label": "Specificity = Positive − Negative"}
)
ax.set_title("Cross-lingual Patching: Specificity (Positive − Negative)")
ax.set_xlabel("Attention Head Index")
ax.set_ylabel("Layer Index")
plt.tight_layout()
plt.savefig("cl_patching_heatmap_specificity.png", dpi=150)
plt.show()

# Position Control diff heatmap
fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(
    pos_ctrl_diff, annot=True, fmt=".2f",
    cmap="RdBu", ax=ax,
    cbar_kws={"label": "Positive − Position Control"}
)
ax.set_title("Cross-lingual Patching: Position Specificity (Positive − Random)")
ax.set_xlabel("Attention Head Index")
ax.set_ylabel("Layer Index")
plt.tight_layout()
plt.savefig("cl_patching_heatmap_pos_ctrl.png", dpi=150)
plt.show()

# Top-K Candidate Heads（对标单语言完全一致）
TOP_K = 20
flat_idx     = np.argsort(boost_pos.flatten())[::-1]
candidate_heads = [(int(i // n_heads), int(i % n_heads)) for i in flat_idx[:TOP_K]]

print(f"Top-{TOP_K} Candidate Heads (CL Patching / Sufficiency):")
print("-" * 65)
for rank, (l, h) in enumerate(candidate_heads):
    boost = boost_pos[l, h]
    spec  = specificity[l, h]
    ctrl  = pos_ctrl_diff[l, h]
    print(f"  Rank {rank+1:2d}: Layer {l:2d}, Head {h:2d} | "
          f"Boost: {boost:+.4f}  Spec: {spec:+.4f}  Pos-Ctrl: {ctrl:+.4f}")
print("-" * 65)


# =============================================
# 7. Mean Ablation — 预计算均值激活
# =============================================
#
# 对标单语言 Cell 8：完全一致
# 在 DE clean 句上计算 per-head 均值激活（attention.self 输出空间）

print("📦 预计算 mean activations (attention.self, per-head)...")

mean_acts = compute_mean_activations(
    model, tokenizer, filtered_dataset, lang="de",
    n_layers=n_layers, n_heads=n_heads, head_dim=head_dim, device=device
)

print(f"✅ Mean activations 预计算完成，shape 示例: {mean_acts[0].shape}")
print(f"   (n_heads={n_heads}, head_dim={head_dim})")

all_de_clean_logits = [r["de_clean_logit"] for r in filtered]
print(f"DE clean baseline mean logit: {np.mean(all_de_clean_logits):.4f}")


# =============================================
# 8. Mean Ablation 扫描
# =============================================
#
# 对标单语言 Cell 9：完全一致
# 在 DE clean 句上消融，metric = logit drop

print(f"\n🚀 开始 Mean Ablation 扫描 ({n_layers}x{n_heads} heads)...")

ablation_heatmap = np.zeros((n_layers, n_heads))

for l in range(n_layers):
    for h in range(n_heads):
        total_drop = 0.0

        for r, item in zip(filtered, filtered_dataset):
            clean_text = item["de_clean"]
            clean_inp  = tokenizer(clean_text, return_tensors="pt").to(device)
            tok        = item["_tok_de"]
            base_logit = r["de_clean_logit"]

            mean_val = mean_acts[l][h]   # [head_dim]

            def mean_ablation_hook(module, input, output, _h=h, _mean=mean_val):
                hidden = output[0] if isinstance(output, tuple) else output
                B, T, D = hidden.shape
                out_h = hidden.reshape(B, T, n_heads, head_dim).clone()
                out_h[:, :, _h, :] = _mean
                new_out = out_h.reshape(B, T, D)
                return (new_out,) + output[1:] if isinstance(output, tuple) else new_out

            handle = model.encoder.encoder.layer[l].attention.self \
                         .register_forward_hook(mean_ablation_hook)
            try:
                with torch.no_grad():
                    out = model(**clean_inp)
                    arc = out[0] if isinstance(out, (tuple, list)) else out.arc_scores
                arc = apply_first_subword_mask(arc, tokenizer, clean_text, device)
                abl_logit = get_arc_logit(arc, tok["subj_tok_clean"], tok["verb_tok_clean"])
                if abl_logit is not None:
                    total_drop += (base_logit - abl_logit)
            finally:
                handle.remove()

        ablation_heatmap[l, h] = total_drop / len(filtered_dataset)

    print(f"  Layer {l:2d} 完成 | Max Logit Drop: {np.max(ablation_heatmap[l]):.4f}")

print("\n✅ Mean Ablation 扫描完成")


# =============================================
# 9. Ablation 热力图 + Joint Score
# =============================================
#
# 对标单语言 Cell 10：
#   单语言：patching vs ablation 2-panel
#   跨语言：用 Positive boost 替代 patching boost，其余完全一致

fig, axes = plt.subplots(1, 2, figsize=(24, 9))

sns.heatmap(boost_pos, annot=True, fmt=".2f", cmap="YlGnBu",
            ax=axes[0], cbar_kws={"label": "Avg Logit Boost (CL Patching, Positive)"})
axes[0].set_title("CL Patching: Sufficiency (Logit Boost, attention.self)")
axes[0].set_xlabel("Head"); axes[0].set_ylabel("Layer")

sns.heatmap(ablation_heatmap, annot=True, fmt=".2f", cmap="YlOrRd",
            ax=axes[1], cbar_kws={"label": "Avg Logit Drop (Mean Ablation)"})
axes[1].set_title("Ablation: Necessity (Mean Ablation, attention.self)")
axes[1].set_xlabel("Head"); axes[1].set_ylabel("Layer")

plt.tight_layout()
plt.savefig("cl_patching_ablation_comparison.png", dpi=150)
plt.show()

# Top-K Essential Heads（对标单语言完全一致）
flat_abl      = np.argsort(ablation_heatmap.flatten())[::-1]
essential_heads = [(int(i // n_heads), int(i % n_heads)) for i in flat_abl[:TOP_K]]

print(f"\nTop-{TOP_K} Essential Heads (Mean Ablation, attention.self):")
print("-" * 55)
for rank, (l, h) in enumerate(essential_heads):
    drop = ablation_heatmap[l, h]
    print(f"  Rank {rank+1:2d}: Layer {l:2d}, Head {h:2d} | Avg Drop: {drop:.4f}")
print("-" * 55)

overlap = set(candidate_heads) & set(essential_heads)
print(f"\n🎯 CL Patching + Ablation 双重确认的 Head: {sorted(overlap)}")

# Joint Score（与单语言完全一致：normalized patch + normalized abl）
patch_norm = (boost_pos - boost_pos.min()) / \
             (boost_pos.max() - boost_pos.min() + 1e-8)
abl_norm   = (ablation_heatmap - ablation_heatmap.min()) / \
             (ablation_heatmap.max() - ablation_heatmap.min() + 1e-8)
joint_score = patch_norm + abl_norm

flat_joint = np.argsort(joint_score.flatten())[::-1]
joint_top  = [(int(i // n_heads), int(i % n_heads)) for i in flat_joint[:TOP_K]]

print(f"\nJoint Top-{TOP_K} (Sufficient + Necessary):")
print("-" * 70)
for rank, (l, h) in enumerate(joint_top[:20]):
    print(f"  {rank+1:2d}. L{l:2d}H{h:2d} | "
          f"patch={boost_pos[l,h]:+.4f} | "
          f"abl={ablation_heatmap[l,h]:.4f} | "
          f"joint={joint_score[l,h]:.4f}")

plt.figure(figsize=(12, 9))
sns.heatmap(joint_score, annot=True, fmt=".2f", cmap="PuBuGn",
            cbar_kws={"label": "Joint Score (normalized patch + abl)"})
plt.title("Joint Score: Sufficient & Necessary Heads (Cross-lingual)")
plt.xlabel("Head"); plt.ylabel("Layer")
plt.tight_layout()
plt.savefig("cl_joint_score.png", dpi=150)
plt.show()


# =============================================
# 10. 组合回路搜索
# =============================================
#
# 对标单语言 Cell 11：完全一致
# pool 构建：优先 overlap（sufficient AND necessary），不足则用 joint top-k
# patching metric：将 EN clean 激活注入 DE corr 的 subj 位置（与单语言 patch 对称）

MAX_SIZE  = 6
POOL_SIZE = 10

pool = sorted(overlap) if len(overlap) >= 3 else joint_top[:POOL_SIZE]
print(f"🎯 候选池 ({len(pool)} 个): {pool}")

combination_results = []

for r_size in range(1, MAX_SIZE + 1):
    sub_circuits = list(itertools.combinations(pool, r_size))
    print(f"  评估大小={r_size} 的组合 ({len(sub_circuits)} 种)...")

    for circuit in sub_circuits:
        layer2heads = {}
        for l, h in circuit:
            layer2heads.setdefault(l, []).append(h)

        total_boost = 0.0

        for r, item in zip(filtered, filtered_dataset):
            src_tok  = item["_tok_en"]
            tgt_tok  = item["_tok_de"]
            src_acts = r["en_clean_acts"]
            tgt_inp  = r["de_corr_inputs"]
            tgt_text = item["de_corr"]
            baseline = r["de_corr_logit"]

            if baseline is None:
                continue

            pos_pairs = [(src_tok["subj_tok_clean"], tgt_tok["subj_tok_corr"])]

            def make_multi_patch_hook(l_idx, h_list, _src=src_acts, _pp=pos_pairs):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    B, T_tgt, D = hidden.shape
                    src_h = _src[f"layer_{l_idx}"].reshape(1, -1, n_heads, head_dim)
                    out_h = hidden.reshape(B, T_tgt, n_heads, head_dim).clone()
                    for hh in h_list:
                        for sp, tp in _pp:
                            if sp is not None and tp is not None \
                                    and sp < src_h.shape[1] and tp < T_tgt:
                                out_h[:, tp, hh, :] = src_h[:, sp, hh, :]
                    new_out = out_h.reshape(B, T_tgt, D)
                    return (new_out,) + output[1:] if isinstance(output, tuple) else new_out
                return hook

            handles = [
                model.encoder.encoder.layer[l_idx].attention.self
                    .register_forward_hook(make_multi_patch_hook(l_idx, h_list))
                for l_idx, h_list in layer2heads.items()
            ]
            try:
                with torch.no_grad():
                    out = model(**tgt_inp)
                    arc = out[0] if isinstance(out, (tuple, list)) else out.arc_scores
                arc = apply_first_subword_mask(arc, tokenizer, tgt_text, device)
                pl = get_arc_logit(arc, tgt_tok["subj_tok_corr"], tgt_tok["verb_tok_corr"])
                if pl is not None:
                    total_boost += (pl - baseline)
            finally:
                for hh in handles:
                    hh.remove()

        combination_results.append({
            "circuit":   circuit,
            "size":      r_size,
            "avg_boost": total_boost / len(filtered_dataset),
        })

combination_results.sort(key=lambda x: x["avg_boost"], reverse=True)

print("\n" + "=" * 60)
print("🏆 Top-10 Sub-Circuits (by Avg CL Logit Boost)")
print("=" * 60)
for i, res in enumerate(combination_results[:10]):
    print(f"  Rank {i+1} [Size {res['size']}]: {res['circuit']} | "
          f"Boost: {res['avg_boost']:+.4f}")


# =============================================
# 11. 最优回路的必要性验证（Circuit Ablation）
# =============================================
#
# 对标单语言 Cell 12：完全一致
# ablation 在 DE clean 句上，消融 circuit heads 后观察 logit drop

def make_abl_hook(h_list, layer_idx):
    """在 attention.self 输出空间，将指定 heads 替换为均值。与单语言完全一致。"""
    mean_val_layer = mean_acts[layer_idx]

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        B, T, D = hidden.shape
        out_h = hidden.reshape(B, T, n_heads, head_dim).clone()
        for hh in h_list:
            out_h[:, :, hh, :] = mean_val_layer[hh]
        new_out = out_h.reshape(B, T, D)
        return (new_out,) + output[1:] if isinstance(output, tuple) else new_out
    return hook


def evaluate_ablation(circuit, name=""):
    """
    对 circuit 做 mean ablation，在 DE clean 句上评估 logit drop。
    与单语言 evaluate_ablation 完全一致：
    drop_pct = (clean_logit - ablated_logit) / |agreement_delta| * 100
    """
    layer2heads = {}
    for l, h in circuit:
        layer2heads.setdefault(l, []).append(h)

    total_clean   = 0.0
    total_corr    = 0.0
    total_ablated = 0.0

    for r, item in zip(filtered, filtered_dataset):
        clean_text = item["de_clean"]
        clean_inp  = tokenizer(clean_text, return_tensors="pt").to(device)
        tok        = item["_tok_de"]

        total_clean += r["de_clean_logit"]
        total_corr  += r["de_corr_logit"]

        handles = [
            model.encoder.encoder.layer[l_idx].attention.self
                .register_forward_hook(make_abl_hook(h_list, l_idx))
            for l_idx, h_list in layer2heads.items()
        ]
        try:
            with torch.no_grad():
                out = model(**clean_inp)
                arc = out[0] if isinstance(out, (tuple, list)) else out.arc_scores
            arc = apply_first_subword_mask(arc, tokenizer, clean_text, device)
            abl_logit = get_arc_logit(arc, tok["subj_tok_clean"], tok["verb_tok_clean"])
            if abl_logit is not None:
                total_ablated += abl_logit
        finally:
            for hh in handles:
                hh.remove()

    n           = len(filtered_dataset)
    avg_clean   = total_clean   / n
    avg_corr    = total_corr    / n
    avg_ablated = total_ablated / n
    avg_delta   = avg_clean - avg_corr
    drop        = avg_clean - avg_ablated
    drop_pct    = drop / abs(avg_delta) * 100 if avg_delta != 0 else 0.0

    print("\n" + "=" * 60)
    print(f"📊 Ablation Result: {name}")
    print("=" * 60)
    print(f"  DE Clean logit : {avg_clean:.4f}")
    print(f"  DE Corr  logit : {avg_corr:.4f}")
    print(f"  Baseline delta : {avg_delta:.4f}")
    print(f"  Ablated logit  : {avg_ablated:.4f}")
    print(f"  Logit drop     : {drop:.4f}")
    print(f"  相对下降(÷Δ)  : {drop_pct:.2f}%")

    return {"name": name, "drop": drop, "drop_pct": drop_pct,
            "avg_clean": avg_clean, "avg_ablated": avg_ablated}


# 1. Best Circuit Ablation
best_circuit = list(combination_results[0]["circuit"])
print(f"\n🚀 Best CL Circuit: {best_circuit}")
res_best = evaluate_ablation(best_circuit, name="Best CL Circuit")

# 2. Top-K Heads Ablation（整个候选池）
print(f"\n🔥 Top-{len(pool)} Heads (整个候选池): {pool}")
res_topk = evaluate_ablation(pool, name=f"Top-{len(pool)} Heads (Pool)")

# 3. Leave-One-Out 分析（对标单语言完全一致）
print("\n🧠 Leave-One-Out Analysis")
loo_results = []
for i in range(len(best_circuit)):
    sub_circuit = [h for j, h in enumerate(best_circuit) if j != i]
    removed = best_circuit[i]
    res = evaluate_ablation(sub_circuit, name=f"Remove {removed}")
    loo_results.append((removed, res["drop_pct"]))

print("\n" + "=" * 60)
print("🏆 Critical Heads (Leave-One-Out，按重要性排序)")
print("=" * 60)
loo_results.sort(key=lambda x: x[1], reverse=True)
for i, (head, drop_pct) in enumerate(loo_results):
    print(f"  Rank {i+1}: Remove {head} → Remaining Drop {drop_pct:.2f}%")

# 4. Top-N Circuits Comparison（对标单语言完全一致）
print("\n🔬 Top-N Circuits Comparison")
topN = 5
multi_results = []
for i in range(min(topN, len(combination_results))):
    circuit = list(combination_results[i]["circuit"])
    print(f"\n🚀 Evaluating Top-{i+1} Circuit: {circuit}")
    res = evaluate_ablation(circuit, name=f"Top-{i+1} Circuit")
    multi_results.append((i + 1, res["drop_pct"]))

print("\n" + "=" * 60)
print("📊 Multi-Circuit Comparison")
print("=" * 60)
for rank, drop_pct in multi_results:
    print(f"  Rank {rank}: Drop {drop_pct:.2f}%")


# =============================================
# 12. Cross-ablation：验证 Causal Role 跨语言共享
# =============================================
#
# 单语言无此步；跨语言独有：
#   ① 用 EN 单语言 best circuit → 消融 DE clean 句（DE 模型）
#      下降 >> 0 说明 EN heads 对 DE 同样 necessary
#   ② 用 DE 单语言 best circuit → 消融 EN clean 句（EN 模型）
#      下降 >> 0 说明 DE heads 对 EN 同样 necessary
#   ③ 随机 6 个 heads 做对照

# 从单语言实验填入
MONO_EN_BEST_CIRCUIT = [(1, 8), (6, 8), (7, 4), (7, 5), (8, 1), (8, 9)]
MONO_DE_BEST_CIRCUIT = [(7, 10), (8, 11), (9, 7), (9, 9), (9, 11), (11, 11)]

import random
random.seed(42)
all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
excluded  = set(MONO_EN_BEST_CIRCUIT) | set(MONO_DE_BEST_CIRCUIT) | set(best_circuit)
random_ctrl_heads = random.sample(
    [x for x in all_heads if x not in excluded], k=6
)

print("\n" + "=" * 70)
print("📐 Cross-ablation: Causal Role 跨语言验证")
print("=" * 70)
print(f"  EN mono best circuit : {MONO_EN_BEST_CIRCUIT}")
print(f"  DE mono best circuit : {MONO_DE_BEST_CIRCUIT}")
print(f"  CL best circuit      : {best_circuit}")
print(f"  Random control       : {random_ctrl_heads}")


def evaluate_ablation_en(circuit, name=""):
    """
    在 EN clean 句 + EN 模型上做 mean ablation。
    对标 evaluate_ablation，但使用 model_en、EN mean acts、EN tok。
    """
    # 预计算 EN mean activations（用 filtered_en 的 en_clean 句）
    # 只在第一次调用时计算一次
    if not hasattr(evaluate_ablation_en, "_mean_acts_en"):
        print("  📦 预计算 EN mean activations...")
        evaluate_ablation_en._mean_acts_en = compute_mean_activations(
            model_en, tokenizer,
            [r["item"] for r in filtered_en],
            lang="en",
            n_layers=n_layers, n_heads=n_heads, head_dim=head_dim, device=device
        )
        print(f"  ✅ EN mean activations 完成，shape: {evaluate_ablation_en._mean_acts_en[0].shape}")

    mean_acts_en = evaluate_ablation_en._mean_acts_en

    def make_abl_hook_en(h_list, layer_idx):
        mean_val_layer = mean_acts_en[layer_idx]
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            B, T, D = hidden.shape
            out_h = hidden.reshape(B, T, n_heads, head_dim).clone()
            for hh in h_list:
                out_h[:, :, hh, :] = mean_val_layer[hh]
            new_out = out_h.reshape(B, T, D)
            return (new_out,) + output[1:] if isinstance(output, tuple) else new_out
        return hook

    layer2heads = {}
    for l, h in circuit:
        layer2heads.setdefault(l, []).append(h)

    total_clean   = 0.0
    total_corr    = 0.0
    total_ablated = 0.0
    n_valid       = 0

    for r in filtered_en:
        item = r["item"]
        tok  = item["_tok_en"]
        if tok["subj_tok_clean"] is None or tok["verb_tok_clean"] is None:
            continue

        clean_text = item["en_clean"]
        clean_inp  = tokenizer(clean_text, return_tensors="pt").to(device)

        total_clean += r["en_clean_logit"]
        total_corr  += r["en_corr_logit"]

        handles = [
            model_en.encoder.encoder.layer[l_idx].attention.self
                .register_forward_hook(make_abl_hook_en(h_list, l_idx))
            for l_idx, h_list in layer2heads.items()
        ]
        try:
            with torch.no_grad():
                out = model_en(**clean_inp)
                arc = out[0] if isinstance(out, (tuple, list)) else out.arc_scores
            arc = apply_first_subword_mask(arc, tokenizer, clean_text, device)
            abl_logit = get_arc_logit(arc, tok["subj_tok_clean"], tok["verb_tok_clean"])
            if abl_logit is not None:
                total_ablated += abl_logit
                n_valid += 1
        finally:
            for hh in handles:
                hh.remove()

    if n_valid == 0:
        print(f"  [{name}] 无有效样本")
        return None

    avg_clean   = total_clean   / n_valid
    avg_corr    = total_corr    / n_valid
    avg_ablated = total_ablated / n_valid
    avg_delta   = avg_clean - avg_corr
    drop        = avg_clean - avg_ablated
    drop_pct    = drop / abs(avg_delta) * 100 if avg_delta != 0 else 0.0

    print("\n" + "=" * 60)
    print(f"📊 EN Ablation Result: {name}")
    print("=" * 60)
    print(f"  EN Clean logit : {avg_clean:.4f}")
    print(f"  EN Corr  logit : {avg_corr:.4f}")
    print(f"  Baseline delta : {avg_delta:.4f}")
    print(f"  Ablated logit  : {avg_ablated:.4f}")
    print(f"  Logit drop     : {drop:.4f}")
    print(f"  相对下降(÷Δ)  : {drop_pct:.2f}%")

    return {"name": name, "drop": drop, "drop_pct": drop_pct,
            "avg_clean": avg_clean, "avg_ablated": avg_ablated}


# ① EN mono best circuit → DE ablation（DE 模型）
print("\n--- ① EN mono circuit → DE ablation ---")
res_en_on_de   = evaluate_ablation(MONO_EN_BEST_CIRCUIT, name="EN mono → DE abl")
res_rand_on_de = evaluate_ablation(random_ctrl_heads,    name="Random   → DE abl [对照]")

# ② DE mono best circuit → EN ablation（EN 模型）
print("\n--- ② DE mono circuit → EN ablation ---")
res_de_on_en   = evaluate_ablation_en(MONO_DE_BEST_CIRCUIT, name="DE mono → EN abl")
res_rand_on_en = evaluate_ablation_en(random_ctrl_heads,    name="Random   → EN abl [对照]")

# ③ CL best circuit 在 DE / EN 双侧验证
print("\n--- ③ CL best circuit → DE ablation & EN ablation ---")
res_cl_on_de = evaluate_ablation(best_circuit,    name="CL best → DE abl")
res_cl_on_en = evaluate_ablation_en(best_circuit, name="CL best → EN abl")

# 汇总表
def _pct(r):
    return f"{r['drop_pct']:+.2f}%" if r else "—"

print("\n" + "=" * 70)
print("📋 Cross-ablation 汇总")
print("=" * 70)
print(f"  {'实验':<40} {'DE drop%':>10} {'EN drop%':>10}")
print("  " + "─" * 62)
print(f"  {'EN mono → DE abl (DE model)':<40} {_pct(res_en_on_de):>10} {'—':>10}")
print(f"  {'DE mono → EN abl (EN model)':<40} {'—':>10} {_pct(res_de_on_en):>10}")
print(f"  {'CL best → DE abl (DE model)':<40} {_pct(res_cl_on_de):>10} {'—':>10}")
print(f"  {'CL best → EN abl (EN model)':<40} {'—':>10} {_pct(res_cl_on_en):>10}")
print(f"  {'Random  → DE abl [对照]':<40} {_pct(res_rand_on_de):>10} {'—':>10}")
print(f"  {'Random  → EN abl [对照]':<40} {'—':>10} {_pct(res_rand_on_en):>10}")
print("\n💡 解读：drop% >> random → heads 在对方语言上 necessary（共享 causal role）")


# =============================================
# 13. 综合可视化（对标单语言 Cell 13 综合图）
# =============================================

fig = plt.figure(figsize=(28, 20))
gs  = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

# ── Row 0: 三张核心热力图 ──
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(boost_pos, annot=True, fmt=".2f", cmap="YlGnBu",
            ax=ax1, cbar_kws={"label": "Avg Boost"})
ax1.set_title("CL Positive Boost (EN_clean→DE, subj)")
ax1.set_xlabel("Head"); ax1.set_ylabel("Layer")

ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(specificity, annot=True, fmt=".2f", cmap="RdBu",
            ax=ax2, cbar_kws={"label": "Specificity"})
ax2.set_title("Specificity (Positive − Negative)")
ax2.set_xlabel("Head"); ax2.set_ylabel("Layer")

ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(joint_score, annot=True, fmt=".2f", cmap="PuBuGn",
            ax=ax3, cbar_kws={"label": "Joint Score"})
ax3.set_title("Joint Score (normalized patch + abl)")
ax3.set_xlabel("Head"); ax3.set_ylabel("Layer")

# ── Row 1 左: Top-N circuit ablation drop 柱状图 ──
ax4 = fig.add_subplot(gs[1, :2])
circ_labels = [f"Top-{r}" for r, _ in multi_results]
circ_drops  = [dp for _, dp in multi_results]
colors4 = ["steelblue" if dp > 0 else "tomato" for dp in circ_drops]
bars4 = ax4.bar(circ_labels, circ_drops, color=colors4)
ax4.axhline(0, color="black", linewidth=0.8)
ax4.set_ylabel("DE Ablation Drop (%)")
ax4.set_title("Top-N CL Circuits: DE Ablation Validation")
for bar, dp in zip(bars4, circ_drops):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3, f"{dp:.1f}%", ha="center", fontsize=9)

# ── Row 1 右: Leave-One-Out 柱状图 ──
ax5 = fig.add_subplot(gs[1, 2])
loo_labels_plot = [f"Remove\nL{l}H{h}" for (l, h), _ in loo_results]
loo_drops_plot  = [dp for _, dp in loo_results]
colors5 = ["steelblue" if dp > 0 else "tomato" for dp in loo_drops_plot]
bars5 = ax5.bar(loo_labels_plot, loo_drops_plot, color=colors5)
ax5.axhline(0, color="black", linewidth=0.8)
ax5.set_ylabel("Remaining Drop (%)")
ax5.set_title("Leave-One-Out: Critical Heads (CL Circuit)")
for bar, dp in zip(bars5, loo_drops_plot):
    ax5.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3, f"{dp:.1f}%", ha="center", fontsize=8)

# ── Row 2 左: Cross-ablation 柱状图 ──
ax6 = fig.add_subplot(gs[2, :2])
cross_labels = [
    "EN mono\n→DE [DE mdl]",
    "DE mono\n→EN [EN mdl]",
    "CL best\n→DE [DE mdl]",
    "CL best\n→EN [EN mdl]",
    "Random\n→DE [对照]",
    "Random\n→EN [对照]",
]
cross_drops = [
    res_en_on_de["drop_pct"]   if res_en_on_de   else 0,
    res_de_on_en["drop_pct"]   if res_de_on_en   else 0,
    res_cl_on_de["drop_pct"]   if res_cl_on_de   else 0,
    res_cl_on_en["drop_pct"]   if res_cl_on_en   else 0,
    res_rand_on_de["drop_pct"] if res_rand_on_de else 0,
    res_rand_on_en["drop_pct"] if res_rand_on_en else 0,
]
colors6 = ["steelblue", "seagreen", "steelblue", "seagreen", "gray", "gray"]
bars6 = ax6.bar(cross_labels, cross_drops, color=colors6)
ax6.axhline(0, color="black", linewidth=0.8)
ax6.set_ylabel("Ablation Drop (%)")
ax6.set_title("Cross-ablation: Causal Role Sharing (blue=DE model, green=EN model, gray=random)")
for bar, dp in zip(bars6, cross_drops):
    ax6.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3, f"{dp:.1f}%", ha="center", fontsize=9)

# ── Row 2 右: 文字汇总 ──
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis("off")
shared_cl_en = set(best_circuit) & set(MONO_EN_BEST_CIRCUIT)
shared_cl_de = set(best_circuit) & set(MONO_DE_BEST_CIRCUIT)
shared_all   = set(best_circuit) & set(MONO_EN_BEST_CIRCUIT) & set(MONO_DE_BEST_CIRCUIT)
summary = (
    f"Cross-lingual Circuit Summary\n"
    f"{'─' * 38}\n"
    f"Samples     : {len(filtered_dataset)}\n"
    f"Mean DE Δ   : {np.mean([r['de_delta'] for r in filtered]):.4f}\n\n"
    f"CL Pool ({len(pool)} heads):\n"
    f"  {pool}\n\n"
    f"Best CL Circuit:\n"
    f"  {best_circuit}\n"
    f"  DE Ablation Drop: {res_best['drop_pct']:.2f}%\n\n"
    f"CL ∩ EN mono : {sorted(shared_cl_en)}\n"
    f"CL ∩ DE mono : {sorted(shared_cl_de)}\n"
    f"CL ∩ EN ∩ DE : {sorted(shared_all)}\n\n"
    f"Cross-ablation:\n"
    f"  EN mono→DE : {_pct(res_en_on_de)}\n"
    f"  DE mono→EN : {_pct(res_de_on_en)}\n"
    f"  CL best→DE : {_pct(res_cl_on_de)}\n"
    f"  CL best→EN : {_pct(res_cl_on_en)}\n"
    f"  Random→DE  : {_pct(res_rand_on_de)}\n"
    f"  Random→EN  : {_pct(res_rand_on_en)}\n"
)
ax7.text(0.02, 0.98, summary, transform=ax7.transAxes,
         fontsize=8.5, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

plt.suptitle("Cross-lingual Parallel Comparison: Summary", fontsize=16, y=1.01)
plt.savefig("cl_parallel_summary.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================
# 14. 实验结论汇总
# =============================================
#
# 对标单语言 Cell 13：格式完全一致

print("=" * 70)
print("📋 Cross-lingual Parallel Comparison 实验结论")
print("=" * 70)

print(f"\n📊 数据统计:")
print(f"  总配对: {len(dataset)} | DE delta 可计算: {len(valid)} | "
      f"有效样本: {len(filtered)}")
print(f"  Mean DE agreement delta (filtered): "
      f"{np.mean([r['de_delta'] for r in filtered]):.4f}")
print(f"  Mean EN agreement delta (filtered_en): "
      f"{np.mean([r['en_delta'] for r in filtered_en]):.4f}")

print(f"\n🎯 候选池 ({len(pool)} 个 heads):")
for l, h in pool:
    tag = "✅[sufficient+necessary]" if (l, h) in overlap else \
          "🔵[sufficient]" if (l, h) in set(candidate_heads) else \
          "🟡[necessary]"
    print(f"  L{l:2d}H{h:2d} {tag} | "
          f"patch={boost_pos[l,h]:+.4f} | "
          f"abl={ablation_heatmap[l,h]:.4f} | "
          f"spec={specificity[l,h]:+.4f}")

print(f"\n🏆 Best CL Circuit: {best_circuit}")
print(f"  CL Patching Boost : {combination_results[0]['avg_boost']:+.4f}")
print(f"  DE Ablation Drop  : {res_best['drop_pct']:.2f}%")
print(f"  Pool Ablation Drop: {res_topk['drop_pct']:.2f}%")

print(f"\n🔗 与单语言 Circuit 重叠:")
print(f"  CL ∩ EN mono = {sorted(shared_cl_en)} ({len(shared_cl_en)} heads)")
print(f"  CL ∩ DE mono = {sorted(shared_cl_de)} ({len(shared_cl_de)} heads)")
print(f"  CL ∩ EN ∩ DE = {sorted(shared_all)} ({len(shared_all)} heads)")

print(f"\n📐 Cross-ablation 结论（causal role 跨语言共享性）:")
print(f"  EN mono → DE: {_pct(res_en_on_de)}  vs  Random: {_pct(res_rand_on_de)}")
print(f"  DE mono → EN: {_pct(res_de_on_en)}  vs  Random: {_pct(res_rand_on_en)}")
print(f"  CL best → DE: {_pct(res_cl_on_de)}  vs  Random: {_pct(res_rand_on_de)}")
print(f"  CL best → EN: {_pct(res_cl_on_en)}  vs  Random: {_pct(res_rand_on_en)}")

print(f"\n💡 Hook 位置: attention.self (pre-projection)")
print(f"   Patching 和 Ablation 使用相同 hook 位置，与单语言实验一致")