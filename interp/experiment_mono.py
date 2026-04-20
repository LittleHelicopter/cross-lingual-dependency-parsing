# ============================================================
# experiment_de.py
# DE 单语言 Activation Patching 实验
# 完整流程：Localization → Necessity → Circuit
#
# 对应原 notebook: test_de.ipynb
# ============================================================

import sys
import json
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
)


# =============================================
# 1. 环境配置与模型加载
# =============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "/mnt/ssd/weicheng/data_interns/yahan/syntactic-parsing/models/last4_de/best_model.pt"
DATA_PATH       = "../data/de_100_cleaned_data.jsonl"

print(f"正在从 {CHECKPOINT_PATH} 加载模型...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
args = checkpoint["args"]

tokenizer  = AutoTokenizer.from_pretrained(args["model_name"])
encoder    = AutoModel.from_pretrained(args["model_name"])
num_labels = checkpoint["model_state_dict"]["rel_biaffine.weight"].shape[0]

model = BiaffineDependencyParser(
    encoder=encoder,
    num_labels=num_labels,
    hidden_size=encoder.config.hidden_size,
    mlp_size=args["mlp_size"],
    dropout=args["dropout"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

n_layers = model.encoder.config.num_hidden_layers
n_heads  = model.encoder.config.num_attention_heads
head_dim = model.encoder.config.hidden_size // n_heads

print(f"模型加载成功！设备: {device} | layers={n_layers}, heads={n_heads}, head_dim={head_dim}")


# =============================================
# 2. 数据加载 + Token Index 解析
# =============================================

dataset = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line.strip()))
print(f"✅ 加载 {len(dataset)} 条数据")

errors = []
for item in dataset:
    res = resolve_token_indices(item, tokenizer, lang="de")
    item["_tok"] = res
    if any(v is None for v in res.values()):
        errors.append(f"id={item['id']} | {res} | clean='{item['de_clean'][:60]}'")

print(f"✅ Index 解析完成，{len(errors)} 条有问题：")
for e in errors[:10]:
    print(" ", e)

# 快速抽样校验
for item in dataset[:3]:
    tok = item["_tok"]
    clean_words = item["de_clean"].split()
    corr_words  = item["de_corr"].split()
    sw, vw = tok["subj_word_idx"], tok["verb_word_idx"]
    print(f"\nid={item['id']}")
    print(f"  subj word_idx={sw} -> '{clean_words[sw] if sw is not None else None}' "
          f"| subj_tok_clean={tok['subj_tok_clean']}")
    print(f"  verb word_idx={vw} -> "
          f"clean='{clean_words[vw] if vw is not None else None}' "
          f"corr='{corr_words[vw] if vw is not None else None}'")
    print(f"  verb_tok_clean={tok['verb_tok_clean']}, verb_tok_corr={tok['verb_tok_corr']}")


# =============================================
# 3. Baseline 评估
# =============================================

print("📊 计算 baseline agreement delta...")

results_baseline = []
for item in dataset:
    clean_logit, corr_logit, delta = get_agreement_delta(
        model, tokenizer, item, lang="de", device=device, tok_key="_tok"
    )
    results_baseline.append({
        "item":        item,
        "clean_logit": clean_logit,
        "corr_logit":  corr_logit,
        "delta":       delta,
    })

valid  = [r for r in results_baseline if r["delta"] is not None]
deltas = [r["delta"] for r in valid]

print(f"\n总样本: {len(dataset)} | 有效样本(index解析成功): {len(valid)}")
print(f"\n--- Baseline Agreement Delta ---")
print(f"Mean clean logit : {np.mean([r['clean_logit'] for r in valid]):.4f}")
print(f"Mean corr  logit : {np.mean([r['corr_logit']  for r in valid]):.4f}")
print(f"Mean delta       : {np.mean(deltas):.4f}")
print(f"delta > 0 比例   : {np.mean([d > 0 for d in deltas]):.2%}")

plt.figure(figsize=(8, 4))
plt.hist(deltas, bins=30, color="steelblue", edgecolor="white")
plt.axvline(0, color="red", linestyle="--", label="delta=0")
plt.xlabel("Agreement Delta (clean_logit - corr_logit)")
plt.ylabel("Count")
plt.title("Baseline Agreement Delta Distribution (DE)")
plt.legend()
plt.tight_layout()
plt.savefig("baseline_delta_de.png", dpi=150)
plt.show()

# 过滤策略：使用全部有效样本（delta <= 0 的样本也保留）
filtered         = [r for r in valid]
filtered_dataset = [r["item"] for r in filtered]

print(f"\n过滤后有效样本: {len(filtered_dataset)} 条")
print(f"Mean delta (filtered): {np.mean([r['delta'] for r in filtered]):.4f}")

# clean logit 分层分析
print("\n--- Clean Logit 分层 ---")
for t in [-5, -3, -2, -1, 0]:
    sub = [r for r in filtered if r["clean_logit"] > t]
    if sub:
        print(f"  clean_logit > {t:2d}: {len(sub):3d} 条 | "
              f"Mean delta = {np.mean([r['delta'] for r in sub]):.4f}")


# =============================================
# 4. 预计算激活与基准值
# =============================================

print("📦 预计算 filtered 数据集的激活与基准值...")

all_clean_acts  = []
all_corr_inputs = []
all_base_deltas = []   # corr 句下的 arc logit（patching 的起点）

for r in filtered:
    item = r["item"]
    c_inp = tokenizer(item["de_clean"], return_tensors="pt").to(device)
    all_clean_acts.append(get_activations(model, c_inp))

    r_inp = tokenizer(item["de_corr"], return_tensors="pt").to(device)
    all_corr_inputs.append(r_inp)

    all_base_deltas.append(r["corr_logit"])

print(f"✅ 预计算完成，共 {len(filtered_dataset)} 条")
print(f"Base corr logit mean: {np.mean(all_base_deltas):.4f}")


# =============================================
# 5. Activation Patching — 逐 Head 扫描
# Metric: patching 后 corr_logit 相对 baseline 的提升
# Hook 位置: attention.self（pre-projection）
# =============================================

print(f"🚀 开始 Activation Patching ({n_layers}x{n_heads} heads)...")

logit_boost_heatmap = np.zeros((n_layers, n_heads))

for l in range(n_layers):
    for h in range(n_heads):
        total_boost = 0.0

        for i, (r, item) in enumerate(zip(filtered, filtered_dataset)):
            clean_acts  = all_clean_acts[i]
            corr_inputs = all_corr_inputs[i]
            base_logit  = all_base_deltas[i]
            corr_text   = item["de_corr"]
            tok         = item["_tok"]

            def patch_hook(module, input, output,
                           _l=l, _h=h, _acts=clean_acts):
                hidden = output[0] if isinstance(output, tuple) else output
                B, T_corr, D = hidden.shape
                clean_hidden = _acts[f"layer_{_l}"]
                _, T_clean, _ = clean_hidden.shape
                out_h = hidden.reshape(B, T_corr, n_heads, head_dim).clone()
                c_h   = clean_hidden.reshape(1, T_clean, n_heads, head_dim)
                min_T = min(T_corr, T_clean)
                out_h[:, :min_T, _h, :] = c_h[:, :min_T, _h, :]
                new_out = out_h.reshape(B, T_corr, D)
                return (new_out,) + output[1:] if isinstance(output, tuple) else new_out

            handle = model.encoder.encoder.layer[l].attention.self \
                         .register_forward_hook(patch_hook)
            try:
                with torch.no_grad():
                    out = model(**corr_inputs)
                    arc = out[0] if isinstance(out, (tuple, list)) else out.arc_scores
                arc = apply_first_subword_mask(arc, tokenizer, corr_text, device)
                patched_logit = get_arc_logit(arc, tok["subj_tok_corr"], tok["verb_tok_corr"])
                if patched_logit is not None:
                    total_boost += (patched_logit - base_logit)
            finally:
                handle.remove()

        logit_boost_heatmap[l, h] = total_boost / len(filtered_dataset)

    print(f"  Layer {l:2d} 完成 | Max Logit Boost: {np.max(logit_boost_heatmap[l]):+.4f}")

print("\n✅ Patching 扫描完成")


# =============================================
# 6. Patching 热力图 + Top-K Head 识别
# =============================================

fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(
    logit_boost_heatmap, annot=True, fmt=".2f",
    cmap="YlGnBu", ax=ax,
    cbar_kws={"label": "Avg Logit Boost (corr→clean patching)"}
)
ax.set_title("Activation Patching: Subject-Verb Agreement Heads (DE)")
ax.set_xlabel("Attention Head Index")
ax.set_ylabel("Layer Index")
plt.tight_layout()
plt.savefig("patching_heatmap_de.png", dpi=150)
plt.show()

TOP_K = 20
flat_idx = np.argsort(logit_boost_heatmap.flatten())[::-1]
candidate_heads = [(int(i // n_heads), int(i % n_heads)) for i in flat_idx[:TOP_K]]

print(f"Top-{TOP_K} Candidate Heads (Patching / Sufficiency):")
print("-" * 55)
for rank, (l, h) in enumerate(candidate_heads):
    boost = logit_boost_heatmap[l, h]
    print(f"  Rank {rank+1:2d}: Layer {l:2d}, Head {h:2d} | Avg Boost: {boost:+.4f}")
print("-" * 55)


# =============================================
# 7. Mean Ablation — 预计算均值激活
# hook 在 attention.self，按 head 维度计算
# =============================================

print("📦 预计算 mean activations (attention.self, per-head)...")

mean_acts = compute_mean_activations(
    model, tokenizer, filtered_dataset, lang="de",
    n_layers=n_layers, n_heads=n_heads, head_dim=head_dim, device=device
)

print(f"✅ Mean activations 预计算完成，shape 示例: {mean_acts[0].shape}")
print(f"   (n_heads={n_heads}, head_dim={head_dim})")

all_clean_logits_f = [r["clean_logit"] for r in filtered]
print(f"Clean baseline mean logit: {np.mean(all_clean_logits_f):.4f}")


# =============================================
# 8. Mean Ablation 扫描
# =============================================

print(f"\n🚀 开始 Mean Ablation 扫描 ({n_layers}x{n_heads} heads)...")

ablation_heatmap = np.zeros((n_layers, n_heads))

for l in range(n_layers):
    for h in range(n_heads):
        total_drop = 0.0

        for i, (r, item) in enumerate(zip(filtered, filtered_dataset)):
            clean_text = item["de_clean"]
            clean_inp  = tokenizer(clean_text, return_tensors="pt").to(device)
            base_logit = all_clean_logits_f[i]
            tok        = item["_tok"]

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

fig, axes = plt.subplots(1, 2, figsize=(24, 9))

sns.heatmap(logit_boost_heatmap, annot=True, fmt=".2f", cmap="YlGnBu",
            ax=axes[0], cbar_kws={"label": "Avg Logit Boost (Patching)"})
axes[0].set_title("Patching: Sufficiency (Logit Boost, attention.self)")
axes[0].set_xlabel("Head"); axes[0].set_ylabel("Layer")

sns.heatmap(ablation_heatmap, annot=True, fmt=".2f", cmap="YlOrRd",
            ax=axes[1], cbar_kws={"label": "Avg Logit Drop (Mean Ablation)"})
axes[1].set_title("Ablation: Necessity (Mean Ablation, attention.self)")
axes[1].set_xlabel("Head"); axes[1].set_ylabel("Layer")

plt.tight_layout()
plt.savefig("patching_ablation_comparison_de.png", dpi=150)
plt.show()

# Top-K Essential Heads
flat_abl = np.argsort(ablation_heatmap.flatten())[::-1]
essential_heads = [(int(i // n_heads), int(i % n_heads)) for i in flat_abl[:TOP_K]]

print(f"\nTop-{TOP_K} Essential Heads (Mean Ablation, attention.self):")
print("-" * 55)
for rank, (l, h) in enumerate(essential_heads):
    drop = ablation_heatmap[l, h]
    print(f"  Rank {rank+1:2d}: Layer {l:2d}, Head {h:2d} | Avg Drop: {drop:.4f}")
print("-" * 55)

overlap = set(candidate_heads) & set(essential_heads)
print(f"\n🎯 Patching + Ablation 双重确认的 Head: {sorted(overlap)}")

# Joint Score（归一化后相加）
patch_norm = (logit_boost_heatmap - logit_boost_heatmap.min()) / \
             (logit_boost_heatmap.max() - logit_boost_heatmap.min() + 1e-8)
abl_norm   = (ablation_heatmap - ablation_heatmap.min()) / \
             (ablation_heatmap.max() - ablation_heatmap.min() + 1e-8)
joint_score = patch_norm + abl_norm

flat_joint = np.argsort(joint_score.flatten())[::-1]
joint_top  = [(int(i // n_heads), int(i % n_heads)) for i in flat_joint[:TOP_K]]

print(f"\nJoint Top-{TOP_K} (Sufficient + Necessary):")
print("-" * 70)
for rank, (l, h) in enumerate(joint_top[:20]):
    print(f"  {rank+1:2d}. L{l:2d}H{h:2d} | "
          f"patch={logit_boost_heatmap[l,h]:+.4f} | "
          f"abl={ablation_heatmap[l,h]:.4f} | "
          f"joint={joint_score[l,h]:.4f}")

plt.figure(figsize=(12, 9))
sns.heatmap(joint_score, annot=True, fmt=".2f", cmap="PuBuGn",
            cbar_kws={"label": "Joint Score (normalized patch + abl)"})
plt.title("Joint Score: Sufficient & Necessary Heads (DE)")
plt.xlabel("Head"); plt.ylabel("Layer")
plt.tight_layout()
plt.savefig("joint_score_de.png", dpi=150)
plt.show()


# =============================================
# 10. 组合回路搜索
# =============================================

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

        for i, (r, item) in enumerate(zip(filtered, filtered_dataset)):
            clean_acts  = all_clean_acts[i]
            corr_inputs = all_corr_inputs[i]
            base_logit  = all_base_deltas[i]
            corr_text   = item["de_corr"]
            tok         = item["_tok"]

            def make_patch_hook(l_idx, h_list, _acts=clean_acts):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    B, T_c, D = hidden.shape
                    clean_h = _acts[f"layer_{l_idx}"]
                    _, T_k, _ = clean_h.shape
                    out_h = hidden.reshape(B, T_c, n_heads, head_dim).clone()
                    c_h   = clean_h.reshape(1, T_k, n_heads, head_dim)
                    min_T = min(T_c, T_k)
                    for hh in h_list:
                        out_h[:, :min_T, hh, :] = c_h[:, :min_T, hh, :]
                    new_out = out_h.reshape(B, T_c, D)
                    return (new_out,) + output[1:] if isinstance(output, tuple) else new_out
                return hook

            handles = [
                model.encoder.encoder.layer[l_idx].attention.self
                    .register_forward_hook(make_patch_hook(l_idx, h_list))
                for l_idx, h_list in layer2heads.items()
            ]
            try:
                with torch.no_grad():
                    out = model(**corr_inputs)
                    arc = out[0] if isinstance(out, (tuple, list)) else out.arc_scores
                arc = apply_first_subword_mask(arc, tokenizer, corr_text, device)
                pl  = get_arc_logit(arc, tok["subj_tok_corr"], tok["verb_tok_corr"])
                if pl is not None:
                    total_boost += (pl - base_logit)
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
print("🏆 Top-10 Sub-Circuits (by Avg Logit Boost)")
print("=" * 60)
for i, res in enumerate(combination_results[:10]):
    print(f"  Rank {i+1} [Size {res['size']}]: {res['circuit']} | "
          f"Boost: {res['avg_boost']:+.4f}")


# =============================================
# 11. 最优回路的必要性验证（Circuit Ablation）
# =============================================

def make_abl_hook(h_list, layer_idx):
    """在 attention.self 输出空间，将指定 heads 替换为均值。"""
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
    对 circuit 做 mean ablation，在 clean 句上评估 logit drop。
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
        tok        = item["_tok"]

        total_clean += r["clean_logit"]
        total_corr  += r["corr_logit"]

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
    print(f"  Clean logit    : {avg_clean:.4f}")
    print(f"  Corr  logit    : {avg_corr:.4f}")
    print(f"  Baseline delta : {avg_delta:.4f}")
    print(f"  Ablated logit  : {avg_ablated:.4f}")
    print(f"  Logit drop     : {drop:.4f}")
    print(f"  相对下降(÷Δ)  : {drop_pct:.2f}%")

    return {"name": name, "drop": drop, "drop_pct": drop_pct}


# 1. Best Circuit Ablation
best_circuit = combination_results[0]["circuit"]
print(f"\n🚀 Best Circuit: {best_circuit}")
res_best = evaluate_ablation(best_circuit, name="Best Circuit")

# 2. Top-K Heads Ablation（整个候选池）
print(f"\n🔥 Top-{len(pool)} Heads (整个候选池): {pool}")
res_topk = evaluate_ablation(pool, name=f"Top-{len(pool)} Heads (Pool)")

# 3. Leave-One-Out 分析
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

# 4. Top-N Circuits Comparison
print("\n🔬 Top-N Circuits Comparison")
topN = 5
multi_results = []
for i in range(min(topN, len(combination_results))):
    circuit = combination_results[i]["circuit"]
    print(f"\n🚀 Evaluating Top-{i+1} Circuit: {circuit}")
    res = evaluate_ablation(circuit, name=f"Top-{i+1} Circuit")
    multi_results.append((i + 1, res["drop_pct"]))

print("\n" + "=" * 60)
print("📊 Multi-Circuit Comparison")
print("=" * 60)
for rank, drop_pct in multi_results:
    print(f"  Rank {rank}: Drop {drop_pct:.2f}%")


# =============================================
# 12. 实验结论汇总
# =============================================

print("=" * 70)
print("📋 DE 单语言实验结论")
print("=" * 70)

print(f"\n📊 数据统计:")
print(f"  总样本: {len(dataset)} | index解析成功: {len(valid)} | "
      f"有效样本: {len(filtered)}")
print(f"  Mean agreement delta (filtered): "
      f"{np.mean([r['delta'] for r in filtered]):.4f}")

print(f"\n🎯 候选池 ({len(pool)} 个 heads):")
for l, h in pool:
    tag = "✅[sufficient+necessary]" if (l, h) in overlap else \
          "🔵[sufficient]" if (l, h) in set(candidate_heads) else \
          "🟡[necessary]"
    print(f"  L{l:2d}H{h:2d} {tag} | "
          f"patch={logit_boost_heatmap[l,h]:+.4f} | "
          f"abl={ablation_heatmap[l,h]:.4f}")

print(f"\n🏆 Best Circuit: {best_circuit}")
print(f"  Ablation Drop: {res_best['drop_pct']:.2f}%")

print(f"\n💡 Hook 位置: attention.self (pre-projection)")
print(f"   Patching 和 Ablation 使用相同 hook 位置，结论一致")
