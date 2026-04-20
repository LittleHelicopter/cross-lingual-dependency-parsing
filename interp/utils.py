# ============================================================
# utils.py — 共用工具函数
# 供 experiment_de.py 和 experiment_cross_lingual.py 使用
# ============================================================

import torch
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────
# Token Index 解析
# ─────────────────────────────────────────────

def get_first_subword_token(tokenizer, text, word_idx):
    """
    返回 word_idx (0-based, text.split() 中的位置)
    对应的第一个 subword token 的位置 index。
    """
    words = text.split()
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=True)
    word_ids = enc.word_ids()
    for i, wid in enumerate(word_ids):
        if wid == word_idx and (i == 0 or word_ids[i - 1] != word_idx):
            return i
    return None


def find_word_idx(text, word_form):
    """
    大小写不敏感地在 text.split() 中查找 word_form。
    先精确匹配，再前缀匹配，均失败返回 None。
    """
    if not word_form:
        return None
    words = text.split()
    for i, w in enumerate(words):
        if w.lower() == word_form.lower():
            return i
    for i, w in enumerate(words):
        if w.lower().startswith(word_form.lower()):
            return i
    return None


def resolve_token_indices(item, tokenizer, lang):
    """
    统一的 token index 解析函数，支持 en / de。

    优先使用 {lang}_indices 中的索引并做校验，
    失败则 fallback 到 {lang}_meta 的字符串匹配。

    参数：
        item      : 数据条目，包含 {lang}_clean / {lang}_corr /
                    {lang}_indices / {lang}_meta 字段
        tokenizer : 分词器
        lang      : "en" 或 "de"

    返回 dict：
        subj_tok_clean, verb_tok_clean : clean 句中的 subword token index
        subj_tok_corr,  verb_tok_corr  : corr  句中的 subword token index
        subj_word_idx, verb_word_idx   : word-level index（用于调试）
    """
    clean_text = item[f"{lang}_clean"]
    corr_text  = item[f"{lang}_corr"]
    indices    = item.get(f"{lang}_indices", {})
    # 优先使用对应语言的 metadata，避免 EN/DE metadata 混用
    meta       = item.get(f"{lang}_meta", item.get("metadata", {}))

    # ── subj word index ──
    subj_word_idx = indices.get("subj")
    words = clean_text.split()
    subj_form = meta.get("subj_form", "")
    if subj_word_idx is not None:
        if subj_word_idx >= len(words) or \
                words[subj_word_idx].lower() != subj_form.lower():
            subj_word_idx = find_word_idx(clean_text, subj_form)
    else:
        subj_word_idx = find_word_idx(clean_text, subj_form)

    # ── verb word index ──
    verb_form = meta.get("verb_form", "")
    verb_word_idx = find_word_idx(clean_text, verb_form)
    if verb_word_idx is None:
        verb_word_idx = indices.get("verb")

    # ── subword token indices ──
    subj_tok_clean = get_first_subword_token(tokenizer, clean_text, subj_word_idx) \
                     if subj_word_idx is not None else None
    verb_tok_clean = get_first_subword_token(tokenizer, clean_text, verb_word_idx) \
                     if verb_word_idx is not None else None
    subj_tok_corr  = get_first_subword_token(tokenizer, corr_text, subj_word_idx) \
                     if subj_word_idx is not None else None
    verb_tok_corr  = get_first_subword_token(tokenizer, corr_text, verb_word_idx) \
                     if verb_word_idx is not None else None

    return {
        "subj_tok_clean": subj_tok_clean,
        "verb_tok_clean": verb_tok_clean,
        "subj_tok_corr":  subj_tok_corr,
        "verb_tok_corr":  verb_tok_corr,
        "subj_word_idx":  subj_word_idx,
        "verb_word_idx":  verb_word_idx,
    }


# ─────────────────────────────────────────────
# Arc Logit & Masking
# ─────────────────────────────────────────────

def apply_first_subword_mask(arc_scores, tokenizer, text, device):
    """对 arc_scores 应用 first-subword mask，遮蔽非词首 subword。"""
    words = text.split()
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=True)
    word_ids = enc.word_ids()
    from train_parser import mask_arc_scores
    mask = torch.tensor([[
        1 if (word_ids[i] is not None and
              (i == 0 or word_ids[i] != word_ids[i - 1])) else 0
        for i in range(len(word_ids))
    ]]).to(device)
    return mask_arc_scores(arc_scores, mask)


def get_arc_logit(arc_scores, subj_tok, verb_tok):
    """取 arc_scores[0, subj_tok, verb_tok]，任一为 None 则返回 None。"""
    if subj_tok is None or verb_tok is None:
        return None
    return arc_scores[0, subj_tok, verb_tok].item()


# ─────────────────────────────────────────────
# Agreement Delta
# ─────────────────────────────────────────────

def get_agreement_delta(model, tokenizer, item, lang, device, tok_key="_tok"):
    """
    agreement_delta = logit(subj→correct_verb | clean句)
                    - logit(subj→incorrect_verb | corr句)
    delta > 0 表示模型对正确形态更有信心。

    tok_key: item 中存储 token index dict 的键（默认 "_tok"；
             跨语言版本使用 "_tok_en" / "_tok_de"）
    """
    tok = item[tok_key] if tok_key in item else item.get(f"_tok_{lang}", item.get("_tok"))
    if tok["subj_tok_clean"] is None or tok["verb_tok_clean"] is None:
        return None, None, None

    # clean
    clean_inp = tokenizer(item[f"{lang}_clean"], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**clean_inp)
        arc = out[0] if isinstance(out, (tuple, list)) else out.arc_scores
    arc = apply_first_subword_mask(arc, tokenizer, item[f"{lang}_clean"], device)
    clean_logit = get_arc_logit(arc, tok["subj_tok_clean"], tok["verb_tok_clean"])

    # corr
    corr_inp = tokenizer(item[f"{lang}_corr"], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**corr_inp)
        arc = out[0] if isinstance(out, (tuple, list)) else out.arc_scores
    arc = apply_first_subword_mask(arc, tokenizer, item[f"{lang}_corr"], device)
    corr_logit = get_arc_logit(arc, tok["subj_tok_corr"], tok["verb_tok_corr"])

    if clean_logit is None or corr_logit is None:
        return None, None, None

    return clean_logit, corr_logit, clean_logit - corr_logit


# ─────────────────────────────────────────────
# Activation Caching
# ─────────────────────────────────────────────

def get_activations(model, inputs):
    """
    在 attention.self 输出（pre-projection）缓存每层激活。
    返回 dict: {"layer_0": tensor[1, seq, D], ..., "layer_N": ...}
    """
    activations = {}

    def save_hook(name):
        def hook(mod, inp, out):
            val = out[0] if isinstance(out, tuple) else out
            activations[name] = val.detach()
        return hook

    handles = [
        layer.attention.self.register_forward_hook(save_hook(f"layer_{i}"))
        for i, layer in enumerate(model.encoder.encoder.layer)
    ]
    with torch.no_grad():
        _ = model(**inputs)
    for h in handles:
        h.remove()
    return activations


# ─────────────────────────────────────────────
# Mean Activation 预计算
# ─────────────────────────────────────────────

def compute_mean_activations(model, tokenizer, items, lang, n_layers, n_heads, head_dim, device):
    """
    对 items 列表中所有样本的 {lang}_clean 句计算每层的
    per-head 均值激活（attention.self 输出空间）。

    返回 dict: {layer_idx: tensor[n_heads, head_dim]}
    """
    layer_acts_collection = {l: [] for l in range(n_layers)}
    for item in items:
        inp  = tokenizer(item[f"{lang}_clean"], return_tensors="pt").to(device)
        acts = get_activations(model, inp)
        for l in range(n_layers):
            layer_acts_collection[l].append(acts[f"layer_{l}"])

    mean_acts = {}
    for l in range(n_layers):
        per_sent = [a.mean(dim=1) for a in layer_acts_collection[l]]
        mean_vec = torch.cat(per_sent, dim=0).mean(dim=0)   # [D]
        mean_acts[l] = mean_vec.reshape(n_heads, head_dim)
    return mean_acts


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_jsonl(path):
    """加载 jsonl 文件，返回以 item["id"] 为键的 dict。"""
    import json
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data[item["id"]] = item
    return data
