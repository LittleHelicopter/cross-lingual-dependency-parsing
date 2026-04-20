"""
Utilities for reading and processing CoNLL-U format files.
"""
import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Sentence:
    """Represents a single sentence from CoNLL-U format."""
    words: List[str]
    pos_tags: List[str]
    heads: List[int]
    deprels: List[str]
    def __len__(self):
        return len(self.words)

def read_conllu(filepath: str) -> List[Sentence]:
    """
    Read a CoNLL-U file and return a list of Sentence objects.
    Args:
        filepath: Path to .conllu file
    Returns:
        List of Sentence objects
    """
    sentences = []
    current_words = []
    current_pos = []
    current_heads = []
    current_deprels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments
            if line.startswith('#'):
                continue
            # Sentence boundary (empty line)
            if not line:
                if current_words:
                    sentences.append(Sentence(
                        words=current_words,
                        pos_tags=current_pos,
                        heads=current_heads,
                        deprels=current_deprels
                    ))
                    current_words = []
                    current_pos = []
                    current_heads = []
                    current_deprels = []
                continue
            # Parse CoNLL-U line
            parts = line.split('\t')
            if len(parts) < 10:
                continue
            # Skip multi-word tokens (e.g., 1-2) and empty nodes (e.g., 1.1)
            token_id = parts[0]
            if '-' in token_id or '.' in token_id:
                continue
            try:
                word_id = int(token_id)
            except ValueError:
                continue
            word = parts[1]  # FORM
            pos = parts[3]   # UPOS
            head = int(parts[6]) if parts[6] != '_' else 0  # HEAD
            deprel = parts[7] if parts[7] != '_' else 'root'  # DEPREL
            current_words.append(word)
            current_pos.append(pos)
            current_heads.append(head)
            current_deprels.append(deprel)
        # Don't forget the last sentence (if file doesn't end with empty line)
        if current_words:
            sentences.append(Sentence(
                words=current_words,
                pos_tags=current_pos,
                heads=current_heads,
                deprels=current_deprels
            ))
    return sentences

def create_label_vocab(sentences: List[Sentence]) -> Dict[str, int]:
    """
    Create a vocabulary mapping from dependency relations to indices.
    NOTE: Reserve 0 for [PAD] to ignore non-supervised positions.
    """
    labels = set()
    for sent in sentences:
        labels.update(sent.deprels)
    sorted_labels = sorted(list(labels))
    label_to_idx = {'[PAD]': 0}
    for i, lab in enumerate(sorted_labels, start=1):
        label_to_idx[lab] = i
    return label_to_idx

def compute_uas_las(pred_heads: torch.Tensor, pred_labels: torch.Tensor,
                    gold_heads: torch.Tensor, gold_labels: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[float, float]:
    """
    Compute Unlabeled Attachment Score (UAS) and Labeled Attachment Score (LAS).
    Only positions with mask==1 are considered.
    """
    mask = mask.bool()
    # UAS: correct head attachments
    head_correct = (pred_heads == gold_heads) & mask
    uas = head_correct.sum().item() / max(1, mask.sum().item())
    # LAS: correct head AND label
    label_correct = (pred_labels == gold_labels) & mask
    las_correct = head_correct & label_correct
    las = las_correct.sum().item() / max(1, mask.sum().item())
    return uas * 100.0, las * 100.0

class DependencyDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for dependency parsing."""
    def __init__(self, sentences: List[Sentence], tokenizer, label_to_idx: Dict[str, int]):
        """
        Args:
            sentences: List of Sentence objects
            tokenizer: HuggingFace tokenizer
            label_to_idx: Mapping from deprel labels to indices
        """
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label_to_idx = label_to_idx
        self.pad_label_id = self.label_to_idx.get('[PAD]', 0)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        # Tokenize with word-level alignment
        tokenized = self.tokenizer(
            sent.words,
            is_split_into_words=True,
            add_special_tokens=True,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=512
        )
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        word_ids = tokenized.word_ids()

        # Build gold heads/labels aligned to tokens, plus first-subword mask
        L = input_ids.size(0)
        gold_heads = [0] * L           # default to 0 (root) but will be masked when needed
        gold_labels = [self.pad_label_id] * L
        first_subword_mask = [0] * L

        # Iterate tokens; word_ids gives None for special tokens
        for i, wid in enumerate(word_ids):
            if wid is None:
                # special tokens/pad -> keep defaults: mask=0, pad label
                continue
            # Is first subword of this word?
            is_first = (i == 0) or (word_ids[i - 1] != wid)
            if not is_first:
                # non-first subwords are not supervised
                continue

            # Mark as supervised position
            first_subword_mask[i] = 1

            # Head in word space (1-based), 0 for root
            head_word_idx = sent.heads[wid]
            if head_word_idx <= 0:
                gold_heads[i] = 0
            else:
                # Find first subword token index for the head word (head_word_idx is 1-based)
                target_wid = head_word_idx - 1
                head_token_idx = None
                for j, wj in enumerate(word_ids):
                    if wj == target_wid:
                        head_token_idx = j
                        break
                if head_token_idx is None:
                    # Head fell out of truncation; skip supervision for this token
                    first_subword_mask[i] = 0
                    gold_heads[i] = 0
                    gold_labels[i] = self.pad_label_id
                    continue
                gold_heads[i] = head_token_idx

            # Label
            deprel = sent.deprels[wid]
            gold_labels[i] = self.label_to_idx.get(deprel, self.pad_label_id)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'heads': torch.tensor(gold_heads, dtype=torch.long),
            'labels': torch.tensor(gold_labels, dtype=torch.long),
            'first_subword_mask': torch.tensor(first_subword_mask, dtype=torch.long),
            'word_ids': word_ids
        }

def collate_fn(batch):
    """Custom collate function for batching."""
    max_len = max(item['input_ids'].size(0) for item in batch)
    input_ids = []
    attention_mask = []
    heads = []
    labels = []
    first_subword_mask = []

    for item in batch:
        seq_len = item['input_ids'].size(0)
        padding_len = max_len - seq_len

        input_ids.append(torch.cat([
            item['input_ids'],
            torch.zeros(padding_len, dtype=torch.long)
        ]))
        attention_mask.append(torch.cat([
            item['attention_mask'],
            torch.zeros(padding_len, dtype=torch.long)
        ]))
        heads.append(torch.cat([
            item['heads'],
            torch.zeros(padding_len, dtype=torch.long)
        ]))
        labels.append(torch.cat([
            item['labels'],
            torch.zeros(padding_len, dtype=torch.long)
        ]))
        first_subword_mask.append(torch.cat([
            item['first_subword_mask'],
            torch.zeros(padding_len, dtype=torch.long)
        ]))

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'heads': torch.stack(heads),
        'labels': torch.stack(labels),
        'first_subword_mask': torch.stack(first_subword_mask)
    }