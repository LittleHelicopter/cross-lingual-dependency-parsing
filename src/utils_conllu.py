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
    
    Args:
        sentences: List of Sentence objects
        
    Returns:
        Dictionary mapping deprel labels to indices
    """
    labels = set()
    for sent in sentences:
        labels.update(sent.deprels)
    
    # Sort for consistency
    sorted_labels = sorted(list(labels))
    label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
    
    return label_to_idx


def compute_uas_las(pred_heads: torch.Tensor, pred_labels: torch.Tensor,
                    gold_heads: torch.Tensor, gold_labels: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[float, float]:
    """
    Compute Unlabeled Attachment Score (UAS) and Labeled Attachment Score (LAS).
    
    Args:
        pred_heads: Predicted head indices [batch_size, seq_len]
        pred_labels: Predicted label indices [batch_size, seq_len]
        gold_heads: Gold head indices [batch_size, seq_len]
        gold_labels: Gold label indices [batch_size, seq_len]
        mask: Validity mask [batch_size, seq_len]
        
    Returns:
        Tuple of (UAS, LAS) as percentages
    """
    mask = mask.bool()
    
    # UAS: correct head attachments
    head_correct = (pred_heads == gold_heads) & mask
    uas = head_correct.sum().item() / mask.sum().item()
    
    # LAS: correct head AND label
    label_correct = (pred_labels == gold_labels) & mask
    las_correct = head_correct & label_correct
    las = las_correct.sum().item() / mask.sum().item()
    
    return uas * 100, las * 100


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
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sent = self.sentences[idx]
        
        # Tokenize with word-level alignment
        # Add special tokens: [CLS] word1 word2 ... [SEP]
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
        
        # Map word indices to token indices (account for [CLS])
        word_ids = tokenized.word_ids()
        
        # Create gold heads and labels (aligned with tokens)
        # Add 1 to heads to account for [CLS] token at position 0
        gold_heads = [0]  # [CLS] has no head
        gold_labels = [0]  # padding label for [CLS]
        
        for i, word_id in enumerate(word_ids[1:], start=1):  # Skip [CLS]
            if word_id is None:  # [SEP] or [PAD]
                gold_heads.append(0)
                gold_labels.append(0)
            else:
                # First subword of each word gets the gold annotation
                if i == 1 or word_ids[i-1] != word_id:
                    head = sent.heads[word_id]
                    # Adjust head index for [CLS] token
                    if head > 0:
                        # Find the first subword token for the head word
                        head_token_idx = next(j for j, wid in enumerate(word_ids) if wid == head - 1)
                        gold_heads.append(head_token_idx)
                    else:
                        gold_heads.append(0)  # root points to [CLS]
                    
                    deprel = sent.deprels[word_id]
                    gold_labels.append(self.label_to_idx[deprel])
                else:
                    # Subsequent subwords copy from first subword
                    gold_heads.append(gold_heads[-1])
                    gold_labels.append(gold_labels[-1])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'heads': torch.tensor(gold_heads, dtype=torch.long),
            'labels': torch.tensor(gold_labels, dtype=torch.long),
            'word_ids': word_ids
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    input_ids = []
    attention_mask = []
    heads = []
    labels = []
    
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
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'heads': torch.stack(heads),
        'labels': torch.stack(labels)
    }