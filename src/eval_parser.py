"""
Evaluation script for trained dependency parser.
"""

import torch
import argparse
import json
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from utils_conllu import (
    read_conllu, DependencyDataset, collate_fn, compute_uas_las
)
from train_parser import BiaffineDependencyParser


def evaluate_detailed(model, dataloader, device, idx_to_label=None):
    """
    Evaluate model with detailed metrics.
    
    Args:
        model: Trained parser model
        dataloader: DataLoader for evaluation
        device: torch device
        idx_to_label: Optional mapping from label indices to names
        
    Returns:
        Dictionary with UAS, LAS, and per-label accuracy
    """
    model.eval()
    
    total_correct_heads = 0
    total_correct_labels = 0
    total_tokens = 0
    
    # For per-label statistics
    label_correct = {}
    label_total = {}
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gold_heads = batch['heads'].to(device)
            gold_labels = batch['labels'].to(device)
            
            # Forward pass
            arc_scores, rel_scores = model(input_ids, attention_mask)
            
            # Predict heads
            pred_heads = arc_scores.argmax(dim=-1)
            
            # Predict labels at predicted heads
            batch_size, seq_len = pred_heads.shape
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
            token_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            rel_scores_at_pred_heads = rel_scores[batch_indices, token_indices, pred_heads]
            pred_labels = rel_scores_at_pred_heads.argmax(dim=-1)
            
            # Compute metrics with mask
            mask = attention_mask.bool()
            
            # UAS: correct head attachments
            head_correct_mask = (pred_heads == gold_heads) & mask
            total_correct_heads += head_correct_mask.sum().item()
            
            # LAS: correct head AND label
            label_correct_mask = (pred_labels == gold_labels) & mask
            las_correct_mask = head_correct_mask & label_correct_mask
            total_correct_labels += las_correct_mask.sum().item()
            
            total_tokens += mask.sum().item()
            
            # Per-label statistics
            if idx_to_label:
                for label_idx in idx_to_label.keys():
                    label_mask = (gold_labels == label_idx) & mask
                    if label_mask.any():
                        label_correct_count = (head_correct_mask & label_mask).sum().item()
                        label_total_count = label_mask.sum().item()
                        
                        if label_idx not in label_correct:
                            label_correct[label_idx] = 0
                            label_total[label_idx] = 0
                        
                        label_correct[label_idx] += label_correct_count
                        label_total[label_idx] += label_total_count
    
    # Compute overall scores
    uas = (total_correct_heads / total_tokens) * 100 if total_tokens > 0 else 0
    las = (total_correct_labels / total_tokens) * 100 if total_tokens > 0 else 0
    
    results = {
        'UAS': uas,
        'LAS': las,
        'num_tokens': total_tokens
    }
    
    # Per-label accuracy
    if idx_to_label:
        per_label_stats = {}
        for label_idx, label_name in idx_to_label.items():
            if label_idx in label_total and label_total[label_idx] > 0:
                accuracy = (label_correct[label_idx] / label_total[label_idx]) * 100
                per_label_stats[label_name] = {
                    'accuracy': accuracy,
                    'count': label_total[label_idx]
                }
        
        results['per_label'] = per_label_stats
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate dependency parser')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test .conllu file')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file for results (JSON)')
    parser.add_argument('--detailed', action='store_true',
                        help='Compute detailed per-label statistics')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get training arguments from checkpoint
    train_args = checkpoint.get('args', {})
    model_name = train_args.get('model_name', 'xlm-roberta-base')
    mlp_size = train_args.get('mlp_size', 256)
    dropout = train_args.get('dropout', 0.33)
    
    # Load label vocabulary
    checkpoint_dir = os.path.dirname(args.checkpoint)
    label_vocab_path = os.path.join(checkpoint_dir, 'label_vocab.json')
    
    if os.path.exists(label_vocab_path):
        with open(label_vocab_path, 'r') as f:
            label_to_idx = json.load(f)
        print(f"Loaded label vocabulary: {len(label_to_idx)} labels")
    else:
        print("Warning: label_vocab.json not found in checkpoint directory")
        # Try to infer from checkpoint
        label_to_idx = None
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load test data
    print(f"Loading test data from {args.test_file}")
    test_sentences = read_conllu(args.test_file)
    print(f"Loaded {len(test_sentences)} test sentences")
    
    # If label_to_idx wasn't loaded, create it from test data
    if label_to_idx is None:
        from utils_conllu import create_label_vocab
        label_to_idx = create_label_vocab(test_sentences)
        print(f"Created label vocabulary from test data: {len(label_to_idx)} labels")
    
    # Create dataset
    test_dataset = DependencyDataset(test_sentences, tokenizer, label_to_idx)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    
    # Initialize model
    print(f"Initializing model: {model_name}")
    encoder = AutoModel.from_pretrained(model_name)
    
    model = BiaffineDependencyParser(
        encoder=encoder,
        num_labels=len(label_to_idx),
        hidden_size=encoder.config.hidden_size,
        mlp_size=mlp_size,
        dropout=dropout
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("\n=== Evaluating ===")
    
    # Evaluate
    if args.detailed:
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        results = evaluate_detailed(model, test_loader, device, idx_to_label)
        
        print(f"\nResults:")
        print(f"UAS: {results['UAS']:.2f}%")
        print(f"LAS: {results['LAS']:.2f}%")
        print(f"Total tokens: {results['num_tokens']}")
        
        if 'per_label' in results:
            print(f"\nPer-label accuracy (top 10 by frequency):")
            sorted_labels = sorted(
                results['per_label'].items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )
            for label, stats in sorted_labels[:10]:
                print(f"  {label:15s}: {stats['accuracy']:5.2f}% (n={stats['count']})")
    else:
        from train_parser import evaluate
        results = evaluate(model, test_loader, device)
        
        print(f"\nResults:")
        print(f"UAS: {results['UAS']:.2f}%")
        print(f"LAS: {results['LAS']:.2f}%")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")
    
    # Compare with checkpoint metrics if available
    if 'uas' in checkpoint and 'las' in checkpoint:
        print(f"\nCheckpoint metrics (dev set):")
        print(f"UAS: {checkpoint['uas']:.2f}%")
        print(f"LAS: {checkpoint['las']:.2f}%")


if __name__ == '__main__':
    main()