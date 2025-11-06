"""
Training script for biaffine dependency parser with XLM-R.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import json
import os

from utils_conllu import (
    read_conllu, create_label_vocab, DependencyDataset,
    collate_fn, compute_uas_las
)


class Biaffine(nn.Module):
    """Biaffine attention layer."""
    
    def __init__(self, in1_features, in2_features, out_features):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.zeros(out_features, in1_features, in2_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, input1, input2):
        """
        Args:
            input1: [batch_size, len1, in1_features]  (dependent)
            input2: [batch_size, len2, in2_features]  (head)
        
        Returns:
            output: [batch_size, len1, len2, out_features]
        """
        batch_size, len1, dim1 = input1.shape
        len2, dim2 = input2.shape[1], input2.shape[2]
        
        # Method: For each output dimension o:
        #   score[b,i,j,o] = input1[b,i,:] @ W[o,:,:] @ input2[b,j,:]
        # Efficiently compute using einsum
        
        # Compute biaffine: input1 @ W @ input2^T
        # output[b, i, j, o] = sum_{d1, d2} input1[b,i,d1] * W[o,d1,d2] * input2[b,j,d2]
        output = torch.einsum('bxi,oij,byj->bxyo', input1, self.weight, input2)
        
        # Add bias
        output = output + self.bias.view(1, 1, 1, -1)
        
        return output


class BiaffineDependencyParser(nn.Module):
    """Biaffine dependency parser with XLM-R encoder."""
    
    def __init__(self, encoder, num_labels, hidden_size=768, mlp_size=256, dropout=0.33):
        """
        Args:
            encoder: Pretrained transformer model (e.g., XLM-R)
            num_labels: Number of dependency relation labels
            hidden_size: Size of encoder hidden states
            mlp_size: Size of MLP layers
            dropout: Dropout rate
        """
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        
        # MLP layers for arc prediction (head selection)
        self.arc_mlp_head = nn.Linear(hidden_size, mlp_size)
        self.arc_mlp_dep = nn.Linear(hidden_size, mlp_size)
        
        # MLP layers for label prediction
        self.rel_mlp_head = nn.Linear(hidden_size, mlp_size)
        self.rel_mlp_dep = nn.Linear(hidden_size, mlp_size)
        
        # Biaffine layers
        self.arc_biaffine = Biaffine(mlp_size, mlp_size, 1)
        self.rel_biaffine = Biaffine(mlp_size, mlp_size, num_labels)
        
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            arc_scores: [batch_size, seq_len, seq_len] - head selection scores
            rel_scores: [batch_size, seq_len, seq_len, num_labels] - relation scores
        """
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        hidden = self.dropout(hidden)
        
        # Arc MLPs
        arc_head = self.activation(self.arc_mlp_head(hidden))
        arc_head = self.dropout(arc_head)  # [batch_size, seq_len, mlp_size]
        
        arc_dep = self.activation(self.arc_mlp_dep(hidden))
        arc_dep = self.dropout(arc_dep)  # [batch_size, seq_len, mlp_size]
        
        # Relation MLPs
        rel_head = self.activation(self.rel_mlp_head(hidden))
        rel_head = self.dropout(rel_head)
        
        rel_dep = self.activation(self.rel_mlp_dep(hidden))
        rel_dep = self.dropout(rel_dep)
        
        # Biaffine attention
        # Arc scores: [batch, seq_len, seq_len, 1] -> [batch, seq_len, seq_len]
        arc_scores = self.arc_biaffine(arc_dep, arc_head).squeeze(-1)
        
        # Relation scores: [batch, seq_len, seq_len, num_labels]
        rel_scores = self.rel_biaffine(rel_dep, rel_head)
        
        return arc_scores, rel_scores
    
    def freeze_encoder(self):
        """Freeze encoder weights for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_arc_loss = 0
    total_rel_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        gold_heads = batch['heads'].to(device)
        gold_labels = batch['labels'].to(device)
        
        # Forward pass
        arc_scores, rel_scores = model(input_ids, attention_mask)
        
        # Mask: ignore padding tokens
        mask = attention_mask.bool()
        
        # Arc loss (head prediction)
        arc_loss_fn = nn.CrossEntropyLoss(reduction='none')
        arc_loss = arc_loss_fn(
            arc_scores.reshape(-1, arc_scores.size(-1)),
            gold_heads.reshape(-1)
        )
        arc_loss = (arc_loss * mask.reshape(-1).float()).sum() / mask.sum()
        
        # Relation loss (label prediction)
        # Use gold heads during training for stability
        batch_size, seq_len = gold_heads.shape
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        token_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        rel_scores_at_gold_heads = rel_scores[batch_indices, token_indices, gold_heads]
        
        rel_loss_fn = nn.CrossEntropyLoss(reduction='none')
        rel_loss = rel_loss_fn(
            rel_scores_at_gold_heads.reshape(-1, rel_scores_at_gold_heads.size(-1)),
            gold_labels.reshape(-1)
        )
        rel_loss = (rel_loss * mask.reshape(-1).float()).sum() / mask.sum()
        
        # Total loss
        loss = arc_loss + rel_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_arc_loss += arc_loss.item()
        total_rel_loss += rel_loss.item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'arc': f'{arc_loss.item():.4f}',
            'rel': f'{rel_loss.item():.4f}'
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'arc_loss': total_arc_loss / len(dataloader),
        'rel_loss': total_rel_loss / len(dataloader)
    }


def evaluate(model, dataloader, device):
    """Evaluate model on dev/test set."""
    model.eval()
    
    total_correct_heads = 0
    total_correct_labels = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gold_heads = batch['heads'].to(device)
            gold_labels = batch['labels'].to(device)
            
            # Forward pass
            arc_scores, rel_scores = model(input_ids, attention_mask)
            
            # Predict heads (argmax)
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
            head_correct = (pred_heads == gold_heads) & mask
            total_correct_heads += head_correct.sum().item()
            
            # LAS: correct head AND label
            label_correct = (pred_labels == gold_labels) & mask
            las_correct = head_correct & label_correct
            total_correct_labels += las_correct.sum().item()
            
            total_tokens += mask.sum().item()
    
    # Compute final scores
    uas = (total_correct_heads / total_tokens) * 100 if total_tokens > 0 else 0
    las = (total_correct_labels / total_tokens) * 100 if total_tokens > 0 else 0
    
    return {'UAS': uas, 'LAS': las}


def main():
    parser = argparse.ArgumentParser(description='Train biaffine dependency parser')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, required=True, help='Path to training .conllu file')
    parser.add_argument('--dev_file', type=str, required=True, help='Path to dev .conllu file')
    parser.add_argument('--test_file', type=str, default=None, help='Path to test .conllu file')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base',
                        help='Pretrained model name')
    parser.add_argument('--mlp_size', type=int, default=256, help='MLP hidden size')
    parser.add_argument('--dropout', type=float, default=0.33, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights')
    parser.add_argument('--load_encoder', type=str, default=None,
                        help='Path to pretrained encoder checkpoint')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory')
    parser.add_argument('--exp_name', type=str, default='exp', help='Experiment name')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load data
    print(f"Loading training data from {args.train_file}")
    train_sentences = read_conllu(args.train_file)
    print(f"Loaded {len(train_sentences)} training sentences")
    
    print(f"Loading dev data from {args.dev_file}")
    dev_sentences = read_conllu(args.dev_file)
    print(f"Loaded {len(dev_sentences)} dev sentences")
    
    if args.test_file:
        print(f"Loading test data from {args.test_file}")
        test_sentences = read_conllu(args.test_file)
        print(f"Loaded {len(test_sentences)} test sentences")
    
    # Create label vocabulary
    # label_to_idx = create_label_vocab(train_sentences)
    # found that there's bug in zh, "orphan" did not occur in training dataset
    label_to_idx = create_label_vocab(train_sentences + dev_sentences + test_sentences)
    print(f"Found {len(label_to_idx)} dependency labels")
    
    # Save label vocabulary
    with open(os.path.join(exp_dir, 'label_vocab.json'), 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    
    # Create datasets
    train_dataset = DependencyDataset(train_sentences, tokenizer, label_to_idx)
    dev_dataset = DependencyDataset(dev_sentences, tokenizer, label_to_idx)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    
    # Initialize model
    print(f"Initializing model: {args.model_name}")
    encoder = AutoModel.from_pretrained(args.model_name)
    
    # Load pretrained encoder if specified
    if args.load_encoder:
        print(f"Loading pretrained encoder from {args.load_encoder}")
        checkpoint = torch.load(args.load_encoder, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    model = BiaffineDependencyParser(
        encoder=encoder,
        num_labels=len(label_to_idx),
        hidden_size=encoder.config.hidden_size,
        mlp_size=args.mlp_size,
        dropout=args.dropout
    )
    
    if args.freeze_encoder:
        print("Freezing encoder weights")
        model.freeze_encoder()
    
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )
    
    # Training loop
    best_uas = 0
    best_las = 0
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_metrics['loss']:.4f} "
              f"(Arc: {train_metrics['arc_loss']:.4f}, Rel: {train_metrics['rel_loss']:.4f})")
        
        # Evaluate
        dev_metrics = evaluate(model, dev_loader, device)
        print(f"Dev UAS: {dev_metrics['UAS']:.2f}%, LAS: {dev_metrics['LAS']:.2f}%")
        
        # Save best model
        if dev_metrics['LAS'] > best_las:
            best_uas = dev_metrics['UAS']
            best_las = dev_metrics['LAS']
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'uas': best_uas,
                'las': best_las,
                'args': vars(args)
            }
            
            torch.save(checkpoint, os.path.join(exp_dir, 'best_model.pt'))
            print(f"Saved best model (LAS: {best_las:.2f}%)")
    
    # Final evaluation on test set
    if args.test_file:
        print("\n=== Final Test Evaluation ===")
        checkpoint = torch.load(os.path.join(exp_dir, 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_dataset = DependencyDataset(test_sentences, tokenizer, label_to_idx)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn
        )
        
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test UAS: {test_metrics['UAS']:.2f}%, LAS: {test_metrics['LAS']:.2f}%")
        
        # Save test results
        with open(os.path.join(exp_dir, 'test_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
    
    print(f"\nBest Dev UAS: {best_uas:.2f}%, LAS: {best_las:.2f}%")


if __name__ == '__main__':
    main()