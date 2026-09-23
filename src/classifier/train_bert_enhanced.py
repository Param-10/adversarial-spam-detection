"""
Enhanced BERT Fine-tuning with Class Balancing

This script includes additional optimizations:
- Class-weighted loss function
- Learning rate scheduling
- Better handling of imbalanced data
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os
import argparse
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SMSDataset(Dataset):
    """Enhanced dataset with class information."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_class_weights(y_train):
    """Compute class weights for imbalanced dataset."""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    return torch.tensor(class_weights, dtype=torch.float32)

def compute_metrics_enhanced(eval_pred: EvalPrediction):
    """Enhanced metrics computation with more detailed spam metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Spam-specific metrics (class 1)
    spam_precision = precision_per_class[1] if len(precision_per_class) > 1 else 0.0
    spam_recall = recall_per_class[1] if len(recall_per_class) > 1 else 0.0
    spam_f1 = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'f1_spam': spam_f1,
        'precision_spam': spam_precision,
        'recall_spam': spam_recall,
        'ham_f1': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
    }

class WeightedTrainer(Trainer):
    """Custom trainer with class-weighted loss."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def fine_tune_bert_enhanced(train_dataset, val_dataset, class_weights=None, 
                           output_dir='models/bert_enhanced_classifier', 
                           model_name='bert-base-uncased', num_epochs=3, 
                           batch_size=16, learning_rate=2e-5):
    """Enhanced BERT fine-tuning with class balancing and validation-based checkpointing."""
    print(f"Enhanced fine-tuning {model_name} for spam classification...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1}
    )
    
    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,  # Warm up 10% of training steps
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_spam",
        greater_is_better=True,
        seed=42,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",  # Cosine learning rate schedule
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb logging
    )
    
    # Create enhanced trainer with class weights, evaluating on validation set
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_enhanced,
    )
    
    # Train the model
    print("Starting enhanced training...")
    trainer.train()
    
    # Evaluate on validation set
    print("Evaluating enhanced model on validation set...")
    eval_results = trainer.evaluate()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Enhanced model saved to: {output_dir}")
    
    return trainer, eval_results

def main():
    """Enhanced training pipeline."""
    parser = argparse.ArgumentParser(description='Enhanced BERT fine-tuning for SMS spam')
    parser.add_argument('--train', default='data/train.csv', help='Training data path')
    parser.add_argument('--val', default=None, help='Validation data path (optional, will split train if omitted)')
    parser.add_argument('--test', default='data/test.csv', help='Test data path (untouched holdout)')
    parser.add_argument('--output', default='models/bert_enhanced_classifier', help='Output directory')
    parser.add_argument('--model', default='bert-base-uncased', help='Pre-trained model name')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class-weighted loss')
    
    args = parser.parse_args()
    
    print("Enhanced BERT Fine-tuning for SMS Spam Classification")
    print("="*60)
    
    # Load data with strict train/val/test partitioning
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    
    if args.val and os.path.exists(args.val):
        val_df = pd.read_csv(args.val)
    elif os.path.exists('data/val.csv'):
        val_df = pd.read_csv('data/val.csv')
    else:
        from sklearn.model_selection import train_test_split
        train_split, val_split = train_test_split(
            train_df, test_size=0.15, random_state=42, stratify=train_df['label']
        )
        train_df = train_split.reset_index(drop=True)
        val_df = val_split.reset_index(drop=True)

    # Check for duplicate message leakage across partitions
    train_msgs = set(train_df['message'].astype(str))
    val_msgs = set(val_df['message'].astype(str))
    test_msgs = set(test_df['message'].astype(str))

    leak_train_val = train_msgs.intersection(val_msgs)
    leak_train_test = train_msgs.intersection(test_msgs)
    if leak_train_val or leak_train_test:
        train_df = train_df[~train_df['message'].astype(str).isin(leak_train_val | leak_train_test)].reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples (untouched): {len(test_df)}")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create datasets
    train_dataset = SMSDataset(train_df['message'], train_df['label'], tokenizer, args.max_length)
    val_dataset = SMSDataset(val_df['message'], val_df['label'], tokenizer, args.max_length)
    test_dataset = SMSDataset(test_df['message'], test_df['label'], tokenizer, args.max_length)
    
    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_df['label'].values)
        print(f"Class weights: {class_weights}")
    
    # Enhanced fine-tuning using validation dataset
    trainer, eval_results = fine_tune_bert_enhanced(
        train_dataset, val_dataset, class_weights, args.output, args.model,
        args.epochs, args.batch_size, args.learning_rate
    )
    
    print(f"Enhanced BERT training complete!")
    print(f"Validation evaluation results:")
    for key, value in eval_results.items():
        if key.startswith('eval_'):
            print(f"  {key}: {value:.4f}")

    # Final held-out evaluation on test set
    print("\nEvaluating on untouched test set...")
    test_metrics = trainer.evaluate(test_dataset)
    for key, value in test_metrics.items():
        if key.startswith('eval_'):
            print(f"  test_{key[5:]}: {value:.4f}")

if __name__ == "__main__":
    main() 