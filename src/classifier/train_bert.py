"""
BERT Fine-tuning for SMS Spam Classification

This script fine-tunes a pre-trained BERT model (bert-base-uncased) 
for binary spam classification using Hugging Face transformers.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import argparse
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SMSDataset(Dataset):
    """Custom dataset for SMS spam data."""
    
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

def load_data(train_path='data/train.csv', test_path='data/test.csv'):
    """Load preprocessed training and testing data."""
    print("Loading preprocessed data...")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df

def compute_metrics(eval_pred: EvalPrediction):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    # Spam-specific metrics (class 1)
    precision_spam, recall_spam, f1_spam, _ = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'f1_spam': f1_spam[1] if len(f1_spam) > 1 else 0.0,
        'precision_spam': precision_spam[1] if len(precision_spam) > 1 else 0.0,
        'recall_spam': recall_spam[1] if len(recall_spam) > 1 else 0.0,
    }

def create_data_loaders(train_df, test_df, tokenizer, batch_size=16, max_length=128):
    """Create PyTorch data loaders."""
    print("Creating data loaders...")
    
    train_dataset = SMSDataset(
        train_df['message'], train_df['label'], tokenizer, max_length
    )
    test_dataset = SMSDataset(
        test_df['message'], test_df['label'], tokenizer, max_length
    )
    
    return train_dataset, test_dataset

def fine_tune_bert(train_dataset, test_dataset, output_dir='models/bert_spam_classifier', 
                   model_name='bert-base-uncased', num_epochs=3, batch_size=16, learning_rate=2e-5):
    """Fine-tune BERT model for spam classification."""
    print(f"Fine-tuning {model_name} for spam classification...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1}
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_spam",
        greater_is_better=True,
        seed=42,
        learning_rate=learning_rate,
        dataloader_pin_memory=False,  # Fix pin_memory warning on Mac
        remove_unused_columns=False,  # Prevent column removal warnings
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to: {output_dir}")
    
    return trainer, eval_results

def evaluate_bert_model(trainer, test_dataset):
    """Detailed evaluation of the BERT model."""
    print("\n=== BERT Model Evaluation ===")
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Ham (0) - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
    print(f"Spam (1) - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"Ham predicted as Ham: {cm[0][0]}")
    print(f"Ham predicted as Spam: {cm[0][1]}")
    print(f"Spam predicted as Ham: {cm[1][0]}")
    print(f"Spam predicted as Spam: {cm[1][1]}")
    
    return {
        'accuracy': accuracy,
        'precision_ham': precision[0],
        'recall_ham': recall[0],
        'f1_ham': f1[0],
        'precision_spam': precision[1],
        'recall_spam': recall[1],
        'f1_spam': f1[1],
        'confusion_matrix': cm
    }

def compare_with_baseline(bert_results, baseline_path='models/baseline_model_info.txt'):
    """Compare BERT results with baseline SVM."""
    print("\n=== Comparison with Baseline ===")
    
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline_info = f.read()
        print("Baseline SVM Results:")
        print(baseline_info)
    
    print(f"\nBERT Results:")
    print(f"Accuracy: {bert_results['accuracy']:.4f}")
    print(f"Spam F1: {bert_results['f1_spam']:.4f}")
    print(f"Spam Precision: {bert_results['precision_spam']:.4f}")
    print(f"Spam Recall: {bert_results['recall_spam']:.4f}")

def save_bert_results(results, output_dir='models/bert_spam_classifier'):
    """Save BERT evaluation results."""
    results_path = os.path.join(output_dir, 'bert_evaluation_results.txt')
    
    with open(results_path, 'w') as f:
        f.write(f"BERT Spam Classification Results\n")
        f.write(f"Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: bert-base-uncased\n\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Ham Precision: {results['precision_ham']:.4f}\n")
        f.write(f"Ham Recall: {results['recall_ham']:.4f}\n")
        f.write(f"Ham F1: {results['f1_ham']:.4f}\n")
        f.write(f"Spam Precision: {results['precision_spam']:.4f}\n")
        f.write(f"Spam Recall: {results['recall_spam']:.4f}\n")
        f.write(f"Spam F1: {results['f1_spam']:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"[[{results['confusion_matrix'][0][0]} {results['confusion_matrix'][0][1]}]\n")
        f.write(f" [{results['confusion_matrix'][1][0]} {results['confusion_matrix'][1][1]}]]\n")
    
    print(f"Results saved to: {results_path}")

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Fine-tune BERT for SMS spam classification')
    parser.add_argument('--train', default='data/train.csv', help='Training data path')
    parser.add_argument('--test', default='data/test.csv', help='Test data path')
    parser.add_argument('--output', default='models/bert_spam_classifier', help='Output directory for model')
    parser.add_argument('--model', default='bert-base-uncased', help='Pre-trained model name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    print("Starting BERT Fine-tuning for SMS Spam Classification")
    print("="*60)
    
    # Load data
    train_df, test_df = load_data(args.train, args.test)
    
    # Create tokenizer for data loading
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create datasets
    train_dataset, test_dataset = create_data_loaders(
        train_df, test_df, tokenizer, args.batch_size, args.max_length
    )
    
    # Fine-tune BERT
    trainer, eval_results = fine_tune_bert(
        train_dataset, test_dataset, args.output, args.model,
        args.epochs, args.batch_size, args.learning_rate
    )
    
    # Detailed evaluation
    bert_results = evaluate_bert_model(trainer, test_dataset)
    
    # Save results
    save_bert_results(bert_results, args.output)
    
    # Compare with baseline
    compare_with_baseline(bert_results)
    
    print(f"\nBERT fine-tuning complete!")

if __name__ == "__main__":
    main() 