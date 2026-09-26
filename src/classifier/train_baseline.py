"""
Baseline SMS Spam Classifier Training

This script trains and evaluates classical machine learning models:
- Multinomial Naive Bayes
- Logistic Regression  
- Support Vector Machine (SVM)

Uses TF-IDF vectorization for feature extraction.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

if __package__:
    from .data_splits import load_data
else:
    from data_splits import load_data

def create_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2)):
    """
    Create TF-IDF vectorizer for feature extraction.
    
    Args:
        max_features (int): Maximum number of features
        ngram_range (tuple): N-gram range for features
        
    Returns:
        TfidfVectorizer: Configured vectorizer
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        strip_accents='unicode',
        lowercase=True
    )

def train_naive_bayes(X_train, y_train):
    """Train Multinomial Naive Bayes classifier."""
    print("\n=== Training Multinomial Naive Bayes ===")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', create_tfidf_vectorizer()),
        ('nb', MultinomialNB())
    ])
    
    # Hyperparameter tuning
    param_grid = {
        'tfidf__max_features': [3000, 5000, 8000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'nb__alpha': [0.1, 0.5, 1.0, 2.0]
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression classifier."""
    print("\n=== Training Logistic Regression ===")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', create_tfidf_vectorizer()),
        ('lr', LogisticRegression(random_state=42, max_iter=2000))
    ])
    
    # Hyperparameter tuning
    param_grid = {
        'tfidf__max_features': [3000, 5000, 8000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'lr__C': [0.1, 1.0, 10.0],
        'lr__penalty': ['l1', 'l2'],
        'lr__solver': ['liblinear']
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_svm(X_train, y_train):
    """Train Support Vector Machine classifier."""
    print("\n=== Training Support Vector Machine ===")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', create_tfidf_vectorizer()),
        ('svm', SVC(random_state=42, probability=True))
    ])
    
    # Hyperparameter tuning (simplified for speed)
    param_grid = {
        'tfidf__max_features': [3000, 5000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'svm__C': [0.1, 1.0, 10.0],
        'svm__kernel': ['linear', 'rbf']
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model for reporting
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n=== Evaluating {model_name} ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], output_dict=True)
    }

def plot_confusion_matrix(cm, model_name, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")
    
    plt.show()

def compare_models(results):
    """Compare model performance and select the best one."""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    model_scores = []
    
    for model_name, result in results.items():
        # Get spam class metrics (class 1) - try different key formats
        class_report = result['classification_report']
        if '1' in class_report:
            spam_metrics = class_report['1']
        elif 1 in class_report:
            spam_metrics = class_report[1] 
        else:
            # Look for 'Spam' class
            spam_metrics = class_report['Spam']
        
        f1_spam = spam_metrics['f1-score']
        precision_spam = spam_metrics['precision']
        recall_spam = spam_metrics['recall']
        accuracy = result['accuracy']
        
        model_scores.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision (Spam)': precision_spam,
            'Recall (Spam)': recall_spam,
            'F1 (Spam)': f1_spam
        })
        
        print(f"{model_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Spam Precision: {precision_spam:.4f}")
        print(f"  Spam Recall: {recall_spam:.4f}")
        print(f"  Spam F1: {f1_spam:.4f}")
        print()
    
    # Find best model based on F1 score for spam
    best_model = max(model_scores, key=lambda x: x['F1 (Spam)'])
    print(f"Best Model: {best_model['Model']} (F1: {best_model['F1 (Spam)']:.4f})")
    
    return best_model['Model'], pd.DataFrame(model_scores)

def save_best_model(results, best_model_name, output_dir='models'):
    """Save the best performing model."""
    os.makedirs(output_dir, exist_ok=True)
    
    best_model = results[best_model_name]['model']
    model_path = os.path.join(output_dir, 'baseline_classifier.pkl')
    
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved: {model_path}")
    
    # Save model info
    info_path = os.path.join(output_dir, 'baseline_model_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Best Baseline Model: {best_model_name}\n")
        f.write(f"Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Accuracy: {results[best_model_name]['accuracy']:.4f}\n")
        f.write("\nModel Parameters:\n")
        for param, value in best_model.get_params().items():
            f.write(f"  {param}: {value}\n")
    
    print(f"Model info saved: {info_path}")

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train baseline SMS spam classifiers')
    parser.add_argument('--train', default='data/train.csv', help='Training data path')
    parser.add_argument('--val', default=None, help='Validation data path for model selection')
    parser.add_argument('--test', default='data/test.csv', help='Test data path')
    parser.add_argument('--output', default='models', help='Output directory for models')
    parser.add_argument('--plots', default='results', help='Directory to save plots')
    
    args = parser.parse_args()
    
    print("Starting Baseline Classifier Training")
    print("="*50)
    
    # Load data
    train_df, val_df, test_df = load_data(args.train, args.test, args.val)
    X_train, y_train = train_df['message'], train_df['label']
    X_val, y_val = val_df['message'], val_df['label']
    
    # Train models
    models = {}
    results = {}
    
    # Train Naive Bayes
    models['Naive Bayes'] = train_naive_bayes(X_train, y_train)
    results['Naive Bayes'] = evaluate_model(models['Naive Bayes'], X_val, y_val, 'Naive Bayes')
    
    # Train Logistic Regression
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train)
    results['Logistic Regression'] = evaluate_model(models['Logistic Regression'], X_val, y_val, 'Logistic Regression')
    
    # Train SVM
    models['SVM'] = train_svm(X_train, y_train)
    results['SVM'] = evaluate_model(models['SVM'], X_val, y_val, 'SVM')
    
    # Compare models
    best_model_name, comparison_df = compare_models(results)
    
    # Save results
    os.makedirs(args.plots, exist_ok=True)
    comparison_df.to_csv(os.path.join(args.plots, 'baseline_validation_comparison.csv'), index=False)
    
    # Select on validation, then evaluate only the selected model on the holdout.
    print('Final held-out test evaluation:')
    test_result = evaluate_model(
        models[best_model_name], test_df['message'], test_df['label'], best_model_name
    )
    save_best_model({best_model_name: test_result}, best_model_name, args.output)
    
    # Skip plotting confusion matrices
    
    print(f"\nBaseline training complete!")

if __name__ == "__main__":
    main() 