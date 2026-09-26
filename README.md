# Adversarial Spam Detection

A machine learning project implementing adversarial training for robust SMS spam detection. This project by Group 5D for AI4ALL explores how iterative adversarial training can improve spam classifier resilience against sophisticated attacks.

## Project Overview

This system implements a complete adversarial training pipeline that alternates between training spam classifiers and generating adversarial spam examples. The goal is to create more robust spam detection models that can handle evolving spam techniques.

### Key Components

- **Baseline Classifiers**: Traditional ML models (Naive Bayes, Logistic Regression, SVM)
- **Advanced BERT Classifier**: Pre-trained transformer model reported at 98.92% accuracy (historical, predates the evaluation-split fix; not rerun with the current loader)
- **Adversarial Generator**: Fine-tuned Qwen3-4B model for generating realistic spam samples
- **Training Loop**: Iterative system that improves classifier robustness through adversarial examples

### Technology Advantages

- **Qwen3-4B Generator**: Latest 2025 model with superior text generation capabilities
- **Memory Efficient**: Optimized 4B parameter model for efficient GPU training
- **T4 GPU Optimized**: Designed for Google Colab T4 GPU training
- **Advanced Chat Format**: Uses modern conversation templates for better spam generation

## Dataset

The project uses the SMS Spam Collection dataset containing 5,574 SMS messages labeled as spam or ham (legitimate messages).

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd adversarial-spam-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up NLTK data (if not already installed):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### Data Preprocessing

Prepare the dataset for training:

```bash
python src/utils/data_preprocessing.py
```

This creates deduplicated `train.csv`, `val.csv`, and `test.csv` files in the
`data/` directory. The canonical partition committed with this repo
(regenerated 2026-09-26 from the raw corpus, `random_state=42`) is
train/val/test = 3,556 / 762 / 762 (spam ≈ 11.9% in every partition), with
0 message overlap across partitions and both classes retained everywhere.

A rerun of this script is deterministic (same input, same outputs) but any
regeneration replaces the committed files, so changes the exact test rows used
for evaluation. Regenerating these files does not reproduce the historical
(pre-split-fix) published metrics.

### Training Baseline Models

Train traditional machine learning classifiers:

```bash
python src/classifier/train_baseline.py --train data/train.csv --val data/val.csv --test data/test.csv
```

### Training BERT Model

Fine-tune BERT for spam classification:

```bash
python src/classifier/train_bert.py --train data/train.csv --val data/val.csv --test data/test.csv --epochs 3
```

### Adversarial Training Pipeline

For the complete adversarial training pipeline, use the Jupyter notebooks (designed for Google Colab with T4 GPU):

#### Step 1: Generator Training

**Notebook**: `notebooks/qwen.ipynb`

- Fine-tunes Qwen3-4B using LoRA for memory-efficient training
- Generates realistic spam samples using advanced conversation format
- Outputs: `qwen3_lora_adapter.zip`

#### Step 2: Complete Adversarial Loop

**Notebook**: `notebooks/complete_loop_colab.ipynb`

- Loads pre-trained BERT classifier (historical 98.92% accuracy)
- Loads trained Qwen3-4B generator
- Implements full adversarial training system
- Iteratively improves classifier robustness
- Requires T4 GPU for optimal performance

**Required Files for Upload:**

1. `qwen3_lora_adapter.zip` (from Step 1)
2. `bert_spam_classifier.zip` (pre-trained BERT model)
3. `train.csv` and `test.csv` (dataset files)

## Methodology

### Adversarial Training Process

1. **Initial Setup**: Load pre-trained BERT classifier (historical 98.92% baseline accuracy)
2. **Generator Loading**: Load fine-tuned Qwen3-4B spam generator with LoRA adapters
3. **Generation Phase**: Use Qwen3-4B to generate sophisticated adversarial spam samples
4. **Evaluation Phase**: Test BERT classifier performance on generated samples
5. **Retraining Phase**: Fine-tune BERT classifier with adversarial samples using LoRA
6. **Iteration**: Repeat the process for 3 iterations to continuously improve robustness

### Model Architecture

- **Baseline Models**: Scikit-learn implementations with TF-IDF vectorization
- **BERT Classifier**: Fine-tuned bert-base-uncased with sequence classification head
- **Generator Model**: Qwen3-4B (4B parameters) with LoRA adapters for parameter-efficient fine-tuning
- **Memory Optimization**: 4-bit quantization for efficient T4 GPU training

### Training Infrastructure

- **Hardware**: Google Colab T4 GPU (15.8 GB memory)
- **Memory Management**: 4-bit quantization with BitsAndBytesConfig
- **Training Method**: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Chat Format**: Qwen3 conversation template (`<|im_start|>...<|im_end|>`)

## Results

> **All figures in this Results section are historical.** They and the checked-in
> model artifacts predate the evaluation-split fix (commit a6295bd) and were not
> rerun with the current shared loader or the corrected train/validation/test
> partitioning. They are retained for the record only; do not treat them as
> current, reproducible results.

### Reproducible results (2026-09-26)

The first numbers produced end-to-end with the corrected pipeline: canonical
70/15/15 partition (see Data Preprocessing above) + shared loader + selection on
validation + single held-out test evaluation. These are rerunnable with the
commands in Usage; no hyperparameters or seeds were changed from the defaults.

**Baseline classifiers** (TF-IDF + GridSearchCV, selection on validation):

| Model (validation) | Accuracy | Spam F1 |
| ------------------ | -------- | ------- |
| Naive Bayes        | 0.9698   | 0.8655  |
| Logistic Regression| 0.9764   | 0.8902  |
| **SVM (selected)** | 0.9803   | **0.9102** |

SVM was selected on validation (highest spam F1) and evaluated **once** on the
untouched 762-row test set: **accuracy 0.9659**, ham precision/recall/F1
0.96 / 1.00 / 0.98, spam precision/recall/F1 1.00 / 0.71 / 0.83.

Artifacts: `results/baseline_validation_comparison.csv`,
`results/reproducible_evaluation.csv`,
`models/reproducible_baseline/baseline_classifier.pkl`.

**Fine-tuned BERT** (bert-base-uncased, 3 epochs, `load_best_model_at_end` on
validation `f1_spam`; best checkpoint = epoch 3 with validation spam F1 0.9379),
evaluated once on the same untouched test set: **accuracy 0.9816**, ham
precision/recall/F1 0.9824 / 0.9970 / 0.9896, spam precision/recall/F1
0.9753 / 0.8681 / 0.9186 (confusion matrix `[[669 2] [12 79]]`).

Artifacts: `results/bert_reproducible_results.txt` plus the figures in
`results/reproducible_evaluation.csv`. The fine-tuned checkpoint itself is
large (LFS) and was not committed; rerun `python src/classifier/train_bert.py
--train data/train.csv --val data/val.csv --test data/test.csv --epochs 3` to
reproduce.

### Model Performance

#### BERT Classifier (Pre-trained)

- **Overall Accuracy**: 98.92% on test set
- **Ham Detection**: 99.17% precision, 99.59% recall
- **Spam Detection**: 97.24% precision, 94.63% recall

#### Baseline Models

- Comparison results available in `results/baseline_model_comparison.csv`

### Qwen3-4B Generator Training

- **Training Time**: ~10-15 minutes on T4 GPU
- **Memory Usage**: ~3GB efficient utilization
- **Training Loss**: Decreased from 3.18 → 1.54
- **Dataset**: 598 spam examples processed

### Adversarial Training Results

Training history and performance improvements through adversarial iterations are tracked in `results/adversarial_results/`.

#### Adversarial Training Results (Qwen3-4B + BERT Classifier)

**Training Setup:**

- Initial BERT classifier accuracy: **98.9%**
- Generator: Fine-tuned Qwen3-4B with LoRA adapters
- Training iterations: 3 rounds of adversarial generation and retraining
- Total adversarial samples generated: **142 challenging samples**

**Performance Progression:**

| Iteration | Detection Rate | Test Accuracy | Samples Generated | Training Set Size |
| --------- | -------------- | ------------- | ----------------- | ----------------- |
| Initial   | -              | 98.9%         | 0                 | 4,449             |
| 1         | 91.7%          | 98.9%         | 48                | 4,497             |
| 2         | 91.7%          | 99.3%         | 48                | 4,545             |
| 3         | 95.7%          | 96.5%         | 46                | 4,591             |

**Historical results and evaluation limitations:**

These numbers and the checked-in model artifacts predate the evaluation-split fix.
They have not been rerun or validated with the new partitioning. The Colab
notebooks remain historical experiments and do not use the corrected CLI loader.

The reported adversarial detection rate rises from 91.7% to 95.7% (4.0 percentage
points), while clean accuracy falls from 98.9% to 96.5% (2.4 percentage points).
Different generated attack samples were used across iterations, so these figures
alone do not establish a robustness improvement on a fixed held-out attack set.
The cause of the clean-accuracy change has not been isolated experimentally.

**Current CLI evaluation contract:**

- Both BERT trainers select checkpoints using validation data. The baseline
  runner also selects its model on validation before the final test evaluation.
- An explicit `--val` path must exist. When omitted, `val.csv` beside the training
  file is used; otherwise 15% of the remaining training rows become validation.
  This fallback does not imply a 70/15/15 split of the original dataset.
- Duplicate matching ignores case and repeated whitespace. Overlap is removed
  from training/validation, preserving the test file. Conflicting labels and
  unusable partitions fail explicitly. This does not detect semantic near-duplicates.
- Previously used test data is not a new independent holdout simply because the
  scripts changed. New performance claims require a fresh, documented evaluation.

Run the offline regression tests with `python -m unittest discover -s tests -v`.
They use synthetic data and mocked trainers; they do not download models or train them.

**Generated Data Logging:**

- All adversarial samples saved with metadata in JSON format
- Per-iteration files: `data/adversarial_iteration_X_data.json`
- Cumulative dataset: `data/all_adversarial_data.json`
- Includes predictions, timestamps, and prompt classifications

This demonstrates successful implementation of adversarial training where the generator creates challenging but realistic spam samples, and the classifier iteratively improves its robustness while maintaining high accuracy on legitimate test data.

## Technologies Used

- **Machine Learning**: scikit-learn, transformers, torch
- **Data Processing**: pandas, numpy, nltk
- **Model Training**: Hugging Face transformers, PEFT (LoRA)
- **Visualization**: matplotlib, seaborn
- **Infrastructure**: Google Colab T4 GPU, 4-bit quantization
