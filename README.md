# Adversarial Spam Detection

A machine learning project implementing adversarial training for robust SMS spam detection. This project by Group 5D for AI4ALL explores how iterative adversarial training can improve spam classifier resilience against sophisticated attacks.

## Project Overview

This system implements a complete adversarial training pipeline that alternates between training spam classifiers and generating adversarial spam examples. The goal is to create more robust spam detection models that can handle evolving spam techniques.

### Key Components

- **Baseline Classifiers**: Traditional ML models (Naive Bayes, Logistic Regression, SVM)
- **Advanced BERT Classifier**: Pre-trained transformer model achieving 98.92% accuracy
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

This creates `train.csv` and `test.csv` files in the `data/` directory.

### Training Baseline Models

Train traditional machine learning classifiers:

```bash
python src/classifier/train_baseline.py --train data/train.csv --test data/test.csv
```

### Training BERT Model

Fine-tune BERT for spam classification:

```bash
python src/classifier/train_bert.py --train data/train.csv --test data/test.csv --epochs 3
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

- Loads pre-trained BERT classifier (98.92% accuracy)
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

1. **Initial Setup**: Load pre-trained BERT classifier (98.92% baseline accuracy)
2. **Generator Loading**: Load fine-tuned Qwen3-4B spam generator with LoRA adapters
3. **Generation Phase**: Use Qwen3-4B to generate sophisticated adversarial spam samples
4. **Evaluation Phase**: Test BERT classifier performance on generated samples
5. **Retraining Phase**: Fine-tune BERT classifier with adversarial samples using LoRA
6. **Iteration**: Repeat process for 3 iterations to continuously improve robustness

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

*Results from final adversarial training loop to be updated after completion...*

**Expected Improvements:**

- Higher baseline accuracy (98.92% vs 96.3%)
- More sophisticated adversarial sample generation
- Better model robustness due to BERT's advanced language understanding
- Faster training with Qwen3-4B's optimized architecture

---

*Final adversarial loop results will be added here upon completion of the enhanced training pipeline.*

## Technologies Used

- **Machine Learning**: scikit-learn, transformers, torch
- **Data Processing**: pandas, numpy, nltk
- **Model Training**: Hugging Face transformers, PEFT (LoRA)
- **Visualization**: matplotlib, seaborn
- **Infrastructure**: Google Colab T4 GPU, 4-bit quantization
