"""
SMS Spam Data Preprocessing Module

This script loads the raw SMS Spam Collection dataset, cleans the text,
and splits it into training and testing sets.
"""

import pandas as pd
import re
import string
import nltk
from sklearn.model_selection import train_test_split
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def load_sms_data(filepath='data/SMSSpamCollection'):
    """
    Load SMS Spam Collection dataset from tab-separated file.
    
    Args:
        filepath (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: DataFrame with 'label' and 'message' columns
    """
    print(f"Loading dataset from {filepath}...")
    
    # Read tab-separated file
    df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    
    print(f"Dataset loaded: {len(df)} messages")
    print(f"Labels distribution:")
    print(df['label'].value_counts())
    
    return df

def clean_text(text):
    """
    Clean and preprocess SMS text.
    
    Args:
        text (str): Raw SMS message
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove phone numbers (simple pattern)
    text = re.sub(r'\b\d{10,}\b', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation but keep some meaningful chars
    text = text.translate(str.maketrans('', '', string.punctuation.replace('!', '').replace('?', '')))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words and len(word) > 1]
    
    return ' '.join(filtered_text)

def preprocess_dataset(df):
    """
    Apply text cleaning to the entire dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    print("Cleaning text messages...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Clean messages
    df_clean['message'] = df_clean['message'].apply(clean_text)
    
    # Remove empty messages after cleaning
    df_clean = df_clean[df_clean['message'].str.len() > 0]
    
    # Convert labels to binary (0=ham, 1=spam)
    df_clean['label_binary'] = df_clean['label'].map({'ham': 0, 'spam': 1})
    
    print(f"Cleaning complete. {len(df_clean)} messages remaining.")
    
    return df_clean

def split_and_save_data(df, val_size=0.15, test_size=0.15, random_state=42, output_dir='data'):
    """
    Split dataset into separate train, validation, and untouched test partitions,
    and verify zero cross-partition duplicate message leakage.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        val_size (float): Fraction of data for validation (checkpoint selection)
        test_size (float): Fraction of data for final held-out testing
        random_state (int): Random seed for reproducibility
        output_dir (str): Directory to save train.csv, val.csv, and test.csv
    """
    train_size = 1.0 - val_size - test_size
    print(f"Splitting dataset: {train_size*100:.0f}% train, {val_size*100:.0f}% val, {test_size*100:.0f}% test")
    
    # Check for duplicates within the dataset
    initial_len = len(df)
    df_dedup = df.drop_duplicates(subset=['message']).copy()
    if len(df_dedup) < initial_len:
        print(f"Deduplicated messages: removed {initial_len - len(df_dedup)} duplicate rows to prevent leakage")
    
    X = df_dedup['message']
    y = df_dedup['label_binary']
    
    # First split off the untouched test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Next split the train and validation sets
    relative_val_size = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=random_state, stratify=y_train_val
    )
    
    # Assert zero cross-partition overlap
    train_msgs = set(X_train)
    val_msgs = set(X_val)
    test_msgs = set(X_test)
    
    assert len(train_msgs.intersection(val_msgs)) == 0, "Data leakage detected: message overlap between train and val"
    assert len(train_msgs.intersection(test_msgs)) == 0, "Data leakage detected: message overlap between train and test"
    assert len(val_msgs.intersection(test_msgs)) == 0, "Data leakage detected: message overlap between val and test"
    print("Partition integrity verified: 0 message overlap across train, val, and test.")
    
    # Create DataFrames
    train_df = pd.DataFrame({'message': X_train, 'label': y_train})
    val_df = pd.DataFrame({'message': X_val, 'label': y_val})
    test_df = pd.DataFrame({'message': X_test, 'label': y_test})
    
    # Save to CSV files
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train set saved: {train_path} ({len(train_df)} samples)")
    print(f"Validation set saved: {val_path} ({len(val_df)} samples)")
    print(f"Test set saved: {test_path} ({len(test_df)} samples)")
    
    # Print distribution
    print("\nTrain set distribution:")
    print(train_df['label'].value_counts())
    print("\nValidation set distribution:")
    print(val_df['label'].value_counts())
    print("\nTest set distribution:")
    print(test_df['label'].value_counts())
    
    return train_df, val_df, test_df

def main():
    """Main preprocessing pipeline."""
    print("=== SMS Spam Data Preprocessing ===\n")
    
    # Load raw data
    df = load_sms_data()
    
    # Clean and preprocess
    df_clean = preprocess_dataset(df)
    
    # Split and save 3-way partitions
    train_df, val_df, test_df = split_and_save_data(df_clean)
    
    print("\nData preprocessing complete!")

if __name__ == "__main__":
    main() 