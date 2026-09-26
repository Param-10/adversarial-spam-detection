"""Shared partition loading for checkpoint and model selection."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def message_keys(frame):
    return frame['message'].str.casefold().str.split().str.join(' ')


def validate_frame(frame, name):
    if not {'message', 'label'}.issubset(frame.columns):
        raise ValueError(f'{name} must contain message and label columns')
    if frame.empty or frame['message'].isna().any():
        raise ValueError(f'{name} must contain non-null messages')
    if not frame['message'].map(lambda value: isinstance(value, str)).all():
        raise ValueError(f'{name} messages must be strings')
    if message_keys(frame).eq('').any() or not frame['label'].isin([0, 1]).all():
        raise ValueError(f'{name} requires non-empty messages and binary labels')


def load_data(train_path='data/train.csv', test_path='data/test.csv', val_path=None,
              val_ratio=0.15, random_state=42):
    """Preserve the test rows; remove overlap from training/validation only.

    An explicit validation path must exist. Otherwise use val.csv beside the
    training file, or split validation from the remaining training rows.
    """
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be between 0 and 1')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    candidate = Path(val_path) if val_path is not None else Path(train_path).with_name('val.csv')
    if val_path is not None:
        try:
            val_df = pd.read_csv(candidate)
        except FileNotFoundError as error:
            raise FileNotFoundError(f'Validation file not found: {candidate}') from error
    else:
        val_df = pd.read_csv(candidate) if candidate.exists() else None
    frames = [('train', train_df), ('test', test_df)]
    if val_df is not None:
        frames.append(('validation', val_df))
    for name, frame in frames:
        validate_frame(frame, name)

    combined = pd.concat([frame for _, frame in frames], ignore_index=True)
    if combined.groupby(message_keys(combined))['label'].nunique().gt(1).any():
        raise ValueError('Identical normalized messages have conflicting labels; resolve them before training')

    def exclude(frame, reserved, name):
        keys = message_keys(frame)
        kept = frame.loc[~keys.isin(reserved) & ~keys.duplicated()].reset_index(drop=True)
        removed = len(frame) - len(kept)
        if removed:
            print(f'{name}: removed {removed} duplicate/overlapping rows')
        return kept

    test_keys = set(message_keys(test_df))
    train_df = exclude(train_df, test_keys, 'Training')
    if val_df is None:
        train_df, val_df = train_test_split(
            train_df, test_size=val_ratio, random_state=random_state,
            stratify=train_df['label'],
        )
    else:
        val_df = exclude(val_df, test_keys, 'Validation')
        train_df = exclude(train_df, set(message_keys(val_df)), 'Training')

    for name, frame in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        validate_frame(frame, name)
        if set(frame['label']) != {0, 1}:
            raise ValueError(f'{name} must retain both classes after partition cleanup')
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df
