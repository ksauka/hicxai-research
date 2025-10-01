"""Preprocessing utilities for the Adult dataset.

Exports:
- preprocess_adult(df): returns a cleaned, numeric DataFrame with an 'income' label column.
"""

from typing import List
import numpy as np
import pandas as pd


def _strip_and_normalize_strings(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.strip()
            .replace({'?': 'Unknown'})
        )
    return df


def preprocess_adult(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode Adult dataset into numeric features.

    Input:
        df: DataFrame containing Adult columns including 'income'.
    Output:
        DataFrame with numeric features; 'income' remains as the target label.
    """
    df = df.copy()

    if 'income' not in df.columns:
        raise ValueError("Expected 'income' column in Adult dataframe")

    # Normalize string columns
    object_cols = [c for c in df.columns if df[c].dtype == 'object']
    df[object_cols] = df[object_cols].fillna('Unknown')
    df = _strip_and_normalize_strings(df, object_cols)

    # Ensure common numeric cols are numeric
    numeric_candidates = [
        'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Fill NaNs: numeric with median, categorical with mode/Unknown
    for c in df.columns:
        if c == 'income':
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            # Calculate median, but use a default value if median is NaN (empty column)
            median_val = df[c].median()
            if pd.isna(median_val):
                # Use sensible defaults for numeric columns if median is NaN
                if c == 'age':
                    median_val = 35
                elif c == 'fnlwgt':
                    median_val = 100000
                elif c == 'education_num':
                    median_val = 9  # HS-grad equivalent
                elif c in ['capital_gain', 'capital_loss']:
                    median_val = 0
                elif c == 'hours_per_week':
                    median_val = 40
                else:
                    median_val = 0  # Default fallback
            df[c] = df[c].fillna(median_val)
        else:
            df[c] = df[c].fillna('Unknown')

    # One-hot encode categorical features except the target
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c != 'income']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Keep label as string categories; sklearn supports string labels
    # Ensure 'income' column is last for readability
    cols = [c for c in df_encoded.columns if c != 'income'] + ['income']
    df_encoded = df_encoded[cols]

    return df_encoded
