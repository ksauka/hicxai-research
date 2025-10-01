
import pandas as pd
import numpy as np
import os
import json

def load_adult_data(data_dir, balance=False, discretize=True):
    """
    Load the Adult dataset with robust feature handling, adapted from XAgent/Agent/utils.py.
    """
    data_path = os.path.join(data_dir, 'adult.data')
    json_path = os.path.join(os.path.dirname(data_dir), 'dataset_info', 'adult.json')
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
        'hours_per_week', 'native_country', 'income'
    ]
    df = pd.read_csv(data_path, names=columns, skipinitialspace=True)
    # Remove rows with missing values (marked as '?')
    df = df.replace('?', np.nan)
    df = df.dropna()
    # Convert numerical columns to appropriate types
    num_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Optionally encode categorical variables using one-hot encoding
    cat_cols = [
        'workclass', 'education', 'marital_status', 'occupation',
        'relationship', 'race', 'sex', 'native_country'
    ]
    if discretize:
        df = pd.get_dummies(df, columns=cat_cols)
    # Encode target
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
    # Load metadata
    with open(json_path, 'r') as f:
        meta = json.load(f)
    # Add feature names, types, and valid values to meta if missing
    meta.setdefault('num_features', num_cols)
    meta.setdefault('cat_features', cat_cols)
    for cat in cat_cols:
        meta.setdefault('feature_values', {})
        meta['feature_values'][cat] = sorted(df[cat].dropna().unique().tolist()) if cat in df else []
    # Add feature ranges for numeric features
    meta.setdefault('feature_ranges', {})
    for num in num_cols:
        if num in df:
            meta['feature_ranges'][num] = (float(df[num].min()), float(df[num].max()))
    return df, meta

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    df, meta = load_adult_data(data_dir)
    print('Data shape:', df.shape)
    print('Metadata:', meta)
