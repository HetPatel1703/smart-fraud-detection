import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath='data/creditcard.csv'):
    """Load the credit card dataset."""
    df = pd.read_csv(filepath)
    return df

def temporal_split(df, test_size=0.2, random_state=42):
    """
    Split by time to avoid leakage.
    Sorts by 'Time' and takes the last test_size fraction as test set.
    """
    df = df.sort_values('Time').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

if __name__ == "__main__":
    # quick test
    df = load_data()
    train, test = temporal_split(df)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")