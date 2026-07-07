import pandas as pd
from sklearn.preprocessing import StandardScaler

def add_time_features(df):
    df = df.copy()
    if 'Time' in df.columns:
        df['Hour'] = (df['Time'] // 3600) % 24
        df['Day'] = df['Time'] // (3600 * 24)
    return df

def prepare_features(df, scaler=None, fit_scaler=False):
    df = add_time_features(df)
    
    # Safely separate target and drop Time
    y = df['Class'] if 'Class' in df.columns else None
    drop_cols = ['Class', 'Time']
    X = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    
    if fit_scaler:
        scaler = StandardScaler()
        X['Amount'] = scaler.fit_transform(X[['Amount']])
    elif scaler is not None:
        X['Amount'] = scaler.transform(X[['Amount']])
        
    return X, y, scaler