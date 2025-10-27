"""Data loading, cleaning, feature engineering, and leak-safe splits."""
import pandas as pd
from pathlib import Path

def load_sample(path: str = 'data/crypto_prices_sample.csv') -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret_1'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    return df

def leak_safe_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

if __name__ == '__main__':
    df = load_sample()
    df = add_ta_features(df)
    tr, va, te = leak_safe_split(df)
    print(f"Train={len(tr)} Val={len(va)} Test={len(te)}")
