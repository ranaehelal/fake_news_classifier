import os
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")

def save_cleaned_data(df: pd.DataFrame, filename: str):
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path, index=False)
    print(f" Saved cleaned data to {path}")


def load_cleaned_data(filename: str) -> pd.DataFrame:

    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f" File does not exist: {path}")
    print(f"Loaded cleaned data from {path}")
    return pd.read_csv(path)
