import os

import joblib
import pandas as pd
import pickle

PROCESSED_DIR = os.path.join("data", "processed")

def save_cleaned_data(df: pd.DataFrame, filename: str):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path, index=False)
    print(f" Saved cleaned data to {path}")


def load_cleaned_data(filename: str) -> pd.DataFrame:

    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f" File does not exist: {path}")
    print(f"Loaded cleaned data from {path}")
    return pd.read_csv(path)

def save_model(model, path):
   if hasattr(model, "save_model"):
      model.save_model(path)
   else:
       joblib.dump(model, path)

def load_pickle(path):
    with open(path, 'rb') as f:
        print("Loading tokenizer from", path)
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print("tokenizer saved to", path)




from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    val_relative_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=random_state)

    print("Data split into train, validation, and test sets:")

    return X_train, X_val, X_test, y_train, y_val, y_test
