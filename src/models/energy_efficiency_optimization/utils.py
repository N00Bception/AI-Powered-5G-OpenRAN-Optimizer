import numpy as np
import pandas as pd
from typing import List

def read_data(filename: str) -> pd.DataFrame:
    """
    Reads the dataset from the given filename and returns a pandas DataFrame object
    """
    df = pd.read_csv(filename)
    return df

def prepare_data(df: pd.DataFrame, input_features: List[str], target: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses the dataset and returns the input and target arrays
    """
    X = df[input_features].values
    y = df[target].values
    return X, y

def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normalizes the input data to have zero mean and unit variance
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm

def denormalize_data(X_norm: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Denormalizes the input data back to its original scale
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_denorm = X_norm * std + mean
    return X_denorm

def save_model(model, filename: str) -> None:
    """
    Saves the trained model to disk
    """
    joblib.dump(model, filename)

def load_model(filename: str):
    """
    Loads the trained model from disk
    """
    return joblib.load(filename)


