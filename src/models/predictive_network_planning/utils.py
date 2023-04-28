import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(data_path):
    """
    Load data from csv file.

    Args:
        data_path (str): Path to the csv file.

    Returns:
        X (numpy array): Features of the dataset.
        y (numpy array): Labels of the dataset.
    """
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def preprocess_data(X_train, X_test):
    """
    Preprocess data using standard scaling.

    Args:
        X_train (numpy array): Features of the training set.
        X_test (numpy array): Features of the test set.

    Returns:
        X_train_scaled (numpy array): Scaled features of the training set.
        X_test_scaled (numpy array): Scaled features of the test set.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

