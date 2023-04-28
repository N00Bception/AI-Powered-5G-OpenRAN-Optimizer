import numpy as np
import pandas as pd

def load_data(file_path):
    """
    Load data from a given file path.
    Args:
        file_path (str): The path to the file to load.
    Returns:
        data (pd.DataFrame): A pandas DataFrame containing the loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the loaded data.
    Args:
        data (pd.DataFrame): A pandas DataFrame containing the loaded data.
    Returns:
        processed_data (np.ndarray): A numpy array containing the preprocessed data.
    """
    # Perform some data preprocessing steps, such as scaling, normalization, etc.
    processed_data = data.to_numpy()
    processed_data = np.divide(processed_data, 100)
    return processed_data

def save_data(data, file_path):
    """
    Save processed data to a given file path.
    Args:
        data (np.ndarray): A numpy array containing the processed data.
        file_path (str): The path to the file to save.
    """
    pd.DataFrame(data).to_csv(file_path, index=False)

