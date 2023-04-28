import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocess_data(data_file_path: str, scaler_type: str = 'min_max') -> pd.DataFrame:
    # Load data
    data = pd.read_csv(data_file_path)

    # Drop irrelevant columns
    data.drop(['id', 'timestamp'], axis=1, inplace=True)

    # Apply scaling
    if scaler_type == 'min_max':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

    # Split into input and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    return X, y

