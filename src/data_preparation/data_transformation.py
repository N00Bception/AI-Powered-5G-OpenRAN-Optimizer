import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.utils.config import read_config


def transform_data(input_path: str, output_path: str) -> None:
    # Load data from input file
    df = pd.read_csv(input_path)

    # Apply data transformations
    config = read_config("config.yml")
    for col in config["data_transformation"]["columns"]:
        if col["type"] == "log":
            df[col["name"]] = df[col["name"]].apply(lambda x: np.log(x) if x > 0 else 0)
        elif col["type"] == "sqrt":
            df[col["name"]] = df[col["name"]].apply(lambda x: np.sqrt(x) if x > 0 else 0)
        elif col["type"] == "inverse":
            df[col["name"]] = df[col["name"]].apply(lambda x: 1/x if x > 0 else 0)

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Save transformed data to output file
    pd.DataFrame(scaled_data, columns=df.columns).to_csv(output_path, index=False)

