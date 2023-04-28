import argparse
import joblib
import numpy as np
from utils import load_data, preprocess_data

# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to saved model file")
parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
args = parser.parse_args()

# Load saved model
model = joblib.load(args.model_path)

# Load and preprocess data
data = load_data(args.data_path)
preprocessed_data = preprocess_data(data)

# Make predictions
predictions = model.predict(preprocessed_data)

# Print predicted labels
print("Predictions:")
print(predictions)
