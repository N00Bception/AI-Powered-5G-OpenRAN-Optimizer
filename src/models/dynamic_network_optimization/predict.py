import torch
import numpy as np
from utils import load_data, preprocess_data
from model import NetworkOptimizer

def predict(model_path, data_path):
    # Load the model from the specified path
    model = NetworkOptimizer()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the data from the specified path
    data = load_data(data_path)
    preprocessed_data = preprocess_data(data)

    # Convert the preprocessed data to a tensor
    input_tensor = torch.from_numpy(preprocessed_data).float()

    # Make predictions using the model
    with torch.no_grad():
        output_tensor = model(input_tensor)
        output = output_tensor.numpy()

    # Postprocess the predictions and return the results
    result = postprocess_predictions(output)
    return result

def postprocess_predictions(predictions):
    # Convert the predictions to a list of recommended actions
    actions = []
    for prediction in predictions:
        if prediction > 0.5:
            actions.append("Add a new network node")
        else:
            actions.append("Remove an existing network node")
    return actions
