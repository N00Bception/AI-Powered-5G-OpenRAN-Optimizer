import logging
from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from models.predictive_network_planning.model import PredictiveNetworkPlanningModel
from utils import load_model, predict_single_sample


# Define the input data schema
class InputData(BaseModel):
    input: List[float]


# Define the response schema
class PredictionResult(BaseModel):
    prediction: float


# Initialize the FastAPI app
app = FastAPI()


# Load the trained model at startup
model = PredictiveNetworkPlanningModel(input_dim=4, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.5)
load_model(model, "models/predictive_network_planning/best_model.pt")


@app.post("/predict", response_model=PredictionResult)
async def predict(data: InputData) -> Dict[str, float]:
    # Extract the input data
    input_data = data.input

    # Predict on the input data
    output = predict_single_sample(model, input_data)

    # Create the response
    response = {"prediction": output}

    return response


# Set up logging
logging.basicConfig(filename="logs/api.log", level=logging.INFO)


if __name__ == "__main__":
    import uvicorn
    
    # Start the FastAPI app using Uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info")

