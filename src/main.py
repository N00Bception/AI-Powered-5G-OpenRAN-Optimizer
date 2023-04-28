import argparse
from datetime import datetime
from utils import config
from utils.logger import Logger
from data_preparation.data_extraction import extract_data
from data_preparation.data_cleaning import clean_data
from data_preparation.data_transformation import transform_data
from models.predictive_network_planning.predict import make_predictions

def main(args):
    # Set up logger
    log_file = f"{config.LOGS_DIR}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    logger = Logger(log_file)
    
    # Extract data
    logger.log("Extracting data...")
    raw_data = extract_data(args.data_file)
    
    # Clean data
    logger.log("Cleaning data...")
    clean_data = clean_data(raw_data)
    
    # Transform data
    logger.log("Transforming data...")
    transformed_data = transform_data(clean_data)
    
    # Make predictions
    logger.log("Making predictions...")
    predictions = make_predictions(transformed_data)
    
    # Save predictions to file
    logger.log("Saving predictions to file...")
    predictions_file = f"{config.PREDICTIONS_DIR}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
    predictions.to_csv(predictions_file, index=False)
    
    logger.log("Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main program.")
    parser.add_argument("data_file", type=str, help="Path to the raw data file.")
    args = parser.parse_args()
    
    main(args)

