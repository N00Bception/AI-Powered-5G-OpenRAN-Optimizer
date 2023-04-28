import os
import pandas as pd
from urllib.request import urlretrieve
from zipfile import ZipFile
from utils.logger import get_logger


class DataExtractor:
    """Class for extracting data from remote sources"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.logger = get_logger(__name__)
        
    def extract_data(self, url, filename):
        """
        Downloads and extracts data from remote source
        
        Args:
            url (str): URL to download data from
            filename (str): Name of the file to save data to
        """
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            self.logger.info(f"Downloading data from {url}")
            urlretrieve(url, file_path)
            self.logger.info("Data downloaded successfully")

        with ZipFile(file_path, 'r') as zip_obj:
            self.logger.info(f"Extracting data from {filename}")
            zip_obj.extractall(self.data_dir)
            self.logger.info("Data extracted successfully")

    def read_csv(self, file_path):
        """
        Reads CSV data into a pandas dataframe
        
        Args:
            file_path (str): Path to CSV file
        
        Returns:
            pandas.DataFrame: DataFrame containing data from CSV file
        """
        self.logger.info(f"Reading data from {file_path}")
        df = pd.read_csv(file_path)
        self.logger.info("Data loaded successfully")
        return df

