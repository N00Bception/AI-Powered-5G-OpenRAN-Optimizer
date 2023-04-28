import unittest
import pandas as pd
from src.utils.data_preparation.data_cleaning import clean_data
from src.utils.data_preparation.data_extraction import extract_data
from src.utils.data_preparation.data_transformation import transform_data


class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.raw_data = pd.read_csv("tests/test_data/raw_data.csv")

    def test_clean_data(self):
        # Test data cleaning function
        cleaned_data = clean_data(self.raw_data)
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertEqual(len(cleaned_data), 4)
        self.assertEqual(cleaned_data.isna().sum().sum(), 0)

    def test_extract_data(self):
        # Test data extraction function
        extracted_data = extract_data(self.raw_data)
        self.assertIsInstance(extracted_data, pd.DataFrame)
        self.assertEqual(len(extracted_data), 4)
        self.assertEqual(len(extracted_data.columns), 3)

    def test_transform_data(self):
        # Test data transformation function
        transformed_data = transform_data(self.raw_data)
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertEqual(len(transformed_data), 4)
        self.assertEqual(len(transformed_data.columns), 2)

if __name__ == '__main__':
    unittest.main()

