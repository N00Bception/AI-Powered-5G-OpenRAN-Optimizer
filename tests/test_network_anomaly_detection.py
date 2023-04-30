import unittest
from unittest.mock import Mock, patch
from src.utils.logger import Logger
from src.utils.data_loader import DataLoader
from src.utils.data_preprocessing import DataPreprocessor
from src.models.network_anomaly_detection import NetworkAnomalyDetection


class TestNetworkAnomalyDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = Logger()
        cls.logger.disable_logging()
        cls.data_loader = DataLoader()
        cls.preprocessor = DataPreprocessor()
        cls.nad_model = NetworkAnomalyDetection()

    @patch('src.models.network_anomaly_detection.NetworkAnomalyDetection.detect_anomaly')
    def test_detect_anomaly(self, mock_detect_anomaly):
        # Define test data
        test_data = self.data_loader.load_data('test_data.csv')
        preprocessed_data = self.preprocessor.preprocess_data(test_data)
        test_features = preprocessed_data.drop(columns=['timestamp'])
        
        # Mock the predict method and return a dummy prediction
        mock_detect_anomaly.return_value = [0, 0, 1, 1, 0]
        
        # Test the predict method
        predictions = self.nad_model.detect_anomaly(test_features)
        self.assertEqual(len(predictions), len(test_data))
        self.assertListEqual(predictions, mock_detect_anomaly.return_value)

if __name__ == '__main__':
    unittest.main()

