import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.models.predictive_network_planning.predictive_network_planning import PredictiveNetworkPlanning


class TestPredictiveNetworkPlanning:

    @pytest.fixture(scope='module')
    def network_planning_model(self):
        model = PredictiveNetworkPlanning()
        return model

    def test_model_loads(self, network_planning_model):
        assert network_planning_model is not None

    def test_model_predict(self, network_planning_model):
        # mock input data
        input_data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        expected_output = np.array([0, 1])

        # mock predict function of the model
        with patch.object(network_planning_model, 'predict') as mock_predict:
            mock_predict.return_value = expected_output

            # call predict function with input data
            output = network_planning_model.predict(input_data)

            # assert output is as expected
            assert np.array_equal(output, expected_output)
            mock_predict.assert_called_once_with(input_data)

    def test_model_train(self, network_planning_model):
        # mock input and target data
        input_data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        target_data = np.array([0, 1])

        # mock fit function of the model
        with patch.object(network_planning_model, 'fit') as mock_fit:
            mock_fit.return_value = MagicMock()

            # call fit function with input and target data
            network_planning_model.fit(input_data, target_data)

            # assert fit function was called once with correct arguments
            mock_fit.assert_called_once_with(input_data, target_data)

