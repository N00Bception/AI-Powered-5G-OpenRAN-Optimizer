import torch.nn as nn

class EnergyEfficiencyModel(nn.Module):
    """
    PyTorch model for energy efficiency optimization in 5G OpenRAN.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(EnergyEfficiencyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

