import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import CustomDataset

class EnergyEfficiencyModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EnergyEfficiencyModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = train_loss/len(train_loader)
    return avg_loss

def main():
    # Hyperparameters
    input_size = 8
    hidden_size = 64
    output_size = 1
    learning_rate = 0.01
    num_epochs = 50
    batch_size = 16

    # Load data
    train_dataset = CustomDataset("data/energy_efficiency/train.csv")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnergyEfficiencyModel(input_size, hidden_size, output_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train model
    criterion = torch.nn.MSELoss()
    for epoch in range(1, num_epochs+1):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        print('Epoch: [{}/{}]\tTrain Loss: {:.4f}'.format(epoch, num_epochs, loss))

    # Save model
    torch.save(model.state_dict(), "models/energy_efficiency.pt")

if __name__ == '__main__':
    main()

