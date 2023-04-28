import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models.network_anomaly_detection import NetworkAnomalyDetection
from utils import NetworkAnomalyDataset, load_data, train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network anomaly detection model')
    parser.add_argument('--data_path', type=str, default='data/network_data.csv', help='Path to network data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()

    # Load and preprocess data
    df = pd.read_csv(args.data_path)
    data = load_data(df)

    # Split data into training and validation sets
    train_set = NetworkAnomalyDataset(data[:8000])
    val_set = NetworkAnomalyDataset(data[8000:])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # Instantiate model and optimizer
    model = NetworkAnomalyDetection(input_size=6, hidden_size=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    train(model, train_loader, val_loader, optimizer, num_epochs=args.num_epochs)

    # Save trained model
    torch.save(model.state_dict(), 'network_anomaly_detection.pt')
