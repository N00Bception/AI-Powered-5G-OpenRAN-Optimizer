import argparse
import logging
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from models.predictive_network_planning.model import PredictiveNetworkPlanningModel
from models.predictive_network_planning.utils import PNPDataset, collate_fn

def train(model: nn.Module,
          train_data: PNPDataset,
          valid_data: PNPDataset,
          epochs: int,
          batch_size: int,
          lr: float,
          weight_decay: float,
          device: str,
          save_path: str) -> Tuple[List[float], List[float]]:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_loss = []
    valid_loss = []
    best_valid_loss = np.inf
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_data)
        train_loss.append(epoch_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(valid_data)
            valid_loss.append(epoch_loss)

            if epoch_loss < best_valid_loss:
                best_valid_loss = epoch_loss
                torch.save(model.state_dict(), save_path)

        logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss[-1]:.4f}, Valid Loss: {valid_loss[-1]:.4f}")

    return train_loss, valid_loss

def main(args: argparse.Namespace) -> None:
    # Load data
    train_data = PNPDataset(data_file=args.train_data_file)
    valid_data = PNPDataset(data_file=args.valid_data_file)

    # Initialize model
    model = PredictiveNetworkPlanningModel(input_dim=train_data.input_dim,
                                           hidden_dim=args.hidden_dim,
                                           num_layers=args.num_layers,
                                           output_dim=train_data.output_dim,
                                           dropout=args.dropout)
    device = torch.device(args.device)
    model.to(device)

    # Train model
    train_loss, valid_loss = train(model=model,
                                    train_data=train_data,
                                    valid_data=valid_data,
                                    epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    device=device,
                                    save_path=args.save_path)

    # Save training and validation loss
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"{args.log_prefix}_{timestamp}.log")
    
    with open(log_file, "w") as f:
        f.write(f"Training Loss: {train_loss}\n")
        f.write(f"Validation Loss: {valid_loss}\n")
    
