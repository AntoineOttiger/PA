import torch
import torch.nn as nn
import torch.optim as optim

import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt

#Ajoute automatiquement le dossier parent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#My packages
from src.models.mlp_models import get_model
from utilities import (
    load_config,
    load_prepared
)

def train_model(model, device, X_train, y_train, X_val, y_val, batch_size=32, num_epochs= 1000, patience=20, lr=0.001):

    print(next(model.parameters()).device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    from torch.utils.data import TensorDataset, DataLoader


    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    epochs = []
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')  # Initialize with a very high value
    epochs_no_improve = 0  # Counter to track epochs without improvement
    best_model_weights = None  # To store the best model's weights

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward
            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)  # accumulate total loss

        avg_train_loss = epoch_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():  # No need to compute gradients during validation
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)            

                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        epochs.append(epoch+1)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict()  # Save the best model weights
            epochs_no_improve = 0  # Reset counter if improvement
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Restore the best model weights after training
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return epochs, train_losses, val_losses


def save_loss_plot(epochs, train_losses, val_losses, plot_path):

    plt.plot(epochs, train_losses, color="blue", label="train")
    plt.plot(epochs, val_losses, color="orange", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("loss (MSE)")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()


def train(config_file):

    mlflow.set_tag("phase", "train")
    mlflow.set_tag("script", "train.py")

    #load config/data
    cfg = load_config(f"config/{config_file}")
    data_path = f"data/prepared/{config_file.split(".")[0]}/"
    X_train, y_train, X_val, y_val = load_prepared(data_path, type="train")

    #to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    X_size = X_train.shape[-1]
    model = get_model(
        cfg['model']['model_name'], X_size
    ).to(device)
    
    #Save pipeline parameters
    mlflow.log_params(cfg['model'])
    mlflow.log_params(cfg['training'])
    mlflow.log_params(cfg['dataset'])
    mlflow.log_params(cfg['prepare'])

    #Train
    epochs, train_losses, val_losses = train_model(model, device, X_train, y_train, X_val, y_val, 
                                            batch_size=cfg['training']['batch_size'], 
                                            num_epochs=cfg['training']['epochs'], 
                                            patience=cfg['training']['patience'], 
                                            lr=cfg['training']['learning_rate'])

    plot_path = "temp/loss.png"
    save_loss_plot(epochs, train_losses, val_losses, plot_path)
    mlflow.log_artifact(plot_path, artifact_path="train")
    
    artifact_path = "model"
    mlflow.pytorch.log_model(model, artifact_path)
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    
    print("Train done")

    return model_uri



if __name__ == "__main__":
    """
    train("exp1.yaml")
    """
    None
