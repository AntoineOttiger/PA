import torch
import torch.nn as nn
import torch.optim as optim

import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import norm
import sys
import os
import pandas as pd
import json
from scipy.interpolate import RegularGridInterpolator

#Ajoute automatiquement le dossier parent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#My packages
from src.models.mlp_models import get_model
from utilities import (
    load_config,
    load_prepared
)
from modules.Abaqus_Import_matrices_v1 import(
    build_matrix
)

def train_model(model, cfg, config_file , device, X_train, y_train, X_val, y_val, batch_size=32, num_epochs= 1000, patience=50, lr=0.001):

    print(next(model.parameters()).device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    from torch.utils.data import TensorDataset, DataLoader, Subset


    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


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

        for batch_ind, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward
            pred = model(batch_X)
            
            if cfg["training"]["loss"] == "mse" :
                loss = criterion(pred, batch_y)
            
            elif cfg["training"]["loss"] == "FEM-based" :
                config_name = config_file.split(".")[0]
                mtx_path = os.path.join("data", "prepared", config_name, "train_mtx")
                mtx_indexes = [i for i in range(batch_ind*batch_size, batch_ind*batch_size+ batch_size)]
                norms_fitness = []
                for index, i in enumerate(mtx_indexes) :
                    sample_mtx_path = os.path.join(mtx_path, str(i))
                    mass_raw = pd.read_csv(os.path.join(sample_mtx_path, "mass_raw.csv"))
                    stif_raw = pd.read_csv(os.path.join(sample_mtx_path, "stif_raw.csv"))
                    X_raw = pd.read_csv(os.path.join(sample_mtx_path, "X_raw.csv"))
                    with open(os.path.join(sample_mtx_path,"eigen_f_Hz_lst.json"), "r") as f:
                        eigen_f_Hz_lst = json.load(f)
                    with open(os.path.join(sample_mtx_path,"mode_lst.json"), "r") as f:
                        mode_lst = json.load(f)

                    # Position dans la liste des modes
                    mode_nb = 0
                    # sélection du mode la colonne Frame
                    cond0 = X_raw['Frame'].str.contains(mode_lst[mode_nb])
                    X_raw = X_raw[cond0]
                    nodes_x = X_raw['X'].to_numpy()
                    nodes_y = X_raw['Y'].to_numpy()
                    nodes_point =np.stack((nodes_x, nodes_y)).transpose()
                    grid = pred[index].clone().cpu().detach().numpy().squeeze()

                    x_grid = np.linspace(0, 400, 24)
                    y_grid = np.linspace(0, 400, 24)

                    # Créer l'interpolateur
                    interp = RegularGridInterpolator((x_grid, y_grid), grid)

                    new_node_values = interp(nodes_point)

                    X_raw['U-U3'] = new_node_values
                    # récuperer la fréquence du mode dans la colonne Frame
                    eigen_f_Hz = eigen_f_Hz_lst[mode_nb]

                    mass_csr, stif_csr, X_csr = build_matrix(X_raw, mass_raw, stif_raw)

                    # Compute the eigen mode equation
                    omega = eigen_f_Hz*2*np.pi
                    fitness = (mass_csr*(omega**2) - stif_csr)@X_csr
                    norms_fitness.append(norm(fitness))

                loss = np.mean(norms_fitness)
                loss = torch.tensor(loss, requires_grad=True)


            else :
                raise ValueError("invalide loss name")

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
    plt.ylabel("loss MSE (mm2)")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()


def get_FEM_inf_loss(cfg):
    
    # build_matrix()
    
    None




def train(config_file, prepared_file = None):


    #load config/data
    cfg = load_config(f"config/{config_file}")
    if prepared_file :
        data_path = f"data/prepared/{prepared_file.split(".")[0]}/"
    else :
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
        cfg['model']['model_name'], X_size, cfg['model']['mlp_layers'], cfg['model']['decoder_channels']
    ).to(device)
    
    #Save pipeline parameters
    mlflow.log_params(cfg['model'])
    mlflow.log_params(cfg['training'])
    mlflow.log_params(cfg['dataset'])
    mlflow.log_params(cfg['prepare'])

    #Train
    epochs, train_losses, val_losses = train_model(model, cfg, config_file, device, X_train, y_train, X_val, y_val, 
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
