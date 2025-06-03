import torch
import torch.nn as nn
import numpy as np

import sys
import os
#Ajoute automatiquement le dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.train import train_model
from src.models.mlp_models import get_model
from utilities import (
    load_config,
    load_prepared
)


def incremental_train(config_file, incr_count = 10, prepared_file = None):
    
    def get_mse(model, X_test, y_test):
        device = next(model.parameters()).device
        X_test = X_test.clone().to(device)
        y_test = y_test.clone().to(device)

        with torch.no_grad():
            y_pred = model(X_test)
            mse = torch.nn.functional.mse_loss(y_pred, y_test)

        return mse

    #load config/data
    cfg = load_config(f"config/{config_file}")
    if prepared_file :
        data_path = f"data/prepared/{prepared_file.split(".")[0]}/"
    else :
        data_path = f"data/prepared/{config_file.split(".")[0]}/"
    X_train, y_train, X_val, y_val = load_prepared(data_path, type="train")
    X_test, y_test = load_prepared(data_path, type="evaluate")

    #to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    X_size = X_train.shape[-1]

    sample_count = X_train.shape[0]
    incr = sample_count//incr_count

    train_sample_counts = []
    mse_list = []

    def reset_mse():
        return torch.tensor(1000)
    mse = reset_mse()

    for i in range(incr_count):
        if i == incr_count :
            sub_X_train = X_train
            sub_y_train = y_train
        
        else :
            train_sample_count = incr*(i+1)
            sub_X_train = X_train[:train_sample_count]
            sub_y_train = y_train[:train_sample_count]

        while mse.item() > 200 : #sometimes training is bugging
            model = get_model(
                cfg['model']['model_name'], X_size, cfg['model']['mlp_layers'], cfg['model']['decoder_channels']
            ).to(device)

                #Train
            epochs, train_losses, val_losses = train_model(model, cfg, config_file, device, sub_X_train, sub_y_train, X_val, y_val, 
                                                    batch_size=cfg['training']['batch_size'], 
                                                    num_epochs=cfg['training']['epochs'], 
                                                    patience=cfg['training']['patience'], 
                                                    lr=cfg['training']['learning_rate'])



            mse = get_mse(model, X_test, y_test)

        print(f"{train_sample_count} samples, MSE: {mse.item():.4f}")

        train_sample_counts.append(train_sample_count)
        mse_list.append(mse.item())
        mse = reset_mse()

    return train_sample_counts, mse_list


if __name__ == "__main__":

    config_file = "best_ds2.yaml"
    prepared_file = "exp_2deg.yaml"
    save_path = f"results/inrc_train/{config_file.split(".")[0]}"
    os.makedirs(save_path)
    train_sample_counts, mse_list= incremental_train(config_file, incr_count = 20, prepared_file=prepared_file)

    np.save(os.path.join(save_path,  'train_sample_count.npy'), np.array(train_sample_counts))
    np.save(os.path.join(save_path,  'mse_list.npy'), np.array(mse_list))