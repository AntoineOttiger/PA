#Ajoute automatiquement le dossier parent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from modules.dataset_func import (
    remap_u_from_sim_grid,
    get_X_from_sim_grid
)
from utilities import load_config

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.model_selection import train_test_split
import joblib

def load_data(data_path, features=["distances", "plaque_epaisseurs","materiaux"]):
    X, material_list = get_X_from_sim_grid(data_path, features)
    y, invalid_results = remap_u_from_sim_grid(data_path, resolution=24)
    X = np.delete(X, invalid_results, axis=0)
    y = np.expand_dims(y, axis=1)

    return X, y


def main(yaml_file) :

    cfg = load_config(f"config/{yaml_file}")
    X, y = load_data(f"data/raw/{cfg['dataset']['dataset_name']}", cfg['prepare']['features'])

    # Normalize data
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = np.abs(y)
    
    #to tenseur
    X_tenseur = torch.from_numpy(X_scaled.astype(np.float32))
    y_tenseur = torch.from_numpy(y_scaled.astype(np.float32))

    # train 0.8, test 0.1, val 0.1
    X_train, X_temp, y_train, y_temp = train_test_split(X_tenseur, y_tenseur, test_size=cfg['prepare']['train_test_split'], random_state=cfg['prepare']['seed'], shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=cfg['prepare']['val_test_split'], shuffle=False)


    save_path = f"data/prepared/{yaml_file.split(".")[0]}/"

    os.makedirs(save_path)

    joblib.dump(scaler_X, save_path + 'scaler_X.save')

    np.save(save_path + 'X_train.npy', X_train)
    np.save(save_path + 'y_train.npy', y_train)

    np.save(save_path + 'X_val.npy', X_val)
    np.save(save_path + 'y_val.npy', y_val)

    np.save(save_path + 'X_test.npy', X_test)
    np.save(save_path + 'y_test.npy', y_test)




if __name__ == "__main__":
    #stocke les data preparées sous forme de np array ()
    main("exp_2deg.yaml")
