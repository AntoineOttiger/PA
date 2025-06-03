#Ajoute automatiquement le dossier parent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from modules.dataset_func import (
    remap_u_from_sim_grid,
    get_X_from_sim_grid,
    get_matrix_data
)
from utilities import load_config


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.model_selection import train_test_split
import joblib
import json

def load_data(config):

    data_path=f"data/raw/{config['dataset']['dataset_name']}"
    features=config['prepare']['features']
    dataset_format = config['dataset']['dataset_format']

    X, material_list = get_X_from_sim_grid(data_path, features)
    y, invalid_results = remap_u_from_sim_grid(data_path, resolution=24, dataset_format=dataset_format)
    X = np.delete(X, invalid_results, axis=0)
    y = np.expand_dims(y, axis=1)


    return X, y


def main(yaml_file) :

    cfg = load_config(f"config/{yaml_file}")

    
    X, y = load_data(config = cfg)
    

    # Normalize data
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = np.abs(y)
    
    #to tenseur
    X_tenseur = torch.from_numpy(X_scaled.astype(np.float32))
    y_tenseur = torch.from_numpy(y_scaled.astype(np.float32))


    # Indices d'origine
    indices = np.arange(len(X))

    # Split les indices au lieu des données
    idx_train, idx_temp = train_test_split(indices, test_size=cfg['prepare']['train_test_split'], random_state=cfg['prepare']['seed'], shuffle=True)

    # Puis, récupère les données à partir des indices
    X_train, X_temp = X_tenseur[idx_train], X_tenseur[idx_temp]
    y_train, y_temp = y_tenseur[idx_train], y_tenseur[idx_temp]

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

    np.save(save_path + "idx_train.npy", idx_train)
    np.save(save_path + "idx_temp.npy", idx_temp)

    if cfg["dataset"]["dataset_format"] == "v2":
        data_path=f"data/raw/{cfg['dataset']['dataset_name']}"
        #ordered folowing prepared data order
        matrix_data = get_matrix_data(data_path, indexes=idx_train)
        for index, i in enumerate(matrix_data) :
            mtx_save_path = os.path.join(save_path, "train_mtx", str(index))
            os.makedirs(mtx_save_path)
        
            i["mass_raw"].to_csv(os.path.join(mtx_save_path, "mass_raw.csv"), index=False)
            i["stif_raw"].to_csv(os.path.join(mtx_save_path, "stif_raw.csv"), index=False)
            i["X_raw"].to_csv(os.path.join(mtx_save_path, "X_raw.csv"), index=False)

            with open(os.path.join(mtx_save_path, "mode_lst.json"), "w") as f:
                json.dump(i["mode_lst"], f)

            with open(os.path.join(mtx_save_path, "eigen_f_Hz_lst.json"), "w") as f:
                json.dump(i["eigen_f_Hz_lst"], f)



if __name__ == "__main__":
    #stocke les data preparées sous forme de np array ()
    main("test_2deg_ds2.yaml")
