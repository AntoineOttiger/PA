#Ajoute automatiquement le dossier parent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import numpy as np
import torch
from pathlib import Path


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def save_config(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def load_prepared(path, type, to_torch = True):

    if type == "train":
        X_train = np.load(path + 'X_train.npy')
        y_train = np.load(path + 'y_train.npy')
        X_val = np.load(path + 'X_val.npy')
        y_val = np.load(path + 'y_val.npy')
        
        if to_torch :
            X_train = torch.from_numpy(X_train.astype(np.float32))
            y_train = torch.from_numpy(y_train.astype(np.float32))
            X_val = torch.from_numpy(X_val.astype(np.float32))
            y_val = torch.from_numpy(y_val.astype(np.float32))
    
        return X_train, y_train, X_val, y_val
    
    if type == "evaluate":
        X_test = np.load(path + 'X_test.npy')
        y_test = np.load(path + 'y_test.npy')

        if to_torch :
            X_test = torch.from_numpy(X_test.astype(np.float32))
            y_test = torch.from_numpy(y_test.astype(np.float32))

        return X_test, y_test

def clean_temp():

    folder_path = Path("temp")
    for item in folder_path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()


if __name__ == "__main__":
    None