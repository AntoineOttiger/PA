#Ajoute automatiquement le dossier parent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
import numpy as np
import time
import matplotlib.cm as cm
import itertools

from utilities import (
    clean_temp,
    load_prepared,
    load_config
)
from modules.tools import(
    heatmaps_from_one_sample,
    get_feature_ranges,
    get_scores_by_feature
)


def get_y_pred(model, X_test):
    model.eval()

    with torch.no_grad():  
        y_pred = model(X_test)

    return y_pred


def get_mse(y_pred, y_test):

    mse_loss = nn.MSELoss()
    mse = mse_loss(y_pred, y_test)

    return mse


def get_relative_error(y_pred, y_test, eps=0.1):
    y_pred = y_pred.clone().cpu()
    y_test = y_test.clone().cpu()

    # Éviter division par zéro
    mask = torch.abs(y_test) > eps
    if mask.sum() == 0:
        return torch.tensor(float('nan'))

    relative_error = torch.abs((y_pred[mask] - y_test[mask]) / y_test[mask])
    return relative_error.mean() * 100  


def get_mae(y_pred, y_test):

    mae_loss = nn.L1Loss()
    mae = mae_loss(y_pred, y_test)

    return mae

def save_features_heatmap(feature_1, feature_2, score, f_name_1, f_name_2, score_name):
    #bins = min(len(np.unique(feature_1)), len(np.unique(feature_2)))
    #bins = [len(np.unique(feature_1)), len(np.unique(feature_2))]
    if f_name_2 == "mat" :
        unique_values, encoded = np.unique(feature_2, return_inverse=True)
        feature_2 = encoded
        bins  = [len(np.unique(feature_1)), 4]
        string_labels = ["aluminium" , "titan", "copper", "steel"]

    else :
        bins = [len(np.unique(feature_1)), len(np.unique(feature_2))]
        string_labels = None

    heatmap, xedges, yedges = np.histogram2d(
        x=feature_1, 
        y=feature_2, 
        #bins=int(np.sqrt(feature_1.shape[0])), 
        bins = bins,
        weights=score
    )
    counts, _, _ = np.histogram2d(
        x=feature_1,
        y=feature_2,
        bins=bins
    )

    #No point = np.nan
    heatmap[counts == 0] = np.nan

    cmap = cm.viridis.copy()
    cmap.set_bad(color='white')  # couleur pour les NaN

    plt.imshow(
        heatmap.T,
        origin='lower',
        cmap=cmap,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect='auto'
    )
    cbar = plt.colorbar()
    cbar.set_label(score_name)
    plt.xlabel(f_name_1)
    
    ticks_x = (xedges[:-1] + xedges[1:]) / 2
    ticks_y = (yedges[:-1] + yedges[1:]) / 2

    labels_x = np.unique(feature_1)
    labels_x = [
        str(round(x, 1)) if x < 1 else str(int(round(x)))
        for x in labels_x
        ]
    
    if string_labels is not None:
        labels_y = string_labels
    
    else :
        labels_y = np.unique(feature_2)
        labels_y = [
            str(round(x, 1)) if x < 1 else str(int(round(x)))
            for x in labels_y
            ]
        plt.ylabel(f_name_2)

    if len(labels_x) > 12 :
            ticks_x = ticks_x[::2]
            labels_x = labels_x[::2]    

    if len(labels_y) > 12 :
            ticks_y = ticks_y[::2]
            labels_y = labels_y[::2]    


    plt.xticks(ticks_x, labels_x)
    plt.yticks(ticks_y, labels_y)

    path = f"temp/fhm_{f_name_1}_{f_name_2}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close() 
    

    return path


def get_AE_by_sample(y_pred, y_test):
    #create copies and convert to np array
    y_pred = y_pred.clone().cpu().numpy()
    y_test = y_test.clone().cpu().numpy()

    AE = np.abs(y_pred-y_test)
    AE = np.squeeze(AE, axis=1)
    AE = AE.reshape(AE.shape[0], -1)
    AE = np.mean(AE, axis=-1)

    return AE


def save_pred_heatmaps(model, X_test, y_test, map_count = 20):

    with torch.no_grad():  
        X_test_sample = X_test[:map_count]
        X_test_sample = X_test_sample
        pred_sample = model(X_test_sample)

    pred_sample = pred_sample.cpu().numpy()
    true_sample = y_test[:map_count].cpu().numpy()

    for i in range(map_count):
        heatmaps_from_one_sample(true_sample[i].squeeze(), pred_sample[i].squeeze(), one_scale=True, display=False)
        plt.savefig(f"temp/hm_{i}.png")
        plt.close()




def evaluate(model_uri, prepared_name, is_plot_f_f =False, is_pred_hm = False):


    # Charge le modèle MLflow
    model = mlflow.pytorch.load_model(model_uri)

    #load test_data
    data_path = f"data/prepared/{prepared_name}/"
    X_test, y_test = load_prepared(data_path, type="evaluate")
    scaler_X = joblib.load(data_path + 'scaler_X.save')

    #to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    s_t = time.time()
    y_pred = get_y_pred(model, X_test)
    e_t = time.time()
    tot_time = e_t - s_t
    mlflow.log_dict({ "pred_time_seconds": tot_time, "pred_time_per_sample_seconds": tot_time/X_test.shape[0]}, "eval_details.json")


    mse = get_mse(y_pred, y_test)
    mae = get_mae(y_pred, y_test)
    relative_error  = get_relative_error(y_pred, y_test)

    mlflow.log_metric("mse-mm2", mse)
    mlflow.log_metric("mae-mm", mae)
    mlflow.log_metric("relative-error-per100", relative_error)

    AE = get_AE_by_sample(y_pred, y_test)


    if is_plot_f_f :
        X_test_denormed = scaler_X.inverse_transform(X_test.clone().cpu().numpy())
        print(np.unique(X_test_denormed[:,2]))
        #en test, temporaire,To DO : rendre dynamique
        cfg = load_config(f"config/{prepared_name}.yaml")
        dataset = cfg["dataset"]["dataset_name"]

        ds1_metadata = {
            "distances" : {"ind":0, "name":"distance (mm)"},
            "plaque_epaisseurs" : {"ind":1, "name":"thickness (mm)"},
            "materiaux" : {"ind":3, "name":"mat"}

        }
        ds2_metadata = {
            "distances_x" : {"ind":0, "name":"distance_x (mm)"},
            "distances_y" : {"ind":1, "name":"distance_y (mm)"},
            "plaque_epaisseurs" : {"ind":2, "name":"thickness (mm)"},
            "materiaux" : {"ind":3, "name":"mat"}
        }

        if dataset == "20250331140859" : # dataset 1
            ds_metadata = ds1_metadata

        elif dataset == "20250428175523" : # dataset 2
            ds_metadata = ds2_metadata

        else :
            print("feature/feature analyse not implemented for this dataset")
            ds_metadata = None

        if ds_metadata :

            for k1, k2 in itertools.combinations(ds_metadata.keys(), 2):
                ind_1 = ds_metadata[k1]["ind"]
                ind_2 = ds_metadata[k2]["ind"]
                feature_1 = X_test_denormed[:,ind_1]
                feature_2 = X_test_denormed[:,ind_2]
                f_name_1 = ds_metadata[k1]["name"]
                f_name_2 = ds_metadata[k2]["name"]

                path = save_features_heatmap(feature_1, feature_2, AE, f_name_1=f_name_1, f_name_2=f_name_2, score_name="absolute error (mm)")
                mlflow.log_artifact(path, artifact_path="test/feature_heatmaps")


    if is_pred_hm :
        #map_count = 2
        map_count = X_test.shape[0]
        save_pred_heatmaps(model, X_test, y_test, map_count)

        for i in range(map_count) :
            mlflow.log_artifact(f"temp/hm_{i}.png", artifact_path="test/pred_heatmaps")

    clean_temp()

    print("Evaluate done")

    return mse


if __name__ == "__main__":
    with mlflow.start_run(run_name="eval") as pipeline_run:
        mlflow.set_tag("script", "evaluate.py")
        model_uri = "runs:/0e5ce74da09a47a48e745ada05756c4d/model"
        evaluate(model_uri, prepared_name="exp_2deg", is_pred_hm=False, is_plot_f_f=True)
    
    None