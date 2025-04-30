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

from utilities import (
    load_prepared
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


def save_features_heatmap(feature_1, feature_2, score, f_name_1, f_name_2, score_name):
    heatmap, xedges, yedges = np.histogram2d(
        x=feature_1, 
        y=feature_2, 
        bins=int(np.sqrt(feature_1.shape[0])), 
        weights=score
    )

    fig, ax = plt.subplots(figsize=(8, 6))  

    cax = ax.imshow(
        heatmap.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower',
        cmap='viridis',
        aspect='auto'  
    )

    fig.colorbar(cax, ax=ax, label=score_name)
    ax.set_xlabel(f_name_1)
    ax.set_ylabel(f_name_2)

    path = f"temp/fhm_{f_name_1}_{f_name_2}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

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


def evaluate(model_uri, prepared_name, is_plot_f_f =False):

    mlflow.set_tag("phase", "evaluate")
    mlflow.set_tag("script", "evaluate.py")

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

    y_pred = get_y_pred(model, X_test)
    
    mse = get_mse(y_pred, y_test)
    AE = get_AE_by_sample(y_pred, y_test)


    if is_plot_f_f :
        X_test_denormed = scaler_X.inverse_transform(X_test.clone().cpu().numpy())
        #en test, temporaire,To DO : rendre dynamique
        feature_1 = X_test_denormed[:,0]
        feature_2 = X_test_denormed[:,1]
        path = save_features_heatmap(feature_1, feature_2, AE, f_name_1="distance", f_name_2="epaisseur", score_name="absolute error")
        mlflow.log_artifact(path, artifact_path="test/feature_heatmaps")

    #map_count = 2
    map_count = X_test.shape[0]
    save_pred_heatmaps(model, X_test, y_test, map_count)

    for i in range(map_count) :
        mlflow.log_artifact(f"temp/hm_{i}.png", artifact_path="test/pred_heatmaps")

    mlflow.log_metric("mse", mse)

    print("Evaluate done")


if __name__ == "__main__":
    with mlflow.start_run(run_name="debug_eval") as pipeline_run:
        model_uri = "runs:/c198eb2b320c47c1b3c53d2efd93ced4/model"
        evaluate(model_uri, prepared_name="exp_test")
    
    None