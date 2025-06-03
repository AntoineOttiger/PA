import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import mlflow
from utilities import (
    load_config,
    save_config
)
from src.train import train
from src.evaluate import evaluate
import optuna

#create a new config from a baseline changing hyperparameters
def create_new_config (bl_config_path, batch_size, lr, mlp_layers, decoder_channels) :
    new_config = load_config(bl_config_path)
    new_config['training']['batch_size'] = batch_size
    new_config['training']['learning_rate'] = lr
    new_config['model']['mlp_layers'] = mlp_layers
    new_config['model']['decoder_channels'] = decoder_channels
    save_config(new_config, "config/temp.yaml")


def objective(trial):

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    bl_config = "exp_test.yaml"

    #mlp layers
    mlp_depth = trial.suggest_int("mlp_depth", 2, 5)
    mlp_layers = []
    for i in range(mlp_depth - 1):
        out_features = trial.suggest_int(f"mlp_units_{i}", 16, 128, step=16)
        mlp_layers.append(out_features)
    final_out = 9
    mlp_layers.append(final_out)

    # Decodeur layers
    decoder_depth = 4
    decoder_channels = []
    for i in range(decoder_depth - 1):
        out_channels = trial.suggest_int(f"decoder_ch_{i}", 16, 128, step=16)
        decoder_channels.append(out_channels)
    decoder_channels.append(1)  # Last layer -> 1 channel

    create_new_config(f"config/{bl_config}", batch_size, lr, mlp_layers, decoder_channels)
    
    with mlflow.start_run(nested=True) as run:
        run_id = run.info.run_id 
        model_uri = train("temp.yaml", prepared_file=bl_config)

        prepared_name = bl_config.split(".")[0]
        mse = evaluate(model_uri, prepared_name, is_pred_hm=False)

        trial.set_user_attr("run_id", run_id)
    
    return mse

if __name__ == "__main__":
    with mlflow.start_run(run_name="optuna_optim_ds1") :
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)


        best_params = study.best_params
        best_mse = study.best_value
        best_run_id = study.best_trial.user_attrs["run_id"]

        mlflow.log_params(best_params)
        mlflow.log_metric("Best mse", best_mse)
        mlflow.set_tag("best_run_id", best_run_id)

        # Afficher les meilleurs résultats
        print("Best hyperparameters: ", best_params)
        print("Best mse : ", best_mse)
        print("Best run ID", best_run_id)
        
        None