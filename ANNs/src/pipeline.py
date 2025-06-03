#Ajoute automatiquement le dossier parent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from src.train import train
from src.evaluate import evaluate
import time

def main(config_file, pipeline_name):
    
    with mlflow.start_run(run_name=pipeline_name) as pipeline_run:
        mlflow.set_tag("type", "pipeline")
        mlflow.set_tag("pipeline_name", pipeline_name)
        # Train
        s_t= time.time()
        model_uri = train(config_file)
        e_t = time.time()
        tot_time_train = e_t- s_t
        # Eval
        prepared_name = config_file.split(".")[0]
        s_t= time.time()
        evaluate(model_uri, prepared_name, is_pred_hm=True)
        e_t = time.time()
        tot_time_eval = e_t- s_t
        mlflow.log_dict({"training_time_seconds": tot_time_train, "eval_time_seconds": tot_time_eval}, "pipeline_details.json")

    print("Pipeline done")

if __name__ == "__main__":

    config_file = "exp_2deg.yaml"
    pipeline_name = config_file.split(".")[0]
    main(config_file, pipeline_name)
