#Ajoute automatiquement le dossier parent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from src.train import train
from src.evaluate import evaluate

def main(config_file, pipeline_name):
    
    with mlflow.start_run(run_name=pipeline_name) as pipeline_run:
        mlflow.set_tag("type", "pipeline")
        mlflow.set_tag("pipeline_name", pipeline_name)
        
        # Child run: Training
        with mlflow.start_run(run_name="Train", nested=True):
            mlflow.set_tag("phase", "train")
            mlflow.set_tag("pipeline_name", pipeline_name)
            model_uri = train(config_file)

        # Child run: Evaluation
        with mlflow.start_run(run_name="Evaluate", nested=True):
            mlflow.set_tag("phase", "evaluate")
            mlflow.set_tag("pipeline_name", pipeline_name)
            prepared_name = config_file.split(".")[0]
            evaluate(model_uri, prepared_name)
    print("Pipeline done")

if __name__ == "__main__":

    pipeline_name = "pipeline_2deg" 
    main("exp_2deg.yaml", pipeline_name)
