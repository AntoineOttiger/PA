# **Dataset creation :**

open folder run_sims or run_sims_V2 in your IDE

run_sims : loss MSE dataset

run_sims_V2 : loss FEM-based dataset

lanch run.py to calculate a grid of simulation

# Train, evaluate, ect.. neural networks :

open folder ANNs in your IDE

to open mlflow viewer  in your browser (to visualize run's results)  : mlflow ui ( terminal command)

#### **scr/prepare.py :**

create prepared data based on a config file

#### scr/pipeline.py :

train and evaluate a model based on a config file. Generate a run ID usfull to reevalute the model from a pipeline later.

##### scr/train.py .

train a model with a config file

#### scr/evaluate.py :

Permit to evaluate a model from a run id. For exemple if you made a run with pipeline.py you can reevaluate the model from this run by providing run id. 

Activate/Desactivate calculation of shape heatmaps and/or feature/feature plots (is_pred_hm=True, is_plot_f_f=True)

##### to retrieve exact results of the report :

###### model non-optim on dataset 1 :

run id : 52e5eb0edc304c98be3dcf519aea4b8d

###### model non-optim on dataset 2 :

run id : d4ffe3fe0cab487c876ad96a53272c9d

###### model optim on dataset 1 :

run id : cc91b319ba7a4361ad337c056bdc7452

###### model optim on dataset 2 :

run id : 0e5ce74da09a47a48e745ada05756c4d


#### scr/hyperparam_opt.py :

lanch an hyperparameter optimization run.
