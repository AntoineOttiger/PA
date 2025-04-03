
from tools import clean_dir
from tools import count_nodes
from itertools import product
import os
import pandas as pd
import time

import json
import shutil
from datetime import datetime

#def paths avant changement du current path
run_sims_path=os.getcwd() 
grid_name = datetime.now().strftime("%Y%m%d%H%M%S")
grid_path = os.path.join(run_sims_path,"outputs", grid_name)
results_dir = os.path.join(grid_path, "results")
positions_path = os.path.join(grid_path, "positions")
global_sim_py = os.path.join(run_sims_path, "sim.py")
sim_path=r"C:\temp\sim.py"
temp_path=r"C:\temp"

shutil.copy(global_sim_py,sim_path)

#maintenant changement current path vers dossier temp
os.chdir(temp_path)

#sim params
params = {
    "largeurs" : [400], #mm
    "hauteurs" : [400], #mm
    "distances" : [25.0], #mm
    "rayons" : [4.5,3],  #mm
    "plaque_epaisseurs" :  [1.0], #mm
    "frequ_max_mode" : [500.0],
    #mesh params
    "elem_size" : [20], 
    "deviationFactor": [0.1],  # max 1
    "minSizeFactor" : [0.1] # max 1
}
params_list = [i for i in params.values()]



#Create folders
os.makedirs(results_dir)
os.makedirs(positions_path)


combinations = product(*params_list)

lines = []

for index, combo in enumerate(combinations):
    current_path = os.getcwd()
    results_path = os.path.join(results_dir,f'{index}.csv')
    
    json_path = os.path.join(current_path, "params.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([results_path, combo], f)

    command = f"abaqus cae noGUI={sim_path}"

    print(f"Launching sim{index}")
    start_time = time.time()
    retcode = os.system(command)
    end_time = time.time()
    

    if os.path.isfile(results_path):
        print(f"Sim {index} done")
        failed = False
        calculation_time = end_time- start_time
        print(f"Sim computation time : {calculation_time:.2f} s")
        inp_path = os.path.join(positions_path, f"{index}.inp")
        shutil.move(os.path.join(temp_path, "vibration.inp"), inp_path )

        try :
            start_time = time.time()
            nodes_count=count_nodes(inp_path)
            end_time = time.time()
            print(f"counting nodes spent {end_time-start_time} s")
        
        except :
            nodes_count="Failed"
            print("counting nodes failed")
        

    else :
        print(f"Sim {index} failed")
        failed = True
        calculation_time = None
        nodes_count = None

    lines.append(list(combo) + [failed, calculation_time, nodes_count])
    clean_dir(current_path)
        
df = pd.DataFrame(lines, columns=[i for i in params.keys()]+["is_failed", "calculation_time (s)", "node count"])     
df.to_excel(os.path.join(grid_path,"params.xlsx"))





print("Script done")


#Commande to launch one simulation : abaqus cae noGUI=sim.py