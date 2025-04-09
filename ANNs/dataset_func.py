import tools as tl
import os
import pandas as pd
import numpy as np
import json


def remap_u_from_sim_grid(sim_grid_path, resolution = None):

    params_path = os.path.join(sim_grid_path,'params.xlsx')
    sim_count = len(os.listdir(os.path.join(sim_grid_path,"results")))
    params_df = pd.read_excel(params_path)
    U_remap = []
    invalid_results = []

    for i in range(sim_count) :
        results_path = os.path.join(sim_grid_path, f"results\\{i}.csv")
        pos_path = os.path.join(sim_grid_path, f"positions\\{i}.inp")
        results = tl.results_from_csv(results_path)
        
        pos = tl.pos_from_inp(pos_path)
        pos = pos[:,:2] # on garde que xy
        try :
            U = results["1"][1] #U mode 1, np.array, shape(nbr_pts, xyz)
        except :
            invalid_results.append(i)
            continue
        
        U = U[:,2] # on garde que z

        if i == 0 :
            #remap params based on the 1st mesh
            elem_size = int(params_df.loc[0, 'elem_size']) # 0=index_sim
            mesh_size_x = int(params_df.loc[0, 'largeurs'])
            mesh_size_y = int(params_df.loc[0, 'hauteurs'])
            if resolution == None :
                resolution = int(np.sqrt(np.shape(U)[0]))

        U_remap.append(tl.remap_U(mesh_size_x, mesh_size_y, resolution, U, pos, method="linear"))
        print(f"{i}/{sim_count} done")
    
    print(f"Not correct results files : {invalid_results}")
    len_failed =len(invalid_results)
    print(f"{len_failed}/{sim_count} importations failed")
    
    return np.array(U_remap), invalid_results

def get_X_from_sim_grid(sim_grid_path, X_name_list = ["distances", "rayons", "plaque_epaisseurs", "materiaux"]) :
    params_path = os.path.join(sim_grid_path,'params.xlsx')
    params_df = pd.read_excel(params_path)

    materiaux_path = r"C:\Users\Antoine\Documents\master\PA\abacus\sim_plaque\run_sims\materiaux_back_up.json"
    with open(materiaux_path, "r", encoding="utf-8") as f:
        materiaux = json.load(f)

    X = []

    for i in X_name_list :
        if i=="materiaux":
            mat_collumn = list(params_df.loc[:, i])
            materiaux_dict = {mat['name']: mat for mat in materiaux}
            young_modulus_list = [materiaux_dict[nom]['young_modulus'] for nom in mat_collumn]
            density_list = [materiaux_dict[nom]['density'] for nom in mat_collumn]
            poisson_modulus_list = [materiaux_dict[nom]['poisson_modulus'] for nom in mat_collumn]
            X.append(young_modulus_list)
            X.append(density_list)
            X.append(poisson_modulus_list)


        else :
            X.append(params_df.loc[:, i])

    return np.array(X).transpose()

def get_freqs_from_sim_grid(sim_grid_path, mode = "1"):
    sim_count = len(os.listdir(os.path.join(sim_grid_path,"results")))
    freqs = []
    for i in range(sim_count) :
        results_path = os.path.join(sim_grid_path, f"results\\{i}.csv")
        results = tl.results_from_csv(results_path)
        frequ_mode = results[mode][0]["freq"]
        freqs.append(frequ_mode)
        

    return np.array(freqs)


if __name__ == "__main__":

    sim_grid_path = r"C:\Users\Antoine\Documents\master\PA\abacus\sim_plaque\ANNs\saved_grid_results\20250331140859"

    X = get_X_from_sim_grid(sim_grid_path, ["distances", "rayons", "plaque_epaisseurs","materiaux"])

    print("")