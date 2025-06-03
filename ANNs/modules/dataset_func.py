#Ajoute automatiquement le dossier parent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import modules.tools as tl
import os
import pandas as pd
import numpy as np
import json
from modules.Abaqus_Import_matrices_v1 import(
    Import_matrix_raw
)

import time

# get matrix data ordered folowing prepared data
def get_matrix_data(sim_grid_path, indexes):
    mtx_path = os.path.join(sim_grid_path, "mtx")

    f_mass = "matrix_clean_mod_MASS1.mtx"
    f_stif = "matrix_clean_mod_STIF1.mtx"

    matrix_data = []

    for i in indexes :
        i_mtx_path = os.path.join(mtx_path, str(i)+"\\")

        MASS1_path = os.path.join(i_mtx_path, f_mass)
        STIF1_path = os.path.join(i_mtx_path, f_stif)
        mat_X_path = os.path.join(sim_grid_path, "results", f"{i}.csv")

        mass_raw, stif_raw, X_raw, mode_lst, eigen_f_Hz_lst = Import_matrix_raw(MASS1_path, STIF1_path, mat_X_path)

        matrix_data.append(
            {
                "mass_raw" : mass_raw,
                "stif_raw" : stif_raw,
                "X_raw" : X_raw,
                "mode_lst" : mode_lst,
                "eigen_f_Hz_lst" : eigen_f_Hz_lst
            }
        )


    return matrix_data


def remap_u_from_sim_grid(sim_grid_path, resolution = None, dataset_format = "v1"):

    params_path = os.path.join(sim_grid_path,'params.xlsx')
    sim_count = len(os.listdir(os.path.join(sim_grid_path,"results")))
    params_df = pd.read_excel(params_path)
    U_remap = []
    invalid_results = []

    for i in range(sim_count) :
        results_path = os.path.join(sim_grid_path, f"results\\{i}.csv")
        pos_path = os.path.join(sim_grid_path, f"positions\\{i}.inp")
        if dataset_format == "v1" :
            results = tl.results_from_csv(results_path)
        elif dataset_format == "v2" :
            results = tl.results_from_csv_v2(results_path)
        
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
        print(f"{i+1}/{sim_count} done")
    
    print(f"Not correct results files : {invalid_results}")
    len_failed =len(invalid_results)
    print(f"{len_failed}/{sim_count} importations failed")
    
    return np.array(U_remap), invalid_results

def get_X_from_sim_grid(sim_grid_path, X_name_list = ["distances", "rayons", "plaque_epaisseurs", "materiaux"]) :
    params_path = os.path.join(sim_grid_path,'params.xlsx')
    params_df = pd.read_excel(params_path)

    materiaux_path = r"C:\Users\Antoine\Documents\master\PA\abacus\sim_plaque\run_sims\materiaux.json"
    with open(materiaux_path, "r", encoding="utf-8") as f:
        materiaux = json.load(f)

    X = []

    for i in X_name_list :
        if i=="materiaux":
            mat_collumn = list(params_df.loc[:, i])
            material_dict = {mat['name']: mat for mat in materiaux}
            young_modulus_list = [material_dict[nom]['young_modulus'] for nom in mat_collumn]
            density_list = [material_dict[nom]['density'] for nom in mat_collumn]
            poisson_modulus_list = [material_dict[nom]['poisson_modulus'] for nom in mat_collumn]
            X.append(young_modulus_list)
            X.append(density_list)
            X.append(poisson_modulus_list)


        else :
            mat_collumn = None
            X.append(params_df.loc[:, i])

    return np.array(X).transpose(), mat_collumn

def get_freqs_from_sim_grid(sim_grid_path, dataset_format, mode = "1"):
    sim_count = len(os.listdir(os.path.join(sim_grid_path,"results")))
    freqs = []
    for i in range(sim_count) :
        results_path = os.path.join(sim_grid_path, f"results\\{i}.csv")
        if dataset_format == "v1" :
            results = tl.results_from_csv(results_path)
        elif dataset_format == "v2" :
            results = tl.results_from_csv_v2(results_path)
        frequ_mode = results[mode][0]["freq"]
        freqs.append(frequ_mode)
        

    return np.array(freqs)


if __name__ == "__main__":
    s_t = time.time()
    matrix_data = get_matrix_data("data\\raw\\20250512105014",[0,1,2,3])
    mass_raw = matrix_data[0]["mass_raw"]
    mode_lst = matrix_data[0]["mode_lst"]
    eigen_f_Hz_lst = matrix_data[0]["eigen_f_Hz_lst"]

    print(mass_raw.head(5))
    mass_raw.to_csv("mon_dataframe.csv", index=False)
    mass_raw_imported = pd.read_csv("mon_dataframe.csv")

    print(mass_raw_imported.head(5))

    e_t = time.time()

    print(f"comp time :{e_t-s_t} s")

    print("")