import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import warnings



def results_from_csv(csv_path) :
    warnings.simplefilter(action='ignore', category=pd.errors.ParserWarning)
    df = pd.read_csv(csv_path,index_col=False, dtype=str)
    df_list = df.iloc[:,0].to_list()
    df_lists = '_BREAK_'.join(df_list).split("-----------------------------------------------------------------")
    
    info_mode = df_lists[::2] # Éléments aux indices pairs
    info_mode.pop(-1) #pas besoin du dernier
    info_mode.pop(0)
    data_mode =  df_lists[1::2]  # Éléments aux indices impairs
    data_mode.pop(0)

    data_per_mode = {}

    for info, data in zip(info_mode, data_mode):

        infos_dict = {}

        mode = int(info.split("Mode")[1].split(':')[0])
        #infos_dict["mode"] = mode
        infos_dict["value"] = float(info.split("Value =")[1].split("Freq =")[0])
        infos_dict["freq"] = float(info.split("Freq =")[1].split("(cycles/time)")[0])
        
        data = data.split("_BREAK_Field Output reported at nodes for part:")[0]
        data = data.split("_BREAK_")
        data.remove("")
        data_extracted = []
        for line in data :
            U = line.split()

            data_extracted .append([float(U[1]),float(U[2]),float(U[3])])

        data_extracted = np.array(data_extracted)
        
        data_per_mode[str(mode)] = [infos_dict, data_extracted]
    

    return data_per_mode


def pos_from_inp(inp_path) :
    data = []
    with open(inp_path, "r") as f:
        f_list = [i.strip() for i in f]
        ind_node = f_list.index("*Node")
        f_list = f_list[ind_node+1:]

        for ligne in f_list:
            ligne = ligne.strip()
            if ligne[0]=="*":
                break
            ligne=ligne.split(",", 1)[1].split(",")
            ligne=[float(i) for i in ligne]
            data.append(ligne)
    
    return np.array(data)


def remap_U(mesh_size_x : int, 
            mesh_size_y : int, 
            resolution: int, # min(mesh_size)/(elem_size*factor) ? 
            U : np.ndarray, 
            pos : np.ndarray, 
            method='linear') : # ou 'cubic', 'nearest'

    xi = np.linspace(0, mesh_size_x, resolution)
    yi = np.linspace(0, mesh_size_y, resolution)
    X, Y = np.meshgrid(xi, yi)

    remaped_U = griddata(pos, U, (X, Y), method=method)

    return remaped_U

def heatmaps_from_one_sample(true : np.ndarray, pred : np.ndarray):

    vmin = min(np.min(true), np.min(pred))
    vmax = max(np.max(true), np.max(pred))

    diff = abs(pred - true)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axs[0].imshow(true, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title("FEM results")
    axs[0].axis('off')

    im1 = axs[1].imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title("NN predictions")
    axs[1].axis('off')

    cbar1 = fig.colorbar(im1, ax=axs[1])
    cbar1.set_label("displacement")

    im2 = axs[2].imshow(diff, cmap='Oranges')
    axs[2].set_title("Absolute Error")
    axs[2].axis('off')
    cbar2 = fig.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()
    
    return None


if __name__ == "__main__":
    None



