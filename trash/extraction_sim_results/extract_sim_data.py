import pandas as pd
import numpy as np
import time


def data_from_csv(csv_path) :
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
        infos_dict["mode"] = mode
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


def data_from_inp(inp_path) :
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


csv_path = "abaqus.csv"
inp_path = "vibration.inp"

start_time = time.time()

#csv_data = data_from_csv(csv_path)

inp_data = data_from_inp(inp_path)

end_time = time.time()

print(f'Elapsed time : {end_time-start_time:.2f} s')
print("Done")
print("")