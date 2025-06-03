import pandas as pd
import numpy as np
import scipy.linalg as sclin 
from scipy.sparse import  csr_matrix, diags, eye
from scipy.sparse.linalg import norm, eigs, inv
import matplotlib.pyplot as plt
from matspy import spy_to_mpl
import scipy.sparse.linalg as spla



def Import_matrix_raw(MASS1_path, STIF1_path, mat_X_path): 
    # importation des matrices
    mass_raw = pd.read_csv(MASS1_path, header = None, names = ['Node_1', 'dof_1','Node_2', 'dof_2', 'mass_t'])
    stif_raw = pd.read_csv(STIF1_path, header = None, names = ['Node_1', 'dof_1','Node_2', 'dof_2', 'E'])
    X_raw = pd.read_csv(mat_X_path)
    
    # enlever les espaces vides dans les noms de colonnes 
    original_col = X_raw.columns  
    new_col = []
    for el in original_col:
        new_col.append(el.replace(' ',''))
    X_raw.columns = new_col
    
    # création des la listes des numéros de mode dans la matrice X
    mode_lst = []
    eigen_f_Hz_lst = []
    for el in X_raw['Frame'].unique():
        if 'Mode' in el:
            mode_lst.append(el.split(':')[0])
            eigen_f_Hz_lst.append(float(el.split('Freq =')[1].split('(')[0].replace(' ','')))
    return mass_raw, stif_raw, X_raw, mode_lst, eigen_f_Hz_lst




def build_matrix(X_raw, mass_raw, stif_raw):
    #prendre Node label si les valeurs U1 U2 U3 et UR1 UR2 UR3
                 
    cond1 = X_raw['U-U1'] == 0
    cond2 = X_raw['U-U2'] == 0
    cond3 = X_raw['U-U3'] == 0
    cond4 = X_raw['UR-UR1'] == 0
    cond5 = X_raw['UR-UR2'] == 0
    cond6 = X_raw['UR-UR3'] == 0
    
    # condition pour filtrer les noeuds qui ont 0 sur les 6 axes et
    #sont donc bloqués
    
    cond_tot = cond1*cond2*cond3*cond4*cond5*cond6  
    
    # liste des noeuds à enlever
    node_to_remove = X_raw[cond_tot]['NodeLabel']
    node_to_keep = X_raw[~cond_tot]['NodeLabel']
    
    cond_sel = ~cond_tot
    
    cond_mass = mass_raw['Node_1'] == 'a'
    for i in node_to_remove:
        cond_mass = cond_mass + (mass_raw['Node_1'] == i)
        cond_mass = cond_mass + (mass_raw['Node_2'] == i)
     
    
    
    cond_stif = stif_raw['Node_1'] == 'a'
    for i in node_to_remove:
        cond_stif = cond_stif + (stif_raw['Node_1'] == i)
        cond_stif = cond_stif + (stif_raw['Node_2'] == i)
    
        
    mass_raw['dim_1'] = mass_raw['Node_1'].astype(str) +'_' +  mass_raw['dof_1'].astype(str)
    mass_raw['dim_2'] = mass_raw['Node_2'].astype(str) +'_' +  mass_raw['dof_2'].astype(str)
    
    stif_raw['dim_1'] = stif_raw['Node_1'].astype(str) +'_' +  stif_raw['dof_1'].astype(str)
    stif_raw['dim_2'] = stif_raw['Node_2'].astype(str) +'_' +  stif_raw['dof_2'].astype(str)
    
    mass = mass_raw[~cond_mass]
    stif = stif_raw[~cond_stif]
    
    X = X_raw[cond_sel]
    
    X_df_1 = pd.DataFrame()
    X_df_1['indx'] = X['NodeLabel'].astype(str) +'_' + '1' 
    X_df_1['value'] = X['U-U1']
    
    X_df_2 = pd.DataFrame()
    X_df_2['indx'] = X['NodeLabel'].astype(str) +'_' + '2' 
    X_df_2['value'] = X['U-U2']
    
    X_df_3 = pd.DataFrame()
    X_df_3['indx'] = X['NodeLabel'].astype(str) +'_' + '3' 
    X_df_3['value'] = X['U-U3']
    
    X_df_4 = pd.DataFrame()
    X_df_4['indx'] = X['NodeLabel'].astype(str) +'_' + '4' 
    X_df_4['value'] = X['UR-UR1']
    
    X_df_5 = pd.DataFrame()
    X_df_5['indx'] = X['NodeLabel'].astype(str) +'_' + '5' 
    X_df_5['value'] = X['UR-UR2']
    
    X_df_6 = pd.DataFrame()
    X_df_6['indx'] = X['NodeLabel'].astype(str) +'_' + '6' 
    X_df_6['value'] = X['UR-UR3']
    
     
    dim_to_int = {}
    n = 0
    for i in node_to_keep:
        for j in [3,4,5]:
            dim_to_int[str(i) + '_' + str(j)] = n
            n=n+1
    for i in node_to_keep:
        for j in [1,2,6]:
            dim_to_int[str(i) + '_' + str(j)] = n
            n=n+1
    
    X_df = pd.concat([X_df_1, X_df_2, X_df_3, X_df_4, X_df_5, X_df_6], axis = 0)
    
    X_df['pos'] = [dim_to_int[el] for el in X_df['indx']]
    
    X_df = X_df.sort_values(by='pos', axis=0)
        
    # Define a simple 5x5 matrix
    data = mass['mass_t'].to_numpy()
    rows = np.array([dim_to_int[el] for el in mass['dim_1'].to_numpy()])
    cols = np.array([dim_to_int[el] for el in mass['dim_2'].to_numpy()])
    
    # sparse matrix
    mass_csr_half = csr_matrix((data, (rows, cols)))
    mass_csr = mass_csr_half + mass_csr_half.T - diags(mass_csr_half.diagonal())
    
    # np.max(mass_csr)
    # np.min(mass_csr)
    # plt.figure()
    # plt.hist(mass_csr.data, bins = 'fd')
    
    # mass_csr_inv  = pinv(mass_csr.todense())
    # mass_csr_inv = csr_matrix(mass_csr_inv)
    
     
    # decompose with lu method
    
    # lu = spla.splu(mass_csr)
    # mass_csr_inv = lu.solve(eye(mass_csr.shape[0]))
     
    # fig, ax =spy_to_mpl(mass_csr) 
    
    # pinv(mass_csr.todense())
    
    data = stif['E'].to_numpy()
    rows = np.array([dim_to_int[el] for el in stif['dim_1'].to_numpy()])
    cols = np.array([dim_to_int[el] for el in stif['dim_2'].to_numpy()])
    
    # sparse matrix
    stif_csr_half = csr_matrix((data, (rows, cols)))
    
    # ATTENTION DOUBLE LES ELEMENTS DE LA DIAGONALE
    stif_csr = stif_csr_half + stif_csr_half.T - diags(stif_csr_half.diagonal())
    
    # np.max(stif_csr)
    # np.min(stif_csr)
    # plt.figure()
    # plt.hist(stif_csr.data, bins = 'fd')
    
    data = X_df['value'].to_numpy()
    rows = np.array([dim_to_int[el] for el in X_df['indx'].to_numpy()])
    # cols = np.array([dim_to_int[el] for el in X_df['indx'].to_numpy()])
    cols = np.array([0 for el in X_df['indx'].to_numpy()])
    
    # sparse matrix
    X_csr = csr_matrix((data, (rows, cols)))
    
    return mass_csr, stif_csr, X_csr


if __name__ == "__main__":

    # Répertoire de travail de abaqus ou se trouve les fichiers
    direct = 'c:/temp/'
    # nom des fichiers des matrices 
    f_mass = "matrix_clean_mod_MASS1.mtx"
    f_stif = "matrix_clean_mod_STIF1.mtx"
    f_X = "mat_X.csv"

    mass_raw, stif_raw, X_raw, mode_lst, eigen_f_Hz_lst = Import_matrix_raw(direct, f_mass, f_stif, f_X)

    # Position dans la liste des modes
    mode_nb = 0

    # sélection du mode la colonne Frame

    cond0 = X_raw['Frame'].str.contains(mode_lst[mode_nb])

    X_raw = X_raw[cond0]

    # récuperer la fréquence du mode dans la colonne Frame

    eigen_f_Hz = eigen_f_Hz_lst[mode_nb]


    mass_csr, stif_csr, X_csr = build_matrix(X_raw, mass_raw, stif_raw)

    # Compute the eigen mode equation
    omega = eigen_f_Hz*2*np.pi
    fitness = (mass_csr*(omega**2) - stif_csr)@X_csr
    print(norm(fitness))

 

# out.toarray().T[0][int(out.shape[0]/2):]

# plt.figure()
# plt.plot(out.toarray().T[0])
    

# eigs(stif_csr)


# plt.figure()
# plt.plot(out.data)
# plt.hist(out.data,bins = 30)

# from scipy.optimize import minimize

# def func(omega_lst):
#     out_lst = [] 
#     for omega in omega_lst:
#         out = (mass_csr*(omega**2) - stif_csr)@X_csr
#         out_lst.append(norm(out))
#         # out_2_norm = np.sqrt(np.sum((out.toarray().T[0][int(out.shape[0]/2):])**2))
#         # out_lst.append(out_2_norm)

       
#     return np.array(out_lst)



# opt = minimize(func, x0=1E3, method = 'Nelder-Mead')

# omega_lst = np.arange(1,1E3,15)
# y = func(omega_lst)

# plt.figure()
# plt.plot(omega_lst,y)

# a = sclin.inv(stif_csr.todense())
# b = (csr_matrix(a)@mass_csr@X_csr)/X_csr

# plt.figure()
# plt.plot(np.array(b.T)[0])

# omega_eigen = np.sqrt(1/np.mean(np.array(b.T)[0][0:int(b.shape[0]/2)]))
# f_eigen = omega_eigen/2/np.pi
 
   