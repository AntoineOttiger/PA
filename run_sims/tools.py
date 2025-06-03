import os
import glob
import json

def clean_dir(dir_path) :

    # Récupérer tous les fichiers du dossier
    fichiers = glob.glob(os.path.join(dir_path, "*"))  # Liste tous les fichiers et dossiers

    # Supprimer uniquement les fichiers qui ne sont pas en .py
    for fichier in fichiers:
        if os.path.isfile(fichier) and not fichier.endswith(".py") :  # Vérifie que c'est un fichier et pas .py
            os.remove(fichier)
            #print(f"🗑 Supprimé : {fichier}")

    #print("✅ Nettoyage terminé.")

    return None

def count_nodes(inp_path):


    # Lire les lignes du fichier
    with open(inp_path, "r") as f:
        lines = f.readlines()

    # Trouver les indices des sections *Node et *Element, type=S3
    start_index = None
    end_index = None

    for i, line in enumerate(lines):
        if "*Node" in line and start_index is None:
            start_index = i
        elif "*Element, type=S3" in line and end_index is None:
            end_index = i
            break  # on s'arrête à la première occurrence de *Element, type=S3

    # Calculer le nombre de lignes entre les deux sections
    if start_index is not None and end_index is not None and end_index > start_index:
        num_lines_between = end_index - start_index - 1  # -1 pour ne pas compter les lignes *Node et *Element
    else:
        num_lines_between = None

    return num_lines_between

class Materiau:
    def __init__(self, name : str, young_modulus : float, density : float, poisson_modulus):
        self.name = name
        self.young_modulus = young_modulus #Mpa
        self.density = density # Tonne/mm3 ?
        self.poisson_modulus = poisson_modulus
    
    def to_dict(self):
        return {
            "name" : self.name,
            "young_modulus" : self.young_modulus,
            "density" : self.density,
            "poisson_modulus" :self.poisson_modulus
        }



if __name__ == "__main__":
    materiaux = [
        Materiau("acier", 210000.0, 7.86e-09, 0.26).to_dict(), #source : script original
        Materiau("cuivre", 124000.0, 8.96e-09, 0.36).to_dict(), # source wikipédia
        Materiau("aluminium", 69000.0, 2.7e-09, 0.33).to_dict(), # source wikipédia
        Materiau("titan", 114000.0, 4.5e-09, 0.34).to_dict() #source simulationmateriaux.com
    ]

    current_path = os.getcwd()

    json_path = os.path.join(current_path, "materiaux.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(materiaux, f)

    