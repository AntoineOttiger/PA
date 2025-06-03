import numpy as np
import cv2


def gen_plate_img(distance_trous, rayons= 4, height=400, width=400, display=False):


    # Créer une image noire (3 canaux pour RGB)
    image = np.zeros((height, width, 3), dtype=np.uint8) +100

    # Dessiner des cercles blancs
    # (image, centre, rayon, couleur, épaisseur)
    
    cv2.circle(image, (distance_trous, distance_trous), rayons, (255, 255, 255), -1)  # Cercle plein
    cv2.circle(image, (width-distance_trous, distance_trous), rayons, (255, 255, 255), -1)
    cv2.circle(image, (distance_trous, height-distance_trous), rayons, (255, 255, 255), -1)
    cv2.circle(image, (width-distance_trous, height-distance_trous), rayons, (255, 255, 255), -1)
    if display :
        # Afficher l'image
        cv2.imshow("Cercles blancs", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


if __name__ == "__main__" :
    image = gen_plate_img(100, display=True)
    None