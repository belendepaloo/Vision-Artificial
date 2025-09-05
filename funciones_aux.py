import numpy as np
import cv2
import matplotlib.pyplot as plt

#ANMS: Adaptive Non-Maximal Suppression
def anms(corners, num_corners=100):
    if len(corners) <= num_corners:
        return corners

    # Inicializar radios con un valor grande
    radii = np.full(len(corners), np.inf)

    for i, (x_i, y_i) in enumerate(corners):
        print(f"Procesando punto {i+1}/{len(corners)}", end='\r')
        for j, (x_j, y_j) in enumerate(corners):
            if i != j:
                dist = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
                if dist < radii[i]:
                    radii[i] = dist

    # Obtener los índices de los puntos con los mayores radios
    selected_indices = np.argsort(radii)[-num_corners:]

    return corners[selected_indices]