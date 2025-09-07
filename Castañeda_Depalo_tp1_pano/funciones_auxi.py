import numpy as np
import cv2
import matplotlib.pyplot as plt

#ANMS: Adaptive Non-Maximal Suppression
def anms(corners, num_corners=500):
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



#Función para imprimir los keypoints ANMS
def imprimir_anms(pts, img):
    #Imprimo los keypoints SIFT antes y despues de ANMS
    print(f"Keypoints antes de ANMS: {len(pts)}")
    keypoints_anms = anms(pts, num_corners=500)
    print(f"Keypoints despues de ANMS: {len(keypoints_anms)}")

    #Mostrar los puntos ANMS sobre fondo blanco
    plt.figure(figsize=(8, 6))
    plt.imshow(img)  # fondo blanco
    plt.scatter(keypoints_anms[:, 0], keypoints_anms[:, 1], s=10, c='blue')
    plt.title("Keypoints despues de ANMS (solo puntos)")
    plt.axis('off')
    plt.show()
    return keypoints_anms

#Función para hacer match entre puntos ANMS y keypoints originales
def match_pts_to_keypoints(pts_anms, keypoints_all):
    keypoints_red = []
    # recorro cada punto (x, y) que fue seleccionado por ANMS
    for pt_anms in pts_anms:
        # recorro todos los keypoints originales de la imagen
        for kp in keypoints_all:
            # si las coordenadas del KeyPoint coinciden con las del punto ANMS, lo consideramos equivalente
            if tuple(map(int, kp.pt)) == tuple(map(int, pt_anms)):
                keypoints_red.append(kp)
                break
    return keypoints_red

#Función para extraer descriptores de los keypoints filtrados
def extraer_descriptores(keypoints_all, keypoints_filtered, descriptors_all):
    # extraigo los descriptores correspondientes a los keypoints filtrados.
    indices = [keypoints_all.index(kp) for kp in keypoints_filtered]
    return descriptors_all[indices]
