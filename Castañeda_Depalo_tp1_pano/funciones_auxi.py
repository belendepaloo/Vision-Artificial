import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

#ANMS: Adaptive Non-Maximal Suppression
def anms(keypoints, num_corners=500):
    """
    Implementación del ANMS real según el algoritmo del paper.
    keypoints: lista de tuplas (x, y, response)
    """
    if len(keypoints) <= num_corners:
        return np.array([[x, y] for (x, y, _) in keypoints])

    # Inicializar radios a infinito
    radii = np.full(len(keypoints), np.inf)

    for i, (x_i, y_i, r_i) in enumerate(keypoints):
        for j, (x_j, y_j, r_j) in enumerate(keypoints):
            if r_j > r_i:
                SD = (x_j - x_i) ** 2 + (y_j - y_i) ** 2
                if SD < radii[i]:
                    radii[i] = SD

    # Obtener índices con mayor radio
    selected_indices = np.argsort(radii)[-num_corners:]

    # Devolver coordenadas seleccionadas
    return np.array([[keypoints[i][0], keypoints[i][1]] for i in selected_indices])

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

def dlt(ori, dst):

    # Construct matrix A and vector b
    A = []
    b = []
    for i in range(4):
        x, y = ori[i]
        x_prima, y_prima = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prima, y * x_prima])
        A.append([0, 0, 0, -x, -y, -1, x * y_prima, y * y_prima])
        b.append(x_prima)
        b.append(y_prima)

    A = np.array(A)
    b = np.array(b)

    # resolvemos el sistema de ecuaciones A * h = b
    # el sistema es de 8x8, por lo que podemos resolverlo si A es inversible

    # resuelve el sistema de ecuaciones para encontrar los parámetros de H
    H = -np.linalg.solve(A, b)

    # agrega el elemento h_33
    H = np.hstack([H, [1]])

    # reorganiza H para formar la matrix en 3x3 to form the 3x3 homography matrix
    H = H.reshape(3, 3)

    return H

#Función para hacer match entre puntos ANMS y keypoints originales
import numpy as np

def match_pts_to_keypoints(pts_anms, keypoints_all, eps=1.5):
    """
    Empareja cada (x,y) de ANMS con el KeyPoint más cercano en 'keypoints_all'.
    Devuelve:
      keypoints_red   -> lista de cv2.KeyPoint filtrados
      indices         -> np.array de índices correspondientes en keypoints_all
    """
    # coords de todos los KP originales
    kp_xy = np.array([kp.pt for kp in keypoints_all], dtype=np.float32)  # (N,2)

    indices = []
    used = set()

    for p in pts_anms:
        # distancia a todos los KP
        dists = np.linalg.norm(kp_xy - p, axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= eps and j not in used:
            indices.append(j)
            used.add(j)
        # Si dists[j] > eps, ignoramos ese punto ANMS (no hay KP suficientemente cercano)

    indices = np.array(indices, dtype=int)
    keypoints_red = [keypoints_all[j] for j in indices]
    return keypoints_red, indices


def extraer_descriptores(descriptors_all, indices):
    """
    Filtra los descriptores originales por índices (coherentes con keypoints_all).
    """
    return descriptors_all[indices]


def ransac_homography(matches, kp1, kp2, threshold=5.0, max_iterations=1000):
    """
    Implement RANSAC algorithm to find homography between two images
    
    Parameters:
    - matches: List of matches between two images
    - threshold: Distance threshold to consider a point as inlier
    - max_iterations: Maximum number of iterations for RANSAC
    
    Returns:
    - best_H: Best homography matrix
    - best_inliers: Indices of inlier matches
    """
    # Extract matched points from keypoints
    if isinstance(matches[0], cv2.DMatch):
        # Case when matches is a list of DMatch objects
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    else:
        # Case when matches is a tuple of (src_pts, dst_pts)
        src_pts, dst_pts = matches
    
    num_matches = len(src_pts)
    if num_matches < 4:
        print("Not enough matches to estimate homography")
        return None, []
    
    best_inliers = []
    best_H = None
    
    for _ in range(max_iterations):
        # Randomly select 4 point pairs
        random_indices = random.sample(range(num_matches), 4)
        
        src_sample = src_pts[random_indices]
        dst_sample = dst_pts[random_indices]
        
        # Calculate homography using DLT
        try:
            H = dlt(src_sample, dst_sample)
            
            # Transform all source points
            src_pts_homogeneous = np.hstack((src_pts, np.ones((num_matches, 1))))
            transformed_pts = np.dot(H, src_pts_homogeneous.T).T
            
            # Normalize homogeneous coordinates
            transformed_pts[:, 0] /= transformed_pts[:, 2]
            transformed_pts[:, 1] /= transformed_pts[:, 2]
            
            # Calculate distance between transformed points and destination points
            distances = np.sqrt(np.sum((transformed_pts[:, :2] - dst_pts)**2, axis=1))
            
            # Find inliers
            inliers = np.where(distances < threshold)[0]
            
            # Update best model if we found more inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_H = H
        except np.linalg.LinAlgError:
            # Skip if matrix is singular
            continue
    
    # Refine homography using all inliers if we found any
    if len(best_inliers) >= 4:
        src_inliers = src_pts[best_inliers]
        dst_inliers = dst_pts[best_inliers]
        
        # Recalculate homography with all inliers
        try:
            refined_H = dlt(src_inliers, dst_inliers)
            best_H = refined_H
        except:
            pass
        
    print(f"RANSAC found {len(best_inliers)} inliers out of {num_matches} matches")
    return best_H, best_inliers

def calculate_panorama_bounds(img1, img2, img3, H_1to2, H_1to3):
    """
    Calculate the bounds of the panoramic image based on the transformations
    
    Parameters:
    - img1, img2, img3: The three input images (anchor, left, right)
    - H_1to2, H_1to3: Homography matrices from anchor to left and right images
    
    Returns:
    - offset_x, offset_y: Offset to apply to keep all pixels in frame
    - panorama_width, panorama_height: Size of the final panorama
    """
    # Get dimensions of the images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h3, w3 = img3.shape[:2]
    
    # Define corners of each image
    corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners3 = np.array([[0, 0], [0, h3], [w3, h3], [w3, 0]], dtype=np.float32).reshape(-1, 1, 2)
    
    # Transform corners using homographies (from anchor to other images)
    transformed_corners2 = cv2.perspectiveTransform(corners2, np.linalg.inv(H_1to2))
    transformed_corners3 = cv2.perspectiveTransform(corners3, np.linalg.inv(H_1to3))
    
    # Combine all corners to find bounds
    all_corners = np.concatenate([corners1, transformed_corners2, transformed_corners3], axis=0)
    
    # Find min and max x, y coordinates
    x_min = np.min(all_corners[:, 0, 0])
    y_min = np.min(all_corners[:, 0, 1])
    x_max = np.max(all_corners[:, 0, 0])
    y_max = np.max(all_corners[:, 0, 1])
    
    # Calculate offset to keep all pixels in frame
    offset_x = -x_min if x_min < 0 else 0
    offset_y = -y_min if y_min < 0 else 0
    
    # Calculate panorama dimensions
    panorama_width = int(np.ceil(x_max + offset_x))
    panorama_height = int(np.ceil(y_max + offset_y))
    
    return offset_x, offset_y, panorama_width, panorama_height

def create_translation_matrix(offset_x, offset_y):
    """Create a translation matrix for the given offsets"""
    return np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)

def create_distance_mask(img):
    """Create a distance mask for blending"""
    h, w = img.shape[:2]
    
    # Create a mask that is 1 where image is non-zero
    mask = np.zeros((h, w), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask[gray > 0] = 1
    
    # Use distance transform to create a weight mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Normalize to 0-1 range
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    
    # Convert to 3 channels
    dist = dist[:, :, np.newaxis]
    return np.repeat(dist, 3, axis=2)

# Función para hacer matching con Lowe ratio, usando kNN, devuelve buenos matches y todos los matches
def lowe_ratio_knn(descA, descB, ratio=0.75, norm=None):
    """
    kNN + Lowe ratio. Devuelve (good_matches, raw_knn)
    - norm: cv2.NORM_L2 (SIFT) o cv2.NORM_HAMMING (ORB/BRIEF). Si None, se infiere del dtype.
    """
    if norm is None:
        norm = cv2.NORM_L2 if descA.dtype != np.uint8 else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(descA, descB, k=2)

    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good, raw

# Función para hacer matching mutuo, devuelve sólo matches consistentes
def mutual_consistency(matchesAB, matchesBA):
    """Conserva sólo A->B que estén confirmados por B->A."""
    back = {(m.queryIdx, m.trainIdx) for m in matchesBA}  # (idxB, idxA) en BA
    return [m for m in matchesAB if (m.trainIdx, m.queryIdx) in back]

# Función para dibujar los mejores N matches, ordenados por distancia
def draw_top_matches(imgA, kpsA, imgB, kpsB, matches, title="", N=50):
    """Dibuja top-N matches por distancia."""
    matches = sorted(matches, key=lambda m: m.distance)[:N]
    vis = cv2.drawMatches(imgA, kpsA, imgB, kpsB, matches, None, flags=2)
    plt.figure(figsize=(16,6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Mejores {len(matches)} matches {title}')
    plt.axis('off')
    plt.show()






# AYUDAAA ARE WE GOING INSANEEEEE???!!!!


def _forward_err(H, src, dst):
    N = len(src)
    sh = np.hstack([src, np.ones((N,1), dtype=np.float64)])
    pr = (H @ sh.T).T
    pr = pr[:, :2] / pr[:, 2:3]
    return np.linalg.norm(pr - dst, axis=1)

def _symmetric_err(H, src, dst):
    e1 = _forward_err(H, src, dst)
    Hi = np.linalg.inv(H)
    e2 = _forward_err(Hi, dst, src)
    return 0.5*(e1 + e2)

def _non_degenerate(pts, eps=1e-3):
    # área del cuadrilátero formado por 4 puntos
    p = pts.astype(np.float64)
    a = 0.5*np.abs(
        p[0,0]*p[1,1] + p[1,0]*p[2,1] + p[2,0]*p[3,1] + p[3,0]*p[0,1]
        - p[1,0]*p[0,1] - p[2,0]*p[1,1] - p[3,0]*p[2,1] - p[0,0]*p[3,1]
    )
    return a > eps

def ransac_homography_new(matches, kp1, kp2, threshold=5.0, max_iterations=2000, seed=0):
    random.seed(seed)

    # 1) Armar arrays de puntos
    if isinstance(matches[0], cv2.DMatch):
        src_pts = np.float64([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float64([kp2[m.trainIdx].pt for m in matches])
    else:
        src_pts, dst_pts = (np.float64(matches[0]), np.float64(matches[1]))

    N = len(src_pts)
    if N < 4:
        print("Not enough matches to estimate homography")
        return None, np.array([], dtype=int)

    best_inliers = np.array([], dtype=int)
    H_best = None

    # 2) RANSAC
    for _ in range(max_iterations):
        idx = random.sample(range(N), 4)
        if not _non_degenerate(src_pts[idx]):  # evitar degeneradas
            continue
        try:
            H = dlt(src_pts[idx], dst_pts[idx])  # tu DLT (4 puntos)
            if H is None or not np.isfinite(H).all():
                continue
        except np.linalg.LinAlgError:
            continue

        e = _symmetric_err(H, src_pts, dst_pts)
        inliers = np.where(e < threshold)[0]

        if inliers.size > best_inliers.size:
            best_inliers = inliers
            H_best = H
            # early stop si el consenso ya es muy alto
            if inliers.size > 0.85 * N:
                break

    if best_inliers.size < 4:
        print("RANSAC failed to find a consensus set ≥ 4.")
        return None, np.array([], dtype=int)

    # 3) Homografía FINAL con TODOS los inliers (mínimos cuadrados, sin RANSAC)
    H_final, _ = cv2.findHomography(src_pts[best_inliers], dst_pts[best_inliers], method=0)

    print(f"RANSAC found {best_inliers.size} inliers out of {N} matches")
    return H_final, best_inliers

