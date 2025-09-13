import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# ----- Metodo de Harris y Shi-Tomasi (3.1) -----
def find_corners(img, method='harris'):
    useHarrisDetector = method=='harris'
    img = np.float32(img)
    corners = cv2.goodFeaturesToTrack(
      img,
      maxCorners=1000,
      qualityLevel=0.05,
      minDistance=11,
      useHarrisDetector=useHarrisDetector
    )

    corners = corners.reshape(-1, 2)
    return corners

def plot_corners(imgs, method='harris'):
    fig, axes = plt.subplots(1, len(imgs), figsize=(15, 5))
    if len(imgs) == 1:
        axes = [axes]

    for i, img in enumerate(imgs):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray, method=method)

        axes[i].imshow(gray, cmap='gray')
        axes[i].scatter(corners[:, 0], corners[:, 1], s=50, marker='+', color='red')
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# ----- Procesameinto de Keypoints y Descriptores - SIFT (3.1) -----
def process_keypoints(img, sift, show=True):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints_red = []
    for kp in keypoints:
        x, y = kp.pt
        if False:
            pass
        else:
            keypoints_red.append(kp)

    pts = np.array([kp.pt for kp in keypoints_red])

    if show:
        plt.figure(figsize=(8, 6))
        plt.imshow(np.ones(img.shape[:2]), cmap='gray')
        plt.scatter(pts[:, 0], pts[:, 1], s=0.1, c='red')
        plt.title("SIFT Keypoints (solo puntos)")
        plt.axis('off')
        plt.show()

    return keypoints, descriptors, keypoints_red, pts

# ------ ANMS (3.2) -----
def anms(keypoints, num_corners=500):
    if len(keypoints) <= num_corners:
        return np.array([[x, y] for (x, y, _) in keypoints])
    radii = np.full(len(keypoints), np.inf)

    for i, (x_i, y_i, r_i) in enumerate(keypoints):
        for j, (x_j, y_j, r_j) in enumerate(keypoints):
            if r_j > r_i:
                SD = (x_j - x_i) ** 2 + (y_j - y_i) ** 2
                if SD < radii[i]:
                    radii[i] = SD

    selected_indices = np.argsort(radii)[-num_corners:]
    return np.array([[keypoints[i][0], keypoints[i][1]] for i in selected_indices])

def imprimir_anms(pts, img):
    print(f"Keypoints antes de ANMS: {len(pts)}")
    keypoints_anms = anms(pts, num_corners = 500)
    print(f"Keypoints despues de ANMS: {len(keypoints_anms)}")

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.scatter(keypoints_anms[:, 0], keypoints_anms[:, 1], s=10, c='blue')
    plt.title("Keypoints despues de ANMS (solo puntos)")
    plt.axis('off')
    plt.show()
    return keypoints_anms

# ------ Matching entre keypoints y puntos ANMS -----
def match_pts_to_keypoints(pts_anms, keypoints_all, eps=1.5):
    kp_xy = np.array([kp.pt for kp in keypoints_all], dtype=np.float32)
    indices = []
    used = set()

    for p in pts_anms:
        dists = np.linalg.norm(kp_xy - p, axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= eps and j not in used:
            indices.append(j)
            used.add(j)

    indices = np.array(indices, dtype=int)
    keypoints_red = [keypoints_all[j] for j in indices]
    return keypoints_red, indices


# ----- Matching de descriptores (3.3) -----
def lowe_ratio_knn(descA, descB, ratio=0.75, norm=None):
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

def cross_check(matchesAB, matchesBA):
    back = {(m.queryIdx, m.trainIdx) for m in matchesBA}
    return [m for m in matchesAB if (m.trainIdx, m.queryIdx) in back]

# ----- Visualizacion de matches (3.3) -----
def draw_top_matches(imgA, kpsA, imgB, kpsB, matches, title="", N=50):
    matches = sorted(matches, key=lambda m: m.distance)[:N]
    
    h_A, w_A = imgA.shape[:2]
    h_B, w_B = imgB.shape[:2]
    
    max_height = 800
    scale_A = max_height / h_A
    scale_B = max_height / h_B
    
    resized_A = cv2.resize(imgA, (int(w_A * scale_A), max_height))
    resized_B = cv2.resize(imgB, (int(w_B * scale_B), max_height))
    
    scaled_kpsA = [cv2.KeyPoint(kp.pt[0] * scale_A, kp.pt[1] * scale_A, kp.size * scale_A, 
                               kp.angle, kp.response, kp.octave, kp.class_id) for kp in kpsA]
    scaled_kpsB = [cv2.KeyPoint(kp.pt[0] * scale_B, kp.pt[1] * scale_B, kp.size * scale_B, 
                               kp.angle, kp.response, kp.octave, kp.class_id) for kp in kpsB]
    
    vis = cv2.drawMatches(resized_A, scaled_kpsA, resized_B, scaled_kpsB, matches, None, flags=2)
    
    fig_height = 10
    total_width = resized_A.shape[1] + resized_B.shape[1]
    aspect_ratio = total_width / max_height
    fig_width = fig_height * aspect_ratio
    
    max_fig_width = 20
    if fig_width > max_fig_width:
        fig_width = max_fig_width
    
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Mejores {len(matches)} matches {title}')
    plt.axis('off')
    plt.show()

# ----- Homografias - DLT (3.4) -----
def dlt(ori, dst):
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
    H = -np.linalg.solve(A, b)

    H = np.hstack([H, [1]])
    H = H.reshape(3, 3)
    return H

# ----- RANSAC (3.5) -----
def create_translation_matrix(offset_x, offset_y):
    return np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)

def create_distance_mask(img):
    h, w = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray = np.where(gray > 0, 255, 0).astype(np.uint8)
    gray_inv = np.where(gray == 0, 255, 0).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(gray_inv)
    
    small_components_coords = []
    for label in range(1, num_labels):
        component_mask = (labels == label)
        component_size = np.sum(component_mask)
        
        if component_size < 200000:
            print(f"Componente {label} tiene {component_size} pixeles")
            coords = np.where(component_mask)
            small_components_coords.extend(list(zip(coords[1], coords[0])))
    print("cantidad de componentes chicas", len(small_components_coords))

    for x, y in small_components_coords:
        gray_final[y, x] = 1

    mask = (gray_final > 0).astype(np.uint8) 
    
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    
    dist = dist[:, :, np.newaxis]
    return np.repeat(dist, 3, axis=2)

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
    p = pts.astype(np.float64)
    a = 0.5*np.abs(
        p[0,0]*p[1,1] + p[1,0]*p[2,1] + p[2,0]*p[3,1] + p[3,0]*p[0,1]
        - p[1,0]*p[0,1] - p[2,0]*p[1,1] - p[3,0]*p[2,1] - p[0,0]*p[3,1]
    )
    return a > eps

def ransac_homography(matches, kp1, kp2, threshold=5.0, max_iterations=2000, seed=0):
    random.seed(seed)

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

    for _ in range(max_iterations):
        idx = random.sample(range(N), 4)
        if not _non_degenerate(src_pts[idx]):
            continue
        try:
            H = dlt(src_pts[idx], dst_pts[idx])
            if H is None or not np.isfinite(H).all():
                continue
        except np.linalg.LinAlgError:
            continue

        e = _symmetric_err(H, src_pts, dst_pts)
        inliers = np.where(e < threshold)[0]

        if inliers.size > best_inliers.size:
            best_inliers = inliers
            H_best = H
            if inliers.size > 0.85 * N:
                break

    if best_inliers.size < 4:
        print("RANSAC failed to find a consensus set ≥ 4.")
        return None, np.array([], dtype=int)
    print(len(src_pts[best_inliers]))
    print(src_pts[best_inliers])
    print(dst_pts[best_inliers])

    H_final, _ = cv2.findHomography(src_pts[best_inliers], dst_pts[best_inliers], method=0)

    print(f"RANSAC found {best_inliers.size} inliers out of {N} matches")
    return H_final, best_inliers

# ----- Bounds de la panoramica (3.6) -----
def calculate_panorama_bounds(img1, img2, img3, H_1to2, H_1to3):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h3, w3 = img3.shape[:2]
    
    corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners3 = np.array([[0, 0], [0, h3], [w3, h3], [w3, 0]], dtype=np.float32).reshape(-1, 1, 2)
    
    transformed_corners2 = cv2.perspectiveTransform(corners2, np.linalg.inv(H_1to2))
    transformed_corners3 = cv2.perspectiveTransform(corners3, np.linalg.inv(H_1to3))
    
    all_corners = np.concatenate([corners1, transformed_corners2, transformed_corners3], axis=0)
    
    x_min = np.min(all_corners[:, 0, 0])
    y_min = np.min(all_corners[:, 0, 1])
    x_max = np.max(all_corners[:, 0, 0])
    y_max = np.max(all_corners[:, 0, 1])
    
    offset_x = -x_min if x_min < 0 else 0
    offset_y = -y_min if y_min < 0 else 0
    
    panorama_width = int(np.ceil(x_max + offset_x))
    panorama_height = int(np.ceil(y_max + offset_y))
    return offset_x, offset_y, panorama_width, panorama_height