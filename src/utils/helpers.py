import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def extract_texture_features(image):
    """
    Extract texture features using GLCM and LBP.
    Standardized for both training and testing.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. GLCM FEATURES
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        symmetric=True,
        normed=True
    )

    glcm_features = [
        graycoprops(glcm, "contrast").mean(),
        graycoprops(glcm, "dissimilarity").mean(),
        graycoprops(glcm, "homogeneity").mean(),
        graycoprops(glcm, "energy").mean(),
        graycoprops(glcm, "correlation").mean(),
        graycoprops(glcm, "ASM").mean(),
    ]

    # 2. LBP FEATURES
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    (lbp_hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
        density=True
    )

    return np.concatenate([glcm_features, lbp_hist])

def get_patch_features(image, patch_size=512):
    """
    Break image into patches and extract features from each.
    """
    h, w = image.shape[:2]
    all_features = []
    
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            features = extract_texture_features(patch)
            all_features.append(features)
            
    return np.array(all_features)
