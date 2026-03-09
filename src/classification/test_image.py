import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from dimensionality.pca import PCAReducer


# -------------------------
# Feature Extraction
# -------------------------

def extract_texture_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # GLCM
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

    # LBP
    radius = 3
    n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    (lbp_hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
        density=True
    )

    features = np.concatenate([glcm_features, lbp_hist])

    return features


# -------------------------
# Load trained models
# -------------------------

model = joblib.load("models/fabric_classifier.pkl")
scaler = joblib.load("models/scaler.pkl")

pca = PCAReducer()
pca.load_model("models/pca_model.pkl")


# -------------------------
# Test Image
# -------------------------
image_path = r"C:\Users\admin\Desktop\cvpr_project\Fabric_defect\data\captured\Lines\line_2018-09-19 15_19_10.492631.jpg"

print("File exists:", os.path.exists(image_path))

img = cv2.imread(image_path)

if img is None:
    print("❌ Error: Could not load image. Check the file path.")
    exit()

print("✅ Image loaded successfully")
features = extract_texture_features(img)

features = features.reshape(1, -1)

# STEP 1 — Match scaler features (32)
features_scaled = scaler.transform(features)

# STEP 2 — Expand to PCA input size (36)
expected_pca_features = pca.n_features_in_

if features_scaled.shape[1] < expected_pca_features:
    padding = np.zeros((1, expected_pca_features - features_scaled.shape[1]))
    features_scaled = np.hstack((features_scaled, padding))

# STEP 3 — PCA
features_pca = pca.transform(features_scaled)

# STEP 4 — Match SVM expected features
features_pca = features_pca[:, :4]

# STEP 5 — Prediction
prediction = model.predict(features_pca)

print("\nPrediction:", prediction[0])