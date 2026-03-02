import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import csv


def extract_texture_features(image):
    """
    Extract ONLY texture features:
    - GLCM features (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
    - LBP histogram (uniform)
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # 1. GLCM FEATURES  (NEW API)
    # -------------------------
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

    # -------------------------
    # 2. LBP FEATURES
    # -------------------------
    radius = 3
    n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    (lbp_hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
        density=True
    )

    # Combine → final feature vector
    feature_vector = np.concatenate([glcm_features, lbp_hist])

    return feature_vector


def extract_features_from_directory(src_dir, output_csv):
    """
    Reads all processed images → extracts texture features → saves into CSV.
    """

    features_list = []

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                path = os.path.join(root, file)
                img = cv2.imread(path)

                if img is None:
                    print(f"Skipping unreadable image: {path}")
                    continue

                features = extract_texture_features(img)

                # label = folder name (e.g., Hole, Line, Vertical, etc.)
                label = os.path.basename(root)

                row = [file, label] + features.tolist()
                features_list.append(row)
                print(f"Extracted: {path}")

    # Save to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename", "label"] + [f"f{i}" for i in range(len(features_list[0]) - 2)]
        writer.writerow(header)
        writer.writerows(features_list)

    print(f"\nFeature extraction completed! Saved to: {output_csv}")


if __name__ == "__main__":
    processed_dir = "data/processed"
    output_file = "data/texture_features.csv"

    extract_features_from_directory(processed_dir, output_file)