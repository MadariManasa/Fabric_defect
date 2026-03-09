import os
import cv2
import csv
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers import extract_texture_features

def extract_features_from_directory(src_dir, output_csv):
    """
    Reads all processed images -> extracts texture features -> saves into CSV.
    """
    features_list = []

    for root, _, files in os.walk(src_dir):
        label = os.path.basename(root)
        if not label: continue # Skip if root is empty
        
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                path = os.path.join(root, file)
                img = cv2.imread(path)

                if img is None:
                    print(f"Skipping unreadable image: {path}")
                    continue

                # For large images, extracting one feature vector might be too coarse.
                # However, for the first pass, we maintain the original logic but fixed.
                features = extract_texture_features(img)

                row = [file, label] + features.tolist()
                features_list.append(row)
                print(f"Extracted: {path} (Label: {label})")

    if not features_list:
        print("No images found to extract features from.")
        return

    # Save to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename", "label"] + [f"f{i}" for i in range(len(features_list[0]) - 2)]
        writer.writerow(header)
        writer.writerows(features_list)

    print(f"\nFeature extraction completed! Total samples: {len(features_list)}")
    print(f"Saved to: {output_csv}")


if __name__ == "__main__":
    processed_dir = "data/processed"
    output_file = "data/texture_features.csv"

    extract_features_from_directory(processed_dir, output_file)