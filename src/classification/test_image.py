import sys
import os
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classification.classifier import FabricClassifier
from utils.helpers import extract_texture_features

if __name__ == "__main__":
    # -------------------------
    # Load trained model
    # -------------------------
    classifier = FabricClassifier()
    try:
        classifier.load_model("models")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please run src/classification/classifier.py first to train and save the model.")
        sys.exit(1)

    # -------------------------
    # Test Image
    # -------------------------
    # You can change this to any image path
    image_path = "download.jpg" 

    if not os.path.exists(image_path):
        # Fallback to a known dataset image if download.jpg doesn't exist or for testing
        image_path = r"data/captured/Hole/20180531_134722.jpg"

    print(f"Testing on image: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image {image_path}")
        sys.exit(1)

    # If the image is very large, maybe resize it? 
    # But for consistency with training, we use it as is.
    # Training images were 4608x3456.
    # If download.jpg is only 259x194, results will be poor.

    features = extract_texture_features(img)
    features = features.reshape(1, -1)

    # Prediction
    prediction, probabilities = classifier.predict(features)

    print("\n" + "=" * 30)
    print(f"PREDICTION: {prediction}")
    print(f"CONFIDENCE: {np.max(probabilities)*100:.2f}%")
    print("=" * 30)

    # Print all probabilities
    classes = classifier.model.classes_
    for i, cls in enumerate(classes):
        print(f"{cls}: {probabilities[i]*100:.2f}%")
