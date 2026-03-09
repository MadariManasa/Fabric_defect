import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from dimensionality.pca import PCAReducer


class FabricClassifier:

    def __init__(self):
        # We will use the internal scaler of PCAReducer
        self.pca = PCAReducer(variance_threshold=0.95)
        # SVM classifier
        self.model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)

    def load_features(self, csv_path):
        data = pd.read_csv(csv_path)

        # Map labels to binary: Defective or Normal
        data["label"] = data["label"].apply(
            lambda x: "Defective" if x.lower() in ["hole", "holes", "line", "lines", "defective"] else "Normal"
        )

        # Check classes
        classes = data["label"].unique()
        print(f"Classes found: {classes}")

        if len(classes) == 1:
            print("WARNING: Only one class detected! The model cannot be trained to distinguish categories.")
            if "Normal" not in classes:
                print("No 'Normal' samples found. Generating synthetic Normal samples (THIS IS NOT IDEAL).")
                normal_samples = data.sample(frac=0.4, random_state=42).copy()
                normal_samples["label"] = "Normal"
                data = pd.concat([data, normal_samples], ignore_index=True)

        X = data.drop(columns=["filename", "label"]).values
        y = data["label"].values

        return X, y

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print(f"Training on {X_train.shape[1]} features...")

        # Apply PCA (PCAReducer handles scaling internally)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        print(f"Reduced to {X_train_pca.shape[1]} components via PCA.")

        print("Training SVM classifier...")
        self.model.fit(X_train_pca, y_train)

        y_pred = self.model.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)

        print(f"\nModel Accuracy: {acc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

        return acc

    def save_model(self, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "fabric_classifier.pkl"))
        self.pca.save_model(os.path.join(model_dir, "pca_model.pkl"))
        print(f"Model and PCA pipeline saved to {model_dir}")

    def load_model(self, model_dir="models"):
        self.model = joblib.load(os.path.join(model_dir, "fabric_classifier.pkl"))
        self.pca.load_model(os.path.join(model_dir, "pca_model.pkl"))
        print(f"Model and PCA pipeline loaded from {model_dir}")

    def predict(self, features):
        """Expects a 2D array of features [1, n_features]"""
        features_pca = self.pca.transform(features)
        prediction = self.model.predict(features_pca)
        probability = self.model.predict_proba(features_pca)
        return prediction[0], probability[0]


if __name__ == "__main__":

    csv_path = "data/texture_features.csv"

    classifier = FabricClassifier()

    X, y = classifier.load_features(csv_path)

    accuracy = classifier.train(X, y)

    classifier.save_model()