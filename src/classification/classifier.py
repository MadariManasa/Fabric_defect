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
import os

# Import your PCA module
from dimensionality.pca import PCAReducer


class FabricClassifier:

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCAReducer(variance_threshold=0.95)
        self.model = SVC(kernel='rbf', C=10, gamma='scale')

    def load_features(self, csv_path):

        data = pd.read_csv(csv_path)

        X = data.drop(columns=["filename", "label"]).values
        y = data["label"].values

        return X, y

    def train(self, X, y):

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Apply PCA
        X_train = self.pca.fit_transform(X_train)
        X_test = self.pca.transform(X_test)

        print("After PCA reduction:", X_train.shape)

        # Train SVM
        print("Training classifier...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print("\nAccuracy:", acc)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

        return acc

    def save_model(self):

        os.makedirs("models", exist_ok=True)

        joblib.dump(self.model, "models/fabric_classifier.pkl")
        joblib.dump(self.scaler, "models/scaler.pkl")

        print("Model saved successfully")


if __name__ == "__main__":

    csv_path = "data/texture_features.csv"

    classifier = FabricClassifier()

    X, y = classifier.load_features(csv_path)

    accuracy = classifier.train(X, y)

    classifier.save_model()