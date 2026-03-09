import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os
from pathlib import Path


class PCAReducer:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.
    Reduces combined texture and edge features to principal components.
    """
    
    def __init__(self, n_components=None, variance_threshold=0.95, random_state=42):
        """
        Initialize PCA Reducer.
        
        Args:
            n_components (int or float): Number of components to keep.
                                         If None, uses variance_threshold.
                                         If float (0-1), keeps components explaining that variance.
                                         If int > 1, keeps that many components.
            variance_threshold (float): If n_components is None, keep components
                                       explaining this much variance (0-1). Default: 0.95
            random_state (int): Random state for reproducibility. Default: 42
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        
        # Initialize PCA (will be configured based on n_components)
        self.pca = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_features_in_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None
    
    def fit(self, X):
        """
        Fit PCA to the training data.
        
        Args:
            X (numpy.ndarray): Training features of shape (n_samples, n_features)
                              Combined texture + edge features
            
        Returns:
            self
        """
        # Store original feature count
        self.n_features_in_ = X.shape[1]
        
        # Standardize the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine n_components
        if self.n_components is None:
            # Use variance threshold
            n_comp = X.shape[1]  # Start with all components
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # Use variance threshold
            n_comp = min(X.shape[0], X.shape[1])
        else:
            # Use fixed number of components
            n_comp = min(self.n_components, X.shape[1])
        
        # Initialize and fit PCA
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca.fit(X_scaled)
        
        # Calculate explained variance
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        
        # If variance threshold is used, find optimal n_components
        if self.n_components is None:
            n_optimal = np.argmax(self.cumulative_variance_ratio_ >= self.variance_threshold) + 1
            self.pca = PCA(n_components=n_optimal, random_state=self.random_state)
            self.pca.fit(X_scaled)
            self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
            self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Transform data using fitted PCA.
        
        Args:
            X (numpy.ndarray): Features to transform of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Transformed features of shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform. Call fit() first.")
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        # Standardize and transform
        X_scaled = self.scaler.transform(X)
        X_transformed = self.pca.transform(X_scaled)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit PCA and transform data in one step.
        
        Args:
            X (numpy.ndarray): Training features
            
        Returns:
            numpy.ndarray: Transformed features
        """
        self.fit(X)
        return self.transform(X)
    
    def get_explained_variance(self):
        """
        Get explained variance information.
        
        Returns:
            dict: Dictionary containing variance information
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted first.")
        
        return {
            'n_components': self.pca.n_components_,
            'explained_variance_ratio': self.explained_variance_ratio_,
            'cumulative_variance_ratio': self.cumulative_variance_ratio_,
            'total_variance_explained': self.cumulative_variance_ratio_[-1]
        }
    
    def plot_explained_variance(self, save_path=None):
        """
        Plot explained variance and cumulative explained variance.
        
        Args:
            save_path (str): Path to save the plot. If None, displays the plot.
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted first.")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Explained variance per component
        axes[0].bar(range(1, len(self.explained_variance_ratio_) + 1),
                    self.explained_variance_ratio_)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Explained Variance by Component')
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Cumulative explained variance
        axes[1].plot(range(1, len(self.cumulative_variance_ratio_) + 1),
                     self.cumulative_variance_ratio_, 'bo-')
        axes[1].axhline(y=self.variance_threshold, color='r', linestyle='--',
                       label=f'Variance threshold ({self.variance_threshold})')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Variance plot saved to {save_path}")
        else:
            plt.show()
    
    def get_components(self):
        """
        Get the principal components (loadings).
        
        Returns:
            numpy.ndarray: Principal components of shape (n_components, n_features)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted first.")
        
        return self.pca.components_
    
    def get_principal_features_importance(self, top_n=10):
        """
        Get the most important features for each principal component.
        
        Args:
            top_n (int): Number of top features to return for each component
            
        Returns:
            dict: Dictionary with importance scores for each component
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted first.")
        
        importance_dict = {}
        for i, component in enumerate(self.pca.components_):
            top_indices = np.argsort(np.abs(component))[-top_n:][::-1]
            importance_dict[f'PC{i+1}'] = {
                'feature_indices': top_indices,
                'loadings': component[top_indices]
            }
        
        return importance_dict
    
    def inverse_transform(self, X_transformed):
        """
        Inverse transform from PCA space back to original feature space.
        
        Args:
            X_transformed (numpy.ndarray): Transformed features
            
        Returns:
            numpy.ndarray: Features in original space
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted first.")
        
        X_pca_inverse = self.pca.inverse_transform(X_transformed)
        X_original = self.scaler.inverse_transform(X_pca_inverse)
        
        return X_original
    
    def summary(self):
        """
        Print a summary of the PCA model.
        """
        if not self.is_fitted:
            print("PCA not fitted yet.")
            return
        
        print("=" * 60)
        print("PCA MODEL SUMMARY")
        print("=" * 60)
        print(f"Input Features: {self.n_features_in_}")
        print(f"Output Components: {self.pca.n_components_}")
        print(f"Dimensionality Reduction: {self.n_features_in_} → {self.pca.n_components_}")
        print(f"Total Variance Explained: {self.cumulative_variance_ratio_[-1]:.4f} ({self.cumulative_variance_ratio_[-1]*100:.2f}%)")
        print(f"Variance Threshold: {self.variance_threshold}")
        print("-" * 60)
        print(f"{'Component':<15} {'Variance Ratio':<20} {'Cumulative Variance':<20}")
        print("-" * 60)
        for i in range(min(5, len(self.explained_variance_ratio_))):
            print(f"PC{i+1:<13} {self.explained_variance_ratio_[i]:<19.4f} {self.cumulative_variance_ratio_[i]:<19.4f}")
        if len(self.explained_variance_ratio_) > 5:
            print("...")
        print("=" * 60)
    
    def save_model(self, model_path):
        """
        Save the fitted PCA model to a file.
        
        Args:
            model_path (str): Path to save the model file (.pkl)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before saving.")
        
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the entire model (PCA + Scaler)
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'n_features_in_': self.n_features_in_,
            'explained_variance_ratio_': self.explained_variance_ratio_,
            'cumulative_variance_ratio_': self.cumulative_variance_ratio_,
            'n_components': self.n_components,
            'variance_threshold': self.variance_threshold,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, model_path)
        print(f"PCA model saved to: {model_path}")
        print(f"  - Model size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    def load_model(self, model_path):
        """
        Load a previously fitted PCA model from a file.
        
        Args:
            model_path (str): Path to the saved model file (.pkl)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.n_features_in_ = model_data['n_features_in_']
        self.explained_variance_ratio_ = model_data['explained_variance_ratio_']
        self.cumulative_variance_ratio_ = model_data['cumulative_variance_ratio_']
        self.n_components = model_data['n_components']
        self.variance_threshold = model_data['variance_threshold']
        self.random_state = model_data['random_state']
        self.is_fitted = True
        
        print(f"PCA model loaded from: {model_path}")
        print(f"  - Components: {self.pca.n_components_}")
        print(f"  - Variance explained: {self.cumulative_variance_ratio_[-1]:.4f}")
    
    def save_transformed_data(self, X_transformed, data_path):
        """
        Save transformed PCA data to a file.
        
        Args:
            X_transformed (numpy.ndarray): Transformed features
            data_path (str): Path to save the data file (.npy)
        """
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(data_path, X_transformed)
        print(f"Transformed data saved to: {data_path}")
        print(f"  - Shape: {X_transformed.shape}")
        print(f"  - File size: {os.path.getsize(data_path) / 1024:.2f} KB")
    
    def load_transformed_data(self, data_path):
        """
        Load previously transformed PCA data from a file.
        
        Args:
            data_path (str): Path to the saved data file (.npy)
            
        Returns:
            numpy.ndarray: Transformed features
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        X_transformed = np.load(data_path)
        print(f"Transformed data loaded from: {data_path}")
        print(f"  - Shape: {X_transformed.shape}")
        return X_transformed
    
    def save_components(self, components_path):
        """
        Save principal components (loadings) to a CSV file for interpretation.
        
        Args:
            components_path (str): Path to save components (.npy or .csv)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before saving components.")
        
        Path(components_path).parent.mkdir(parents=True, exist_ok=True)
        
        if components_path.endswith('.npy'):
            np.save(components_path, self.pca.components_)
        else:
            # Save as CSV for better readability
            np.savetxt(components_path, self.pca.components_, delimiter=',')
        
        print(f"Components saved to: {components_path}")
        print(f"  - Shape: {self.pca.components_.shape}")


if __name__ == "__main__":
    # Example usage
    print("PCA Dimensionality Reduction Example")
    print("-" * 60)
    
    # Create synthetic combined features (texture + edge features)
    n_samples = 100
    n_texture_features = 20
    n_edge_features = 16
    n_total_features = n_texture_features + n_edge_features
    
    # Generate random combined features
    X = np.random.randn(n_samples, n_total_features)
    
    # Initialize PCA with 95% variance threshold
    pca_reducer = PCAReducer(variance_threshold=0.95)
    
    # Fit and transform data
    X_transformed = pca_reducer.fit_transform(X)
    
    print(f"PCA initialized and fitted successfully!")
    print(f"  - Input shape: {X.shape}")
    print(f"  - Output shape: {X_transformed.shape}")
    print(f"  - Original features: {X.shape[1]}")
    print(f"  - PCA components: {X_transformed.shape[1]}")
    print(f"  - Variance explained: {pca_reducer.cumulative_variance_ratio_[-1]:.4f}")
    
    # Print summary
    pca_reducer.summary()
    
    # Save outputs for classification
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS FOR CLASSIFICATION")
    print("=" * 60)
    
    # Create models directory if needed
    models_dir = "../../../models"
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the fitted PCA model
    model_path = os.path.join(models_dir, "pca_model.pkl")
    pca_reducer.save_model(model_path)
    
    # Save the transformed data
    data_path = os.path.join(models_dir, "pca_transformed_data.npy")
    pca_reducer.save_transformed_data(X_transformed, data_path)
    
    # Save principal components
    components_path = os.path.join(models_dir, "pca_components.csv")
    pca_reducer.save_components(components_path)
    
    print("\n" + "=" * 60)
    print("LOADING SAVED OUTPUTS")
    print("=" * 60)
    
    # Create a new PCA instance and load the saved model
    pca_loader = PCAReducer()
    pca_loader.load_model(model_path)
    
    # Load transformed data
    loaded_data = pca_loader.load_transformed_data(data_path)
    
    print(f"\nModel and data ready for classification!")
    print(f"  - Use {model_path} in classification module")
    print(f"  - Use {data_path} as feature input to classifier")
