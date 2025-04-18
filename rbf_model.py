import numpy as np

class RBFModel:
    """Radial Basis Function (RBF) model for classification."""

    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self._is_fitted = False

    def fit(self, X, y):
        """Trains the RBF model on the given dataset."""
        print("Initializing RBF model...")

        # Validate input data
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) must match")

        # Store training data
        self.X_train_ = X
        self.y_train_ = y

        # Debugging: Print shapes
        print(f"Training data shape: {self.X_train_.shape}")
        print(f"Training labels shape: {self.y_train_.shape}")

        # Compute pairwise RBF similarities for training data
        self.similarity_matrix_ = np.exp(-self.gamma * np.linalg.norm(X[:, np.newaxis] - X, axis=2) ** 2)

        # Store the model as fitted
        self._is_fitted = True
        print("Training completed. Similarity matrix computed.")

    def predict(self, X):
        """Predicts labels for the given input data."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")

        # Debugging: Print shapes
        print(f"Prediction data shape: {X.shape}")
        print(f"Training data shape: {self.X_train_.shape}")
        print(f"Training labels shape: {self.y_train_.shape}")

        predictions = []
        for sample in X:
            # Compute RBF similarity
            similarities = np.exp(-self.gamma * np.linalg.norm(self.X_train_ - sample, axis=1) ** 2)
            predicted_label = self.y_train_[np.argmax(similarities)]
            predictions.append(predicted_label)

        return np.array(predictions)