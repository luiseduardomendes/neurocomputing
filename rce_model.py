import numpy as np

class RCEModel:
    """Reduced Classification Error (RCE) model using hyperspheres."""

    def __init__(self, num_prototypes=5, gamma=1.0, learning_rate=0.1, max_epochs=100):
        self.num_prototypes = num_prototypes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self._is_fitted = False

    def fit(self, X, y):
        """Trains the RCE model on the given dataset."""
        print("Initializing RCE model...")
        self.prototypes_ = []  # List of prototypes (hypersphere centers)
        self.radii_ = []       # List of radii for each hypersphere
        self.labels_ = []      # List of labels for each hypersphere

        # Training loop
        print("Starting training...")
        for epoch in range(self.max_epochs):
            all_correct = True  # Flag to check if all patterns are correctly classified

            for i, sample in enumerate(X):
                distances = np.array([np.linalg.norm(sample - p) for p in self.prototypes_])
                within_radii = distances <= np.array(self.radii_)

                if np.any(within_radii):  # If the sample falls within any hypersphere
                    closest_idx = np.argmin(distances[within_radii])
                    closest_label = self.labels_[np.where(within_radii)[0][closest_idx]]

                    if closest_label == y[i]:  # Correct classification
                        continue
                    else:  # Misclassification
                        all_correct = False
                        for j, label in enumerate(self.labels_):
                            if label != y[i] and distances[j] <= self.radii_[j]:
                                self.radii_[j] *= (1 - self.learning_rate)  # Reduce radius
                        self.prototypes_.append(sample)  # Insert new prototype
                        self.radii_.append(self.gamma)  # Initialize radius
                        self.labels_.append(y[i])
                else:  # Unknown classification
                    all_correct = False
                    self.prototypes_.append(sample)  # Insert new prototype
                    self.radii_.append(self.gamma)  # Initialize radius
                    self.labels_.append(y[i])

            if all_correct:  # If all patterns are correctly classified, stop training
                print(f"All patterns correctly classified at epoch {epoch + 1}.")
                break

        print("Training completed.")
        self._is_fitted = True

    def predict(self, X):
        """Predicts labels for the given input data."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")

        predictions = []
        for sample in X:
            distances = np.array([np.linalg.norm(sample - p) for p in self.prototypes_])
            within_radii = distances <= np.array(self.radii_)

            if np.any(within_radii):  # If the sample falls within any hypersphere
                closest_idx = np.argmin(distances[within_radii])
                predicted_label = self.labels_[np.where(within_radii)[0][closest_idx]]
            else:  # If no hypersphere contains the sample, assign the label of the closest prototype
                predicted_label = self.labels_[np.argmin(distances)]
            predictions.append(predicted_label)

        return np.array(predictions)