from brian2 import Hz, ms, SpikeGeneratorGroup, run
import numpy as np
from sklearn.preprocessing import StandardScaler

class DimensionalitySafeKernelRCE:
    """Robust classification model using hyperspheres for RCE."""

    def __init__(self, num_prototypes=5, gamma=1.0, learning_rate=0.1,
                 max_epochs=100, activation_threshold=0.9):
        self.num_prototypes = num_prototypes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.activation_threshold = activation_threshold
        self._is_fitted = False

    def fit(self, X, y):
        """Trains the model on the given dataset."""
        print("Initializing RCE classifier...")
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

class SpikingRBFClassifier:
    """Spiking Neural Network-based RBF Classifier."""

    def __init__(self, gamma=1.0, learning_rate=0.1, momentum=0.9):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._is_fitted = False

    def fit(self, X, y):
        """Fits the model to the data."""
        print("Initializing training data...")
        self.X_train_ = X
        self.y_train_ = y
        self.weights_ = np.zeros(X.shape[1])
        print("Training data initialized.")

        # Training loop
        print("Starting training...")
        for epoch in range(100):  # Fixed number of epochs
            for i, sample in enumerate(X):
                similarities = np.exp(-self.gamma * np.linalg.norm(self.X_train_ - sample, axis=1) ** 2)
                predicted_label = self.y_train_[np.argmax(similarities)]
                if predicted_label != y[i]:
                    self.weights_ += self.learning_rate * (sample - self.weights_)
                    self.weights_ *= (1 - self.momentum)  # Apply momentum
        print("Training completed.")
        self._is_fitted = True

    def predict(self, X):
        """Predicts labels for the given input data."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")

        predictions = []
        for sample in X:
            # Compute RBF similarity
            similarities = np.exp(-self.gamma * np.linalg.norm(self.X_train_ - sample, axis=1) ** 2)
            predicted_label = self.y_train_[np.argmax(similarities)]
            predictions.append(predicted_label)
        return np.array(predictions)


if __name__ == "__main__":
    from brian2 import SpikeGeneratorGroup, run, ms

    # Teste simples de SpikeGeneratorGroup
    print("Testing SpikeGeneratorGroup...")
    indices = [0, 1, 2]
    times = [10 * ms, 20 * ms, 30 * ms]
    spike_group = SpikeGeneratorGroup(3, indices, times)
    run(10 * ms)
    print("Spike simulation completed successfully.")