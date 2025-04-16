from brian2 import Hz, ms, SpikeGeneratorGroup, run
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

class DimensionalitySafeKernelRCE:
    """Robust classification model using kernel RCE with LDA and spike-based feature transformation."""
    
    def __init__(self, num_prototypes=5, kernel='rbf', gamma=1.0, learning_rate=0.1, 
                 momentum=0.9, max_epochs=100, activation_threshold=0.9):
        self.num_prototypes = num_prototypes
        self.kernel = kernel
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.activation_threshold = activation_threshold

        self.scaler = StandardScaler()
        self.lda = LinearDiscriminantAnalysis()
        self._is_fitted = False

    def spike_encode(self, X):
        """Encodes input data into spike trains."""
        spike_times = []
        neuron_indices = []
        for i, sample in enumerate(X):
            for j, value in enumerate(sample):
                # Normalize value to [0, 1]
                value = max(0.0, min(1.0, value))
                # Convert feature value to spike time
                spike_time = int((1.0 - value) * 100) * ms
                if spike_time not in spike_times:  # Ensure unique spike times
                    spike_times.append(spike_time)
                    neuron_indices.append(j)
        return SpikeGeneratorGroup(len(X[0]), neuron_indices, spike_times)

    def fit(self, X, y):
        """Trains the model on given dataset."""
        print("Starting data scaling...")
        X_scaled = self.scaler.fit_transform(X)
        print("Data scaling completed.")

        print("Fitting LDA...")
        self.lda.fit(X_scaled, y)
        print("LDA fitting completed.")

        print("Transforming data with LDA...")
        X_lda = self.lda.transform(X_scaled)
        print("Data transformation completed.")

        print("Encoding data as spikes...")
        spike_group = self.spike_encode(X_lda)
        print("Spike encoding completed.")

        # Initialize prototypes
        print("Initializing prototypes...")
        self.prototypes_ = np.zeros((self.num_prototypes, X_lda.shape[1]))
        self.radii_ = np.zeros(self.num_prototypes)
        self.labels_ = np.zeros(self.num_prototypes, dtype=int)
        print("Prototypes initialized.")

        # Assign prototypes based on class distribution
        print("Assigning prototypes...")
        unique_classes = np.unique(y)
        for idx, cls in enumerate(unique_classes):
            class_samples = X_lda[y == cls]
            chosen = class_samples[np.random.choice(len(class_samples), self.num_prototypes // len(unique_classes), replace=True)]
            self.prototypes_[idx * (self.num_prototypes // len(unique_classes)):(idx+1) * (self.num_prototypes // len(unique_classes))] = chosen
            self.labels_[idx * (self.num_prototypes // len(unique_classes)):(idx+1) * (self.num_prototypes // len(unique_classes))] = cls
        print("Prototypes assigned.")

        self._is_fitted = True
        print("Model fitting completed.")
        return self

    def predict(self, X):
        """Predicts labels for the given input data."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")

        X_scaled = self.scaler.transform(X)
        X_lda = self.lda.transform(X_scaled)
        similarities = np.dot(X_lda, self.prototypes_.T)

        predictions = np.array([self.labels_[np.argmax(similarities[i])] for i in range(X.shape[0])])
        return predictions

    def run(self, X):
        """Runs spike simulation for the given input data."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")

        print("Encoding data as spikes...")
        spike_group = self.spike_encode(X)
        print("Spike encoding completed.")

        print("Running spike simulation...")
        run(100 * ms)  # Simula a atividade de spikes por 100 ms
        print("Spike simulation completed.")

class SpikingRBFClassifier:
    """Spiking Neural Network-based RBF Classifier."""
    
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.scaler = StandardScaler()
        self._is_fitted = False

    def spike_encode(self, X):
        """Encodes input data into spike trains."""
        spike_times = []
        neuron_indices = []
        for i, sample in enumerate(X):
            for j, value in enumerate(sample):
                # Normalize value to [0, 1]
                value = max(0.0, min(1.0, value))
                # Convert feature value to spike time
                spike_time = int((1.0 - value) * 100) * ms
                spike_times.append(spike_time)
                neuron_indices.append(j)
        print(f"Spike times: {spike_times}")
        print(f"Neuron indices: {neuron_indices}")
        return SpikeGeneratorGroup(len(X[0]), neuron_indices, spike_times)

    def fit(self, X, y):
        """Fits the model to the data."""
        X_scaled = self.scaler.fit_transform(X)
        self.X_train_ = X_scaled
        self.y_train_ = y
        self._is_fitted = True

    def predict(self, X):
        """Predicts labels for the given input data."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")

        X_scaled = self.scaler.transform(X)
        predictions = []
        for sample in X_scaled:
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