from brian2 import Hz, ms, SpikeGeneratorGroup, NeuronGroup, Synapses, SpikeMonitor, run
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
    """Improved Spiking Neural Network-based RBF Classifier."""

    def __init__(self, gamma=1.0, sim_time=100*ms, num_classes=3, learning_rate=0.01):
        self.gamma = gamma
        self.sim_time = sim_time
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self._is_fitted = False

    def fit(self, X, y):
        """Fits the model to the data."""
        print("Initializing training data...")
        self.X_train_ = X
        self.y_train_ = y

        # Initialize synaptic weights (randomly for now)
        self.weights_ = np.random.rand(len(X[0]), self.num_classes)
        print("Training data initialized.")

        # Precompute spike trains for all samples
        print("Precomputing spike trains...")
        self.spike_trains_ = [self._generate_spike_train(sample) for sample in X]

        # Training loop
        print("Starting training...")
        for epoch in range(10):  # Fixed number of epochs
            for i, spike_times in enumerate(self.spike_trains_):
                input_group = SpikeGeneratorGroup(len(spike_times), indices=np.arange(len(spike_times)), times=spike_times)

                # Simulate spiking dynamics
                output_group = NeuronGroup(self.num_classes, 'dv/dt = -v / (10*ms) : 1', threshold='v > 1', reset='v = 0')
                synapses = Synapses(input_group, output_group, 'w : 1', on_pre='v_post += w')
                synapses.connect()
                synapses.w = self.weights_.flatten()

                monitor = SpikeMonitor(output_group)
                run(self.sim_time)

                # Update weights based on spike activity (gradient-like update)
                spike_counts = monitor.count
                target_class = y[i]
                for j in range(self.num_classes):
                    if j == target_class:
                        self.weights_[:, j] += self.learning_rate * spike_counts[j]  # Strengthen correct class
                    else:
                        self.weights_[:, j] -= self.learning_rate * spike_counts[j]  # Weaken incorrect classes

            # Normalize weights to prevent divergence
            self.weights_ = self.weights_ / np.linalg.norm(self.weights_, axis=0)

        print("Training completed.")
        self._is_fitted = True

    def predict(self, X):
        """Predicts labels for the given input data."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")

        predictions = []
        for sample in X:
            # Convert input sample to spike train
            spike_times = self._generate_spike_train(sample)
            input_group = SpikeGeneratorGroup(len(sample), indices=np.arange(len(sample)), times=spike_times)

            # Simulate spiking dynamics
            output_group = NeuronGroup(self.num_classes, 'dv/dt = -v / (10*ms) : 1', threshold='v > 1', reset='v = 0')
            synapses = Synapses(input_group, output_group, on_pre='v_post += w')
            synapses.connect()
            synapses.w = self.weights_.flatten()

            monitor = SpikeMonitor(output_group)
            run(self.sim_time)

            # Determine the predicted label based on firing rates
            spike_counts = monitor.count
            predicted_label = np.argmax(spike_counts)
            predictions.append(predicted_label)

        return np.array(predictions)

    def _generate_spike_train(self, sample):
        """Generates a spike train for the given sample using rate coding."""
        # Normalize the sample to ensure all values are between 0 and 1
        min_value = np.min(sample)
        if min_value < 0:
            sample = sample - min_value  # Shift all values to be non-negative

        max_value = np.max(sample)
        if max_value > 0:
            sample = sample / max_value  # Normalize to [0, 1]

        # Convert normalized values to spike times (earlier for larger values)
        spike_times = (1.0 - sample) * float(self.sim_time)  # Invert so larger values spike earlier
        return [t * ms for t in spike_times]


if __name__ == "__main__":
    from brian2 import SpikeGeneratorGroup, run, ms

    # Teste simples de SpikeGeneratorGroup
    print("Testing SpikeGeneratorGroup...")
    indices = [0, 1, 2]
    times = [10 * ms, 20 * ms, 30 * ms]
    spike_group = SpikeGeneratorGroup(3, indices, times)
    run(10 * ms)
    print("Spike simulation completed successfully.")