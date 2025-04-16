from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import KFold

prefs.codegen.target = 'numpy'

class CoincidenceRBF:
    def __init__(self, centroids):
        self.centroids = centroids  # Fixed spikes (centroids)

    def predict(self, X):
        """Predicts the class based on coincidence detection."""
        predictions = []
        for sample in X:
            distances = []
            for centroid in self.centroids:
                # Calculate the time difference between input spikes and centroid spikes
                time_diff = np.abs(sample - centroid)
                distances.append(np.sum(time_diff))
            predictions.append(np.argmin(distances))  # Class with minimum distance
        return predictions


class CoincidenceRCE:
    def __init__(self, centroids, radii):
        self.centroids = centroids  # Fixed spikes (centroids)
        self.radii = radii  # Activation radii for each prototype

    def predict(self, X):
        """Predicts the class based on coincidence detection with radii."""
        predictions = []
        for sample in X:
            activations = []
            for i, centroid in enumerate(self.centroids):
                # Calculate the time difference between input spikes and centroid spikes
                time_diff = np.abs(sample - centroid)
                activation = np.sum(time_diff <= self.radii[i])  # Count spikes within radius
                activations.append(activation)
            predictions.append(np.argmax(activations))  # Class with maximum activation
        return predictions


def train_rbf_rce(X_train, y_train):
    """Train RBF and RCE models using centroids."""
    # Calculate centroids for each class
    centroids = []
    radii = []
    for cls in np.unique(y_train):
        class_samples = X_train[y_train == cls]
        centroid = np.mean(class_samples, axis=0)
        radius = np.max(np.linalg.norm(class_samples - centroid, axis=1))
        centroids.append(centroid)
        radii.append(radius)

    return np.array(centroids), np.array(radii)


def evaluate_rbf_rce_cross_validation(X, y, n_splits=5):
    """Evaluate RBF and RCE models using 5-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rbf_accuracies = []
    rce_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train RBF and RCE models
        centroids, radii = train_rbf_rce(X_train, y_train)
        rbf_model = CoincidenceRBF(centroids)
        rce_model = CoincidenceRCE(centroids, radii)

        # Evaluate RBF
        rbf_predictions = rbf_model.predict(X_test)
        rbf_accuracy = np.mean(rbf_predictions == y_test)
        rbf_accuracies.append(rbf_accuracy)
        print(f"RBF Accuracy for fold {fold + 1}: {rbf_accuracy * 100:.2f}%")

        # Evaluate RCE
        rce_predictions = rce_model.predict(X_test)
        rce_accuracy = np.mean(rce_predictions == y_test)
        rce_accuracies.append(rce_accuracy)
        print(f"RCE Accuracy for fold {fold + 1}: {rce_accuracy * 100:.2f}%")

    # Print overall results
    print("\nOverall RBF Accuracy: {:.2f}% ± {:.2f}%".format(
        np.mean(rbf_accuracies) * 100, np.std(rbf_accuracies) * 100))
    print("Overall RCE Accuracy: {:.2f}% ± {:.2f}%".format(
        np.mean(rce_accuracies) * 100, np.std(rce_accuracies) * 100))


def coincidence_detection(X, centroids):
    """
    Apply coincidence detection to classify samples based on centroids.
    :param X: Input samples (spike times).
    :param centroids: Fixed spikes (centroids for each class).
    :return: Delays of spiking for each sample.
    """
    delays = []

    # Define neuron model
    eqs = '''
    dv/dt = (I - v) / (10*ms) : 1
    I : 1
    '''

    # Create neuron group
    neuron = NeuronGroup(1, eqs, threshold='v > 1', reset='v = 0', method='exact')

    # Create synapses (fixed and variable spikes)
    fixed_spike = SpikeGeneratorGroup(1, [0], [0*ms])  # Placeholder for fixed spike
    variable_spike = SpikeGeneratorGroup(1, [0], [0*ms])  # Placeholder for variable spike
    syn_fixed = Synapses(fixed_spike, neuron, on_pre='v_post += 1.0')  # Increased impact
    syn_variable = Synapses(variable_spike, neuron, on_pre='v_post += 1.0')  # Increased impact
    syn_fixed.connect()
    syn_variable.connect()

    # Spike monitor to record spike times
    spike_monitor = SpikeMonitor(neuron)

    # Store the initial state of the simulation
    store('initial')

    # Process each sample
    for sample in X:
        sample_delays = []
        for centroid in centroids:
            # Set spike times
            fixed_spike.set_spikes([0], [centroid[0] * ms])
            variable_spike.set_spikes([0], [sample[0] * ms])

            # Reset simulation state
            restore('initial')

            # Run simulation
            run(100*ms)  # Increased simulation time

            # Calculate delay
            if len(spike_monitor.t) > 0:  # Check if the neuron spiked
                spike_time = spike_monitor.t[0] / ms  # Get the spike time in ms
                delay = spike_time - centroid[0]  # Calculate delay relative to fixed spike
                sample_delays.append(delay)
            else:
                sample_delays.append(None)  # No spike occurred

        delays.append(sample_delays)

    return delays

def coincidence_detection_with_plot(X, centroids):
    """
    Apply coincidence detection and plot delay of spiking vs delta t.
    :param X: Input samples (spike times).
    :param centroids: Fixed spikes (centroids for each class).
    """
    for i, centroid in enumerate(centroids):
        delays = []
        delta_t_values = range(-10, 11)  # Delta t from -10 to +10 ms

        # Define neuron model
        eqs = '''
        dv/dt = (I - v) / (10*ms) : 1
        I : 1
        '''

        # Create neuron group
        neuron = NeuronGroup(1, eqs, threshold='v > 1', reset='v = 0', method='exact')

        # Create synapses (fixed and variable spikes)
        fixed_spike = SpikeGeneratorGroup(1, [0], [centroid[0] * ms])  # Fixed spike
        variable_spike = SpikeGeneratorGroup(1, [0], [0*ms])  # Placeholder for variable spike
        syn_fixed = Synapses(fixed_spike, neuron, on_pre='v_post += 1.0')  # Increased impact
        syn_variable = Synapses(variable_spike, neuron, on_pre='v_post += 1.0')  # Increased impact
        syn_fixed.connect()
        syn_variable.connect()

        # Spike monitor to record spike times
        spike_monitor = SpikeMonitor(neuron)

        # Store the initial state of the simulation
        store('initial')

        # Process each delta t
        for delta_t in delta_t_values:
            # Set spike times
            variable_spike.set_spikes([0], [(centroid[0] + delta_t) * ms])

            # Reset simulation state
            restore('initial')

            # Run simulation
            run(100*ms)

            # Calculate delay
            if len(spike_monitor.t) > 0:  # Check if the neuron spiked
                spike_time = spike_monitor.t[0] / ms  # Get the spike time in ms
                delay = spike_time - centroid[0]  # Calculate delay relative to fixed spike
                delays.append(delay)
            else:
                delays.append(None)  # No spike occurred

        # Plot results
        plt.figure(figsize=(8, 5))
        plt.plot(delta_t_values, delays, marker='o')
        plt.xlabel('Delta t (ms)')
        plt.ylabel('Delay of Spiking (ms)')
        plt.title(f'Coincidence Detection: Centroid {i + 1}')
        plt.grid()
        plt.show()

if __name__ == "__main__":
    # Load Iris dataset
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target

    # Normalize features to spike times
    X_iris = X_iris / np.max(X_iris)  # Normalize to [0, 1]
    X_iris = X_iris * 100  # Map to spike times

    # Train RBF and RCE models
    print("\n=== Iris Dataset ===")
    centroids_iris, radii_iris = train_rbf_rce(X_iris, y_iris)
    print("Centroids for Iris Dataset:")
    print(centroids_iris)

    # Apply coincidence detection with plot for Iris
    coincidence_detection_with_plot(X_iris, centroids_iris)

    # Load Wine dataset
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target

    # Normalize features to spike times
    X_wine = X_wine / np.max(X_wine)  # Normalize to [0, 1]
    X_wine = X_wine * 100  # Map to spike times

    # Train RBF and RCE models
    print("\n=== Wine Dataset ===")
    centroids_wine, radii_wine = train_rbf_rce(X_wine, y_wine)

    # Apply coincidence detection with plot for Wine
    coincidence_detection_with_plot(X_wine, centroids_wine)