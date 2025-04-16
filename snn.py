from brian2 import *
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import load_iris, load_wine
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
prefs.codegen.target = 'numpy'


from sklearn.model_selection import KFold

def kfold_cross_validation(X, y, model_type, input_size, hidden_size, output_size, k=5):
    """
    Perform K-Fold Cross-Validation.
    :param X: Input data (spike times).
    :param y: Target labels.
    :param model_type: 'RBF' or 'RCE'.
    :param input_size: Number of input neurons.
    :param hidden_size: Number of hidden neurons.
    :param output_size: Number of output neurons.
    :param k: Number of folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{k}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Initialize and train the model
        model = SpikingMLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size, mode=model_type)
        model.build_network()
        model.train(X_train, y_train, epochs=10)

        # Predict and evaluate
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        accuracies.append(accuracy)
        print(f"Accuracy for fold {fold + 1}: {accuracy * 100:.2f}%")

    # Print overall results
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"\nOverall Accuracy ({model_type}): {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%")
    return accuracies

class SpikingMLP:
    def __init__(self, input_size, hidden_size, output_size, mode='RBF'):
        """
        Initialize the Spiking MLP.
        :param input_size: Number of input neurons.
        :param hidden_size: Number of hidden neurons.
        :param output_size: Number of output neurons.
        :param mode: 'RBF' or 'RCE' for the output layer logic.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mode = mode  # 'RBF' or 'RCE'
        self.centroids = None
        self.radii = None

        # Define neuron model
        self.eqs = '''
        dv/dt = (I - v) / (10*ms) : 1
        I : 1
        '''

    def build_network(self):
        # Input layer
        self.input_layer = SpikeGeneratorGroup(self.input_size, [0], [0*ms])  # Inicializa com um spike fictício

        # Hidden layer
        self.hidden_layer = NeuronGroup(self.hidden_size, self.eqs, threshold='v > 1', reset='v = 0', method='exact')
        self.hidden_synapses = Synapses(self.input_layer, self.hidden_layer, on_pre='v_post += 0.5')
        self.hidden_synapses.connect()

        # Output layer
        self.output_layer = NeuronGroup(self.output_size, self.eqs, threshold='v > 1', reset='v = 0', method='exact')
        self.output_synapses = Synapses(self.hidden_layer, self.output_layer, on_pre='v_post += 0.5')
        self.output_synapses.connect()

    def train(self, X, y, epochs=10):
        """
        Train the Spiking MLP.
        :param X: Input data (spike times).
        :param y: Target labels.
        :param epochs: Number of training epochs.
        """
        print("Training Spiking MLP...")
        self.centroids = []
        self.radii = []

        for epoch in range(epochs):
            for i, sample in enumerate(X):
                spike_times = [(j, t*ms) for j, t in enumerate(sample) if t > 0]
                if spike_times:  # Verifica se há spikes válidos
                    indices, times = zip(*spike_times)
                    self.input_layer.set_spikes(indices, times)
                else:
                    self.input_layer.set_spikes([], [])  # Nenhum spike para esta amostra
                run(100*ms)

            # Update centroids and radii for RBF or RCE
            for cls in np.unique(y):
                class_samples = X[y == cls]
                centroid = np.mean(class_samples, axis=0)
                self.centroids.append(centroid)
                if self.mode == 'RCE':
                    radius = np.max(np.linalg.norm(class_samples - centroid, axis=1))
                    self.radii.append(radius)

    def predict(self, X):
        """
        Predict using the Spiking MLP.
        :param X: Input data (spike times).
        :return: Predicted labels.
        """
        predictions = []
        for sample in X:
            spike_times = [(j, t*ms) for j, t in enumerate(sample) if t > 0]
            if spike_times:  # Verifica se há spikes válidos
                indices, times = zip(*spike_times)
                self.input_layer.set_spikes(indices, times)
            else:
                self.input_layer.set_spikes([], [])  # Nenhum spike para esta amostra
            run(100*ms)

            # Calculate activations for RBF or RCE
            activations = []
            for i, centroid in enumerate(self.centroids):
                time_diff = np.abs(sample - centroid)
                if self.mode == 'RBF':
                    activation = -np.sum(time_diff)  # Similarity (negative distance)
                elif self.mode == 'RCE':
                    activation = np.sum(time_diff <= self.radii[i])  # Count spikes within radius
                activations.append(activation)

            predictions.append(np.argmax(activations))
        return predictions

def plot_accuracy(accuracies, title):
    """Plot accuracy over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

def plot_class_distribution(y_true, y_pred, title):
    """Plot class distribution of predictions vs true labels."""
    plt.figure(figsize=(8, 5))
    plt.hist([y_true, y_pred], bins=np.arange(len(np.unique(y_true)) + 1) - 0.5, label=['True', 'Predicted'], alpha=0.7)
    plt.xticks(np.arange(len(np.unique(y_true))))
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Normalize features to spike times
    X = X / np.max(X)  # Normalize to [0, 1]
    X = X * 100  # Map to spike times

    # Perform K-Fold Cross-Validation for Iris dataset
    print("\n=== K-Fold Cross-Validation (Iris - RBF) ===")
    kfold_cross_validation(X, y, model_type='RBF', input_size=4, hidden_size=10, output_size=3, k=5)

    print("\n=== K-Fold Cross-Validation (Iris - RCE) ===")
    kfold_cross_validation(X, y, model_type='RCE', input_size=4, hidden_size=10, output_size=3, k=5)

    # Load Wine dataset
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target

    # Normalize features to spike times
    X_wine = X_wine / np.max(X_wine)  # Normalize to [0, 1]
    X_wine = X_wine * 100  # Map to spike times

    # Perform K-Fold Cross-Validation for Wine dataset
    print("\n=== K-Fold Cross-Validation (Wine - RBF) ===")
    kfold_cross_validation(X_wine, y_wine, model_type='RBF', input_size=X_wine.shape[1], hidden_size=10, output_size=len(np.unique(y_wine)), k=5)

    print("\n=== K-Fold Cross-Validation (Wine - RCE) ===")
    kfold_cross_validation(X_wine, y_wine, model_type='RCE', input_size=X_wine.shape[1], hidden_size=10, output_size=len(np.unique(y_wine)), k=5)