from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from utils_model import plot_snn_connections

class SNN_Autoencoder:
    """Spiking Neural Network Autoencoder for feature extraction."""

    def __init__(self, input_size, hidden_size, sim_time=100 * ms, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sim_time = sim_time
        self.learning_rate = learning_rate
        self._is_trained = False

        # Initialize weights
        self.encoder_weights = np.random.rand(input_size, hidden_size)
        self.decoder_weights = np.random.rand(hidden_size, input_size)

        # Build the static network components
        self._build_network()

    def _build_network(self):
        """Builds and stores the SNN network structure once."""
        self.input_group = SpikeGeneratorGroup(self.input_size, indices=[], times=[] * ms)

        eqs = '''
            dv/dt = (I-v)/tau : 1
            I : 1
            tau : second
        '''

        self.hidden_group = NeuronGroup(self.hidden_size,
                                        model=eqs,
                                        threshold='v > 1', reset='v = 0', method='exact')

        self.encoder_synapses = Synapses(self.input_group, self.hidden_group,
                                         model='''w : 1
                                                  dApre/dt = -Apre / (20*ms) : 1 (event-driven)
                                                  dApost/dt = -Apost / (20*ms) : 1 (event-driven)''',
                                         on_pre='''v_post += w
                                                   Apre += 1.
                                                   w = clip(w + Apost, 0, 1)''',
                                         on_post='''Apost += 1.
                                                    w = clip(w + Apre, 0, 1)''')
        self.encoder_synapses.connect()
        self.encoder_synapses.w = self.encoder_weights.flatten()

        self.output_group = NeuronGroup(
            self.input_size,
            eqs,
            threshold='v > 0.5',  # more sensitive threshold
            reset='v = 0'
        )

        self.decoder_synapses = Synapses(
            self.hidden_group, self.output_group,
            model='w : 1',
            on_pre='v_post += 2*w'  # amplify effect
        )

        self.decoder_synapses.connect()
        self.decoder_synapses.w = self.decoder_weights.flatten()

        # Bundle into a network
        self.net = Network(collect())
        self.net.store('initialized')

    def _generate_spike_train(self, sample):
        """Generates spike train indices and times using rate coding."""
        min_value = np.min(sample)
        if min_value < 0:
            sample = sample - min_value  # Make all values non-negative

        max_value = np.max(sample)
        if max_value > 0:
            sample = sample / max_value  # Normalize to [0, 1]

        spike_times = []
        spike_indices = []

        for i, val in enumerate(sample):
            if val > 0:  # Only generate spike if value is non-zero
                time = (1.0 - val) * float(self.sim_time)  # earlier spike for higher value
                spike_times.append(time * ms)
                spike_indices.append(i)

        return spike_indices, spike_times


    def train(self, X, epochs=10):
        """Trains the SNN Autoencoder."""
        print("Starting training...")
        plot_weights(self.encoder_weights, "Initial Encoder Weights")
        plot_weights(self.decoder_weights, "Initial Decoder Weights")

        for epoch in range(epochs):
            total_loss = 0

            for sample in X:
                spike_indices, spike_times = self._generate_spike_train(sample)

                # Update spike input
                self.input_group.set_spikes(indices=np.arange(self.input_size), times=spike_times)

                # Reset network state
                self.net.restore('initialized')

                # Run the simulation
                self.net.run(self.sim_time)

                # Hebbian-like encoder update
                pre_activity = sample.reshape(-1, 1)
                post_activity = self.hidden_group.v[:].reshape(1, -1)
                print("Post-activity:", post_activity)
                self.encoder_weights += self.learning_rate * pre_activity @ post_activity

                # Decoder update based on error signal
                reconstructed = self.output_group.v[:]
                hidden_activity = self.hidden_group.v[:].reshape(-1, 1)
                output_error = (sample - reconstructed).reshape(1, -1)
                self.decoder_weights += self.learning_rate * hidden_activity @ output_error

                # Push updated weights to the network
                self.encoder_synapses.w = self.encoder_weights.flatten()
                self.decoder_synapses.w = self.decoder_weights.flatten()

                # Compute loss
                loss = np.mean((sample - reconstructed) ** 2)
                total_loss += loss

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X):.4f}")
            plot_weights(self.encoder_weights, f"Encoder Weights After Epoch {epoch + 1}")
            plot_weights(self.decoder_weights, f"Decoder Weights After Epoch {epoch + 1}")

        plot_weights(self.encoder_weights, "Final Encoder Weights")
        plot_weights(self.decoder_weights, "Final Decoder Weights")
        self._is_trained = True
        print("Training completed.")

    def extract_features(self, X):
        """Extracts features using the trained encoder."""
        if not self._is_trained:
            raise RuntimeError("Autoencoder must be trained before extracting features.")

        features = []
        for sample in X:
            spike_indices, spike_times = self._generate_spike_train(sample)
            print(np.array(spike_times).shape)
            self.input_group.set_spikes(indices=np.arange(self.input_size), times=spike_times)
            self.net.restore('initialized')
            self.net.run(self.sim_time)
            features.append(self.hidden_group.v[:])

        return np.array(features)


def plot_weights(weights, title):
    """Plots a heatmap of the weights."""
    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Weight Magnitude')
    plt.title(title)
    plt.xlabel('Target Neurons')
    plt.ylabel('Source Neurons')
    plt.show()


if __name__ == "__main__":
    # Instantiate the autoencoder
    X = np.random.rand(5, 3)  # 5 samples, 3 features
    snn = SNN_Autoencoder(input_size=3, hidden_size=5, sim_time=100*ms)
    snn.train(X, epochs=3)

    # Extract features
    features = snn.extract_features(X)
    print("Extracted features shape:", features.shape)

    # Plot the connections
    plot_snn_connections(snn, title="SNN Connections After Training")
