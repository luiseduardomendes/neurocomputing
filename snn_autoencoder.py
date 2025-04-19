import torch
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np
import time  # Import the time module

class SNN_Autoencoder(nn.Module):
    """Spiking Neural Network Autoencoder for feature extraction using SNNtorch."""

    def __init__(self, input_size, hidden_size, sim_time=100, learning_rate=0.01, threshold=0.9):
        super(SNN_Autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sim_time = int(sim_time)  # Convert Brian2 time to integer timesteps
        self.learning_rate = learning_rate
        self._is_trained = False

        # Initialize weights
        self.encoder_weights = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.decoder_weights = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)

        # Define spiking neuron layers
        self.encoder_neuron = snn.Leaky(beta=0.9, threshold=threshold)
        self.decoder_neuron = snn.Leaky(beta=0.9, threshold=threshold)

    def forward(self, x):
        """Forward pass through the autoencoder."""
        batch_size = x.size(0)

        # Initialize membrane potentials
        mem_encoder = torch.zeros(batch_size, self.hidden_size)
        mem_decoder = torch.zeros(batch_size, self.input_size)

        # Spike activity
        spikes_encoder = torch.zeros(batch_size, self.hidden_size)
        spikes_decoder = torch.zeros(batch_size, self.input_size)

        # Simulate over time
        for t in range(self.sim_time):
            # Encoder: Input -> Hidden
            input_current = torch.matmul(x, self.encoder_weights)
            mem_encoder, spikes_encoder = self.encoder_neuron(input_current, mem_encoder)

            # Decoder: Hidden -> Output
            output_current = torch.matmul(spikes_encoder, self.decoder_weights)
            mem_decoder, spikes_decoder = self.decoder_neuron(output_current, mem_decoder)

        return spikes_decoder, spikes_encoder

    def train_model(self, X, epochs=10, batch_size=32):
        """Trains the SNN Autoencoder with batch processing."""
        print("Starting training...")
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Convert the dataset to a PyTorch tensor
        X = torch.tensor(X, dtype=torch.float32)

        # Split the data into batches
        num_samples = X.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

        for epoch in range(epochs):
            total_loss = 0

            for batch_idx in range(num_batches):
                # Get the current batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch = X[start_idx:end_idx]

                # Forward pass
                optimizer.zero_grad()
                reconstructed, _ = self.forward(batch)

                # Compute loss
                loss = loss_fn(reconstructed, batch)
                total_loss += loss.item()

                # Backward pass and weight update
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}")

        self._is_trained = True
        print("Training completed.")

    def extract_features(self, X):
        """Extracts features using the trained encoder."""
        if not self._is_trained:
            raise RuntimeError("Autoencoder must be trained before extracting features.")

        features = []
        for sample in X:
            sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            _, spikes_encoder = self.forward(sample)
            features.append(spikes_encoder.detach().numpy().squeeze())  # Remove extra dimensions

        return np.array(features)


def plot_weights(weights, title):
    """Plots a heatmap of the weights."""
    plt.figure(figsize=(8, 6))
    plt.imshow(weights.detach().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Weight Magnitude')
    plt.title(title)
    plt.xlabel('Target Neurons')
    plt.ylabel('Source Neurons')
    plt.show()


if __name__ == "__main__":
    # Start the timer
    start_time = time.time()

    # Instantiate the autoencoder
    X = np.random.rand(50, 3)  # 5 samples, 3 features
    snn_autoencoder = SNN_Autoencoder(input_size=3, hidden_size=2, sim_time=10, learning_rate=0.5)
    snn_autoencoder.train_model(X, epochs=25, batch_size=5)

    # Extract features
    features = snn_autoencoder.extract_features(X)
    print("Extracted features shape:", features.shape)

    # Plot the encoder and decoder weights
    # plot_weights(snn_autoencoder.encoder_weights, "Encoder Weights")
    # plot_weights(snn_autoencoder.decoder_weights, "Decoder Weights")

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")
