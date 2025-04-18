import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def visualize_latent_space(features, labels, target_names=None, method='pca', title='Latent Feature Space'):
    """
    Visualizes the latent features (2D projection).
    
    Parameters:
    - features: array-like of shape (n_samples, n_features)
    - labels: array-like of shape (n_samples,)
    - target_names: list of class names
    - method: 'pca' or 'tsne'
    - title: plot title
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

    reduced = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for class_idx in np.unique(labels):
        mask = labels == class_idx
        label_name = target_names[class_idx] if target_names else str(class_idx)
        plt.scatter(reduced[mask, 0], reduced[mask, 1], label=label_name, alpha=0.6)
    
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_snn_connections(autoencoder, title="SNN Connections with Weight-Based Fading"):
    """
    Plots the connections in the SNN Autoencoder with fading based on synaptic weights.

    Parameters:
        autoencoder (SNN_Autoencoder): The autoencoder instance to visualize.
        title (str): Title of the plot.
    """
    # Extract encoder connections and weights
    encoder_source = autoencoder.encoder_synapses.i[:]  # Presynaptic neurons
    encoder_target = autoencoder.encoder_synapses.j[:]  # Postsynaptic neurons
    encoder_weights = autoencoder.encoder_synapses.w[:]  # Synaptic weights

    # Extract decoder connections and weights
    decoder_source = autoencoder.decoder_synapses.i[:]  # Presynaptic neurons
    decoder_target = autoencoder.decoder_synapses.j[:]  # Postsynaptic neurons
    decoder_weights = autoencoder.decoder_synapses.w[:]  # Synaptic weights

    # Normalize weights for alpha scaling (0 to 1)
    encoder_weights_normalized = encoder_weights / np.max(encoder_weights)
    decoder_weights_normalized = decoder_weights / np.max(decoder_weights)

    # Create positions for neurons
    input_positions = np.linspace(0, 1, autoencoder.input_size)
    hidden_positions = np.linspace(0, 1, autoencoder.hidden_size)
    output_positions = np.linspace(0, 1, autoencoder.input_size)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot input neurons
    for i, pos in enumerate(input_positions):
        ax.scatter(0, pos, color='blue', s=50, label='Input Neurons' if i == 0 else "")
    # Plot hidden neurons
    for i, pos in enumerate(hidden_positions):
        ax.scatter(0.5, pos, color='orange', s=50, label='Hidden Neurons' if i == 0 else "")
    # Plot output neurons
    for i, pos in enumerate(output_positions):
        ax.scatter(1, pos, color='green', s=50, label='Output Neurons' if i == 0 else "")

    # Draw encoder connections (Input -> Hidden) with fading
    for src, tgt, weight in zip(encoder_source, encoder_target, encoder_weights_normalized):
        ax.plot([0, 0.5], [input_positions[src], hidden_positions[tgt]], color='gray', alpha=weight)

    # Draw decoder connections (Hidden -> Output) with fading
    for src, tgt, weight in zip(decoder_source, decoder_target, decoder_weights_normalized):
        ax.plot([0.5, 1], [hidden_positions[src], output_positions[tgt]], color='gray', alpha=weight)

    # Add labels and legend
    ax.set_title(title)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["Input Layer", "Hidden Layer", "Output Layer"])
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    plt.show()