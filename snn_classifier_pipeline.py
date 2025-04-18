import numpy as np

class SNN_ClassifierPipeline:
    """
    Combines an SNN Autoencoder with a downstream classifier (RCE or RBF).
    - Trains the SNN unsupervised.
    - Extracts features (from encoder layer).
    - Trains the classifier supervised on extracted features.
    """

    def __init__(self, snn_model, classifier):
        self.snn = snn_model
        self.classifier = classifier
        self._is_trained = False

    def train(self, X, y, snn_epochs=10):
        """Train the pipeline: SNN autoencoder + classifier."""
        print("🔁 Starting unsupervised SNN training...")
        self.snn.train(X, epochs=snn_epochs)

        print("📦 Extracting latent features...")
        features = self.snn.extract_features(X)

        print("🧠 Training classifier...")
        self.classifier.fit(features, y)

        self._is_trained = True
        print("✅ Pipeline training complete.")

    def predict(self, X):
        """Predict using trained pipeline."""
        if not self._is_trained:
            raise RuntimeError("Pipeline not trained yet")

        print("📦 Extracting features for prediction...")
        features = self.snn.extract_features(X)

        print("🔮 Running classifier prediction...")
        return self.classifier.predict(features)
