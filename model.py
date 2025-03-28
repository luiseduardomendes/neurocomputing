import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

class DimensionalitySafeKernelRCE:
    """Robust classification model using kernel RCE with LDA feature transformation."""
    
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

    def fit(self, X, y):
        """Trains the model on given dataset."""
        X_scaled = self.scaler.fit_transform(X)
        self.lda.fit(X_scaled, y)
        X_lda = self.lda.transform(X_scaled)
        
        # Initialize prototypes
        self.prototypes_ = np.zeros((self.num_prototypes, X_lda.shape[1]))
        self.radii_ = np.zeros(self.num_prototypes)
        self.labels_ = np.zeros(self.num_prototypes, dtype=int)
        
        # Assign prototypes based on class distribution
        unique_classes = np.unique(y)
        for idx, cls in enumerate(unique_classes):
            class_samples = X_lda[y == cls]
            chosen = class_samples[np.random.choice(len(class_samples), self.num_prototypes // len(unique_classes), replace=True)]
            self.prototypes_[idx * (self.num_prototypes // len(unique_classes)):(idx+1) * (self.num_prototypes // len(unique_classes))] = chosen
            self.labels_[idx * (self.num_prototypes // len(unique_classes)):(idx+1) * (self.num_prototypes // len(unique_classes))] = cls

        self._is_fitted = True
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

