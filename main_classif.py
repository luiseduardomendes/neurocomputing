import numpy as np
from snn_autoencoder import SNN_Autoencoder
from rce_model import RCEModel
from rbf_model import RBFModel
from snn_classifier_pipeline import SNN_ClassifierPipeline
from sklearn.model_selection import KFold
from utils import plot_confusion_matrix, print_classification_report

def load_and_preprocess_data(dataset_name, scale=True):
    """Loads and preprocesses the dataset."""
    if dataset_name == 'wine':
        from sklearn.datasets import load_wine
        data = load_wine()
        X, y = data.data, data.target
        target_names = data.target_names
    elif dataset_name == 'iris':
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data, data.target
        target_names = data.target_names
    elif dataset_name == 'hand':
        from data_loader import load_and_process_data
        X, y, original_dim = load_and_process_data("dataset")
        target_names = ["thumbs_up", "thumbs_down", "thumbs_left", "thumbs_right", "fist_closed"]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y, target_names

def train_and_evaluate(X, y, target_names, dataset_name, n_splits=5):
    """Trains and evaluates the SNN pipeline with RCE and RBF classifiers."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # === SNN Autoencoder + RCE Classifier ===
        print("\n🔁 Training SNN Autoencoder + RCE Classifier Pipeline...")
        snn = SNN_Autoencoder(input_size=X.shape[1], hidden_size=3, sim_time=100, learning_rate=0.1)
        rce = RCEModel(gamma=1.0, learning_rate=0.5)
        pipeline_rce = SNN_ClassifierPipeline(snn_model=snn, classifier=rce)
        pipeline_rce.train(X_train, y_train, snn_epochs=25)
        y_pred_rce = pipeline_rce.predict(X_val)

        # Evaluate RCE pipeline
        print("\n📊 Evaluating SNN + RCE Classifier...")
        print_classification_report(y_val, y_pred_rce)

        # === SNN Autoencoder + RBF Classifier ===
        print("\n🔁 Training SNN Autoencoder + RBF Classifier Pipeline...")
        rbf = RBFModel(gamma=0.5)
        pipeline_rbf = SNN_ClassifierPipeline(snn_model=snn, classifier=rbf)
        pipeline_rbf.train(X_train, y_train, snn_epochs=25)
        y_pred_rbf = pipeline_rbf.predict(X_val)

        # Evaluate RBF pipeline
        print("\n📊 Evaluating SNN + RBF Classifier...")
        print_classification_report(y_val, y_pred_rbf)

def main():
    print("Available datasets:")
    print("1. Hand gesture dataset")
    print("2. Wine dataset")
    print("3. Iris dataset")
    print("4. All datasets")

    choice = input("\nSelect dataset (1-4): ")

    if choice == "1":
        datasets = ['hand']
    elif choice == "2":
        datasets = ['wine']
    elif choice == "3":
        datasets = ['iris']
    elif choice == "4":
        datasets = ['hand', 'wine', 'iris']
    else:
        print("Invalid choice.")
        return

    for dataset_name in datasets:
        print(f"\n=== Processing Dataset: {dataset_name.upper()} ===")
        scale = dataset_name != "hand"  # No StandardScaler for spike data
        X, y, target_names = load_and_preprocess_data(dataset_name, scale=scale)

        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
        train_and_evaluate(X, y, target_names, dataset_name, n_splits=5)

if __name__ == "__main__":
    main()