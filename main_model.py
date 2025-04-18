from data_loader import load_and_process_data
from sklearn.model_selection import KFold
from utils import plot_confusion_matrix, plot_accuracy_curve, print_classification_report
from brian2 import ms
from snn_autoencoder import SNN_Autoencoder
from rce_model import RCEModel 
from rbf_model import RBFModel

import numpy as np

def load_and_preprocess_data(dataset_name):
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
        X, y, original_dim = load_and_process_data("dataset")
        target_names = ["thumbs_up", "thumbs_down", "thumbs_left", "thumbs_right", "fist_closed"]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return X, y, target_names

def train_and_evaluate(X, y, target_names, dataset_name, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rce_fold_accuracies = []
    rbf_fold_accuracies = []

    rce_all_y_true, rce_all_y_pred = [], []
    rbf_all_y_true, rbf_all_y_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # === Train SNN Autoencoder ===
        print("\nTraining SNN Autoencoder...")
        autoencoder = SNN_Autoencoder(input_size=X_train.shape[1], hidden_size=10, sim_time=100*ms, learning_rate=0.01)
        autoencoder.train(X_train, epochs=10)

        # Extract features
        print("Extracting features...")
        X_train_features = autoencoder.extract_features(X_train)
        X_val_features = autoencoder.extract_features(X_val)

        # === RCE Classifier ===
        print("\nTraining RCE Classifier...")
        rce_model = RCEModel(num_prototypes=5, gamma=1.0, learning_rate=0.1, max_epochs=100)
        rce_model.fit(X_train_features, y_train)
        rce_pred = rce_model.predict(X_val_features)
        rce_acc = np.mean(rce_pred == y_val)
        print(f"RCE Classifier Accuracy: {rce_acc * 100:.2f}%")
        rce_fold_accuracies.append(rce_acc)
        rce_all_y_true.extend(y_val)
        rce_all_y_pred.extend(rce_pred)

        # === RBF Classifier ===
        print("\nTraining RBF Classifier...")
        rbf_model = RBFModel(gamma=1.0)
        rbf_model.fit(X_train_features, y_train)
        rbf_pred = rbf_model.predict(X_val_features)
        rbf_acc = np.mean(rbf_pred == y_val)
        print(f"RBF Classifier Accuracy: {rbf_acc * 100:.2f}%")
        rbf_fold_accuracies.append(rbf_acc)
        rbf_all_y_true.extend(y_val)
        rbf_all_y_pred.extend(rbf_pred)

    # Print final results
    print("\n=== Final Results ===")
    print("\nRCE Classifier:")
    print(f"Mean Accuracy: {np.mean(rce_fold_accuracies) * 100:.2f}%")
    print(f"Standard Deviation: {np.std(rce_fold_accuracies) * 100:.2f}%")

    print("\nRBF Classifier:")
    print(f"Mean Accuracy: {np.mean(rbf_fold_accuracies) * 100:.2f}%")
    print(f"Standard Deviation: {np.std(rbf_fold_accuracies) * 100:.2f}%")

    # Plot accuracy curves
    print("\nPlotting accuracy curves...")
    plot_accuracy_curve(rce_fold_accuracies)
    plot_accuracy_curve(rbf_fold_accuracies)

    # Plot confusion matrices
    print("\nPlotting confusion matrices...")
    plot_confusion_matrix(rce_all_y_true, rce_all_y_pred, target_names, title="RCE Confusion Matrix")
    plot_confusion_matrix(rbf_all_y_true, rbf_all_y_pred, target_names, title="RBF Confusion Matrix")

    # Print classification reports
    print("\nClassification Reports:")
    print("\nRCE Classifier:")
    print_classification_report(rce_all_y_true, rce_all_y_pred)
    print("\nRBF Classifier:")
    print_classification_report(rbf_all_y_true, rbf_all_y_pred)

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
        X, y, target_names = load_and_preprocess_data(dataset_name)

        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
        train_and_evaluate(X, y, target_names, dataset_name, n_splits=5)


if __name__ == "__main__":
    main()