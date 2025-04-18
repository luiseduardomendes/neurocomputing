from data_loader import load_and_process_data
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from utils import plot_confusion_matrix, plot_accuracy_curve, print_classification_report
from model import DimensionalitySafeKernelRCE, SpikingRBFClassifier
from brian2 import prefs
prefs.codegen.target = 'numpy'  # Use numpy for code generation in Brian2

import numpy as np

def load_and_preprocess_data(dataset_name, scale=True):
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
    
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y, target_names

def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    # Train model
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    # Validate
    y_pred = model.predict(X_val)
    acc = np.mean(y_pred == y_val)

    print(f"{model_name} accuracy: {acc * 100:.2f}%")
    return acc, y_pred

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

        # === Dimensionality-Safe Kernel RCE Classifier ===
        print("\nTraining Dimensionality-Safe Kernel RCE Classifier...")
        rce_model = DimensionalitySafeKernelRCE(
            num_prototypes=1000, 
            gamma=1.0, 
            learning_rate=0.5, 
            max_epochs=1000, 
            activation_threshold=0.9
        )
        rce_model.fit(X_train, y_train)
        rce_pred = rce_model.predict(X_val)
        rce_acc = np.mean(rce_pred == y_val)
        print(f"Dimensionality-Safe Kernel RCE Classifier Accuracy: {rce_acc * 100:.2f}%")
        rce_fold_accuracies.append(rce_acc)
        rce_all_y_true.extend(y_val)
        rce_all_y_pred.extend(rce_pred)

        # === Spiking RBF Classifier ===
        print("\nTraining Spiking RBF Classifier...")
        rbf_model = SpikingRBFClassifier(gamma=0.005)  # Try smaller or larger values
        rbf_model.fit(X_train, y_train)
        rbf_pred = rbf_model.predict(X_val)
        rbf_acc = np.mean(rbf_pred == y_val)
        print(f"Spiking RBF Classifier Accuracy: {rbf_acc * 100:.2f}%")
        rbf_fold_accuracies.append(rbf_acc)
        rbf_all_y_true.extend(y_val)
        rbf_all_y_pred.extend(rbf_pred)

    # Print final results
    print("\n=== Final Results ===")
    print("\nDimensionality-Safe Kernel RCE Classifier:")
    print(f"Mean Accuracy: {np.mean(rce_fold_accuracies) * 100:.2f}%")
    print(f"Standard Deviation: {np.std(rce_fold_accuracies) * 100:.2f}%")

    print("\nSpiking RBF Classifier:")
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
    print("\nDimensionality-Safe Kernel RCE Classifier:")
    print_classification_report(rce_all_y_true, rce_all_y_pred)
    print("\nSpiking RBF Classifier:")
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
        scale = dataset_name != "hand"  # No StandardScaler for spike data
        X, y, target_names = load_and_preprocess_data(dataset_name, scale=scale)

        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
        train_and_evaluate(X, y, target_names, dataset_name, n_splits=5)


if __name__ == "__main__":
    main()
