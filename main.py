from model import DimensionalitySafeKernelRCE, SpikingRBFClassifier
from data_loader import load_and_process_data
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import StandardScaler
from utils import plot_confusion_matrix, plot_accuracy_curve, print_classification_report

import numpy as np

def load_and_preprocess_data(dataset_name):
    # Load dataset
    if dataset_name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
    elif dataset_name == 'iris':
        data = load_iris()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
    elif dataset_name == 'hand':
        X, y, original_dim = load_and_process_data("dataset")
        target_names = ["thumbs_up", "thumbs_down", "thumbs_left", "thumbs_right", "fist_closed"]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Normalize data
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
    
    # Initialize models
    rce_model = DimensionalitySafeKernelRCE(
        num_prototypes=30,  # Number of prototypes
        kernel='rbf',       # Kernel type
        gamma=0.1,         # Kernel parameter
        learning_rate=0.01, # Learning rate
        momentum=0.9,      # Momentum parameter
        max_epochs=300,    # Maximum epochs
        activation_threshold=0.01  # Activation threshold
    )

    spiking_rbf_model = SpikingRBFClassifier(
        gamma=0.1  # Kernel parameter
    )
    
    # Store results
    rce_fold_accuracies = []
    spiking_rbf_fold_accuracies = []
    rce_all_y_true = []
    rce_all_y_pred = []
    spiking_rbf_all_y_true = []
    spiking_rbf_all_y_pred = []

    print(f"\nStarting {n_splits}-fold cross-validation for {dataset_name}...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Evaluate models
        rce_acc, rce_pred = evaluate_model(rce_model, X_train, X_val, y_train, y_val, "RCE")
        spiking_rbf_model.fit(X_train, y_train)
        spiking_rbf_pred = spiking_rbf_model.predict(X_val)
        spiking_rbf_acc = np.mean(spiking_rbf_pred == y_val)

        print(f"Spiking RBF accuracy: {spiking_rbf_acc * 100:.2f}%")

        # Store results
        rce_fold_accuracies.append(rce_acc)
        spiking_rbf_fold_accuracies.append(spiking_rbf_acc)
        rce_all_y_true.extend(y_val)
        rce_all_y_pred.extend(rce_pred)
        spiking_rbf_all_y_true.extend(y_val)
        spiking_rbf_all_y_pred.extend(spiking_rbf_pred)

    # Final evaluation
    print(f"\nCross-validation results for {dataset_name}:")
    print("\nRCE Model:")
    print("Individual fold accuracies: {}".format(["{:.2f}%".format(acc * 100) for acc in rce_fold_accuracies]))
    print("Mean accuracy: {:.2f}%".format(np.mean(rce_fold_accuracies) * 100))
    print("Standard deviation: {:.2f}%".format(np.std(rce_fold_accuracies) * 100))

    print("\nSpiking RBF Model:")
    print("Individual fold accuracies: {}".format(["{:.2f}%".format(acc * 100) for acc in spiking_rbf_fold_accuracies]))
    print("Mean accuracy: {:.2f}%".format(np.mean(spiking_rbf_fold_accuracies) * 100))
    print("Standard deviation: {:.2f}%".format(np.std(spiking_rbf_fold_accuracies) * 100))

    # Plot results for both models
    print(f"\nRCE Model Results for {dataset_name}:")
    plot_confusion_matrix(rce_all_y_true, rce_all_y_pred, class_names=target_names)
    plot_accuracy_curve(rce_fold_accuracies)
    print_classification_report(rce_all_y_true, rce_all_y_pred)

    print(f"\nSpiking RBF Model Results for {dataset_name}:")
    plot_confusion_matrix(spiking_rbf_all_y_true, spiking_rbf_all_y_pred, class_names=target_names)
    plot_accuracy_curve(spiking_rbf_fold_accuracies)
    print_classification_report(spiking_rbf_all_y_true, spiking_rbf_all_y_pred)

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
        print("Invalid choice. Please select 1-4.")
        return
    
    for dataset_name in datasets:
        print(f"\n=== Processing Dataset: {dataset_name.upper()} ===")
        
        # Load and normalize data
        X, y, target_names = load_and_preprocess_data(dataset_name)
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Train and evaluate models
        train_and_evaluate(X, y, target_names, dataset_name)

if __name__ == "__main__":
    main()

