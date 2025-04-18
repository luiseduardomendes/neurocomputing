import os
import time
import numpy as np
import argparse
from snn_autoencoder import SNN_Autoencoder
from rce_model import RCEModel
from rbf_model import RBFModel
from snn_classifier_pipeline import SNN_ClassifierPipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from utils import plot_confusion_matrix, print_classification_report

def parse_arguments():
    """
    Parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--hidden_size", type=int, default=3, help="Size of the hidden layer")
    parser.add_argument("--sim_time", type=int, default=100, help="Simulation time for SNN")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for SNN")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter for RCE")
    parser.add_argument("--rce_learning_rate", type=float, default=0.5, help="Learning rate for RCE")
    parser.add_argument("--rbf_gamma", type=float, default=0.5, help="Gamma parameter for RBF")
    return vars(parser.parse_args())

def save_run_results(params, results, folder="runs"):
    """
    Save the parameters and results of the current run to a unique folder.
    :param params: Dictionary containing the parameters of the run.
    :param results: Dictionary containing the results of the run.
    :param folder: Base folder to save the results.
    """
    # Create a unique folder for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_folder = os.path.join(folder, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    # Save parameters
    params_file = os.path.join(run_folder, "parameters.txt")
    with open(params_file, "w") as f:
        f.write("=== Run Parameters ===\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    # Save results
    results_file = os.path.join(run_folder, "results.txt")
    with open(results_file, "w") as f:
        f.write("=== Run Results ===\n")
        for dataset_name, dataset_results in results.items():
            f.write(f"\n=== Dataset: {dataset_name.upper()} ===\n")
            for model_name, model_results in dataset_results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"Mean Accuracy: {model_results['mean_accuracy'] * 100:.2f}%\n")
                f.write(f"Standard Deviation: {model_results['std_accuracy'] * 100:.2f}%\n")

    print(f"Results saved to {run_folder}")

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
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y, target_names

def train_and_evaluate(X, y, target_names, dataset_name, n_splits=5, params=None):
    """Trains and evaluates the SNN pipeline with RCE and RBF classifiers."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {"RCE": {"fold_accuracies": []}, "RBF": {"fold_accuracies": []}}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        epochs = params["epochs"]  # Number of epochs for SNN training

        # Normalize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # === SNN Autoencoder + RCE Classifier ===
        print("\n🔁 Training SNN Autoencoder + RCE Classifier Pipeline...")
        snn = SNN_Autoencoder(input_size=X.shape[1], hidden_size=params["hidden_size"], sim_time=params["sim_time"], learning_rate=params["learning_rate"])
        rce = RCEModel(gamma=params["gamma"], learning_rate=params["rce_learning_rate"])
        pipeline_rce = SNN_ClassifierPipeline(snn_model=snn, classifier=rce)
        pipeline_rce.train(X_train, y_train, snn_epochs=epochs)
        y_pred_rce = pipeline_rce.predict(X_val)

        # Evaluate RCE pipeline
        acc_rce = np.mean(y_pred_rce == y_val)
        results["RCE"]["fold_accuracies"].append(acc_rce)

        # === SNN Autoencoder + RBF Classifier ===
        print("\n🔁 Training SNN Autoencoder + RBF Classifier Pipeline...")
        rbf = RBFModel(gamma=params["rbf_gamma"])
        pipeline_rbf = SNN_ClassifierPipeline(snn_model=snn, classifier=rbf)
        pipeline_rbf.train(X_train, y_train, snn_epochs=epochs)
        y_pred_rbf = pipeline_rbf.predict(X_val)

        # Evaluate RBF pipeline
        acc_rbf = np.mean(y_pred_rbf == y_val)
        results["RBF"]["fold_accuracies"].append(acc_rbf)

    # Calculate mean and std for each model
    for model_name in results.keys():
        results[model_name]["mean_accuracy"] = np.mean(results[model_name]["fold_accuracies"])
        results[model_name]["std_accuracy"] = np.std(results[model_name]["fold_accuracies"])

    return results

def main():
    print("Available datasets:")
    print("1. Hand gesture dataset")
    print("2. Wine dataset")
    print("3. Iris dataset")
    print("4. All datasets")

    #choice = input("\nSelect dataset (1-4): ")
    choice = "4"  # For testing purposes, we can set the choice directly

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

    params = parse_arguments()

    all_results = {}

    for dataset_name in datasets:
        print(f"\n=== Processing Dataset: {dataset_name.upper()} ===")
        scale = dataset_name != "hand"  # No StandardScaler for spike data
        X, y, target_names = load_and_preprocess_data(dataset_name, scale=scale)

        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
        results = train_and_evaluate(X, y, target_names, dataset_name, n_splits=5, params=params)
        all_results[dataset_name] = results

    save_run_results(params, all_results)

if __name__ == "__main__":
    main()