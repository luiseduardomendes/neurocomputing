from model import DimensionalitySafeKernelRCE
from data_loader import load_and_process_data
from sklearn.model_selection import KFold
from utils import plot_confusion_matrix, plot_accuracy_curve, print_classification_report

import numpy as np

def main():
    print("Loading data...")
    X, y, original_dim = load_and_process_data("dataset")

    # Initialize 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    all_y_true = []
    all_y_pred = []

    print("Starting 5-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print("\nFold {}/5".format(fold + 1))
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Initialize model
        model = DimensionalitySafeKernelRCE(
            num_prototypes=30,
            kernel='rbf',
            gamma=0.03,
            learning_rate=0.022,
            momentum=0.98,
            max_epochs=200,
            activation_threshold=0.005
        )

        # Train
        print("Training...")
        model.fit(X_train, y_train)

        # Validate
        y_pred = model.predict(X_val)
        acc = np.mean(y_pred == y_val)
        fold_accuracies.append(acc)

        print("Fold accuracy: {:.2f}%".format(acc * 100))

        # Store predictions
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

    # Final evaluation
    print("\nCross-validation results:")
    print("Individual fold accuracies: {}".format(["{:.2f}%".format(acc * 100) for acc in fold_accuracies]))
    print("Mean accuracy: {:.2f}%".format(np.mean(fold_accuracies) * 100))
    print("Standard deviation: {:.2f}%".format(np.std(fold_accuracies) * 100))

    # Plot results
    plot_confusion_matrix(all_y_true, all_y_pred, class_names=["thumbs_up", "thumbs_down", "thumbs_left", "thumbs_right", "fist_closed"])
    plot_accuracy_curve(fold_accuracies)
    print_classification_report(all_y_true, all_y_pred)

if __name__ == "__main__":
    main()

