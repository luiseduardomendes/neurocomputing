from model import DimensionalitySafeKernelRCE
from data_loader import load_and_process_data
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from utils import plot_confusion_matrix, plot_accuracy_curve, print_classification_report

import numpy as np

def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    # Modelltraining
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    # Validierung
    y_pred = model.predict(X_val)
    acc = np.mean(y_pred == y_val)

    print(f"{model_name} accuracy: {acc * 100:.2f}%")
    return acc, y_pred

def main():
    print("Loading data...")
    X, y, original_dim = load_and_process_data("dataset")

    # Kreuzvalidierung initialisieren
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Speicherung der Ergebnisse
    rce_fold_accuracies = []
    rbf_fold_accuracies = []
    rce_all_y_true = []
    rce_all_y_pred = []
    rbf_all_y_true = []
    rbf_all_y_pred = []

    print(f"Starting {n_splits}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        
        # Daten aufteilen
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Modelle initialisieren
        rce_model = DimensionalitySafeKernelRCE(
            num_prototypes=30,  # Anzahl der Prototypen
            kernel='rbf',       # Kernel-Typ
            gamma=0.1,         # Kernel-Parameter
            learning_rate=0.01, # Lernrate
            momentum=0.9,      # Momentum-Parameter
            max_epochs=300,    # Maximale Epochen
            activation_threshold=0.01  # Aktivierungsschwelle
        )

        rbf_model = SVC(
            kernel='rbf',      # Kernel-Typ
            gamma="scale",     # Kernel-Parameter
            C=0.6,            # Regularisierungsparameter
            class_weight='balanced',  # Klassenausgleich
            random_state=42    # Zufallszahlengenerator
        )

        # Modelle auswerten
        rce_acc, rce_pred = evaluate_model(rce_model, X_train, X_val, y_train, y_val, "RCE")
        rbf_acc, rbf_pred = evaluate_model(rbf_model, X_train, X_val, y_train, y_val, "RBF")

        # Ergebnisse speichern
        rce_fold_accuracies.append(rce_acc)
        rbf_fold_accuracies.append(rbf_acc)
        rce_all_y_true.extend(y_val)
        rce_all_y_pred.extend(rce_pred)
        rbf_all_y_true.extend(y_val)
        rbf_all_y_pred.extend(rbf_pred)

    # Endauswertung
    print("\nCross-validation results:")
    print("\nRCE Model:")
    print("Individual fold accuracies: {}".format(["{:.2f}%".format(acc * 100) for acc in rce_fold_accuracies]))
    print("Mean accuracy: {:.2f}%".format(np.mean(rce_fold_accuracies) * 100))
    print("Standard deviation: {:.2f}%".format(np.std(rce_fold_accuracies) * 100))

    print("\nRBF Model:")
    print("Individual fold accuracies: {}".format(["{:.2f}%".format(acc * 100) for acc in rbf_fold_accuracies]))
    print("Mean accuracy: {:.2f}%".format(np.mean(rbf_fold_accuracies) * 100))
    print("Standard deviation: {:.2f}%".format(np.std(rbf_fold_accuracies) * 100))

    
    # Ergebnisse für beide Modelle plotten
    class_names = ["thumbs_up", "thumbs_down", "thumbs_left", "thumbs_right", "fist_closed"]
    
    print("\nRCE Model Results:")
    plot_confusion_matrix(rce_all_y_true, rce_all_y_pred, class_names=class_names)
    plot_accuracy_curve(rce_fold_accuracies)
    print_classification_report(rce_all_y_true, rce_all_y_pred)

    print("\nRBF Model Results:")
    plot_confusion_matrix(rbf_all_y_true, rbf_all_y_pred, class_names=class_names)
    plot_accuracy_curve(rbf_fold_accuracies)
    print_classification_report(rbf_all_y_true, rbf_all_y_pred)

if __name__ == "__main__":
    main()

