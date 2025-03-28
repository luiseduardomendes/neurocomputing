import numpy as np
from sklearn.model_selection import KFold
from data_loader import load_and_process_data
from model import DimensionalitySafeKernelRCE

def train_and_evaluate(dataset_path, num_folds=5):
    """Loads data and performs cross-validation on the model."""
    try:
        print "Loading data..."
        X, y, _ = load_and_process_data(dataset_path)
        
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_accuracies = []

        print "Starting cross-validation..."
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print "\nFold", fold, "/", num_folds

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = DimensionalitySafeKernelRCE(num_prototypes=30, gamma=0.03, learning_rate=0.022, 
                                                momentum=0.98, max_epochs=200, activation_threshold=0.005)

            print "Training model..."
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            acc = np.mean(y_pred == y_val)
            fold_accuracies.append(acc)
            print "Fold accuracy: ", acc * 100, "%"

        print "\nCross-validation summary:")
        print "Individual fold accuracies:", [str(a*100)+"%"  for a in fold_accuracies]
        print "Mean accuracy: ", np.mean(fold_accuracies) * 100, "%"
        print "Standard deviation: ", np.std(fold_accuracies) * 100, "%"

    except Exception as e:
        print "Error:", str(e)

if __name__ == "__main__":
    train_and_evaluate("dataset")

