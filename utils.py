import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """Plots the confusion matrix using matplotlib."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Plot the numbers in each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_accuracy_curve(accuracy_per_fold):
    """Plots accuracy across multiple folds."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(accuracy_per_fold) + 1), accuracy_per_fold, marker='o', linestyle='-')
    plt.title("Model Accuracy Over Folds")
    plt.xlabel("Fold Number")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, len(accuracy_per_fold) + 1))
    plt.ylim(0, 1)  # Accuracy is between 0 and 1
    plt.grid()
    plt.show()

def print_classification_report(y_true, y_pred):
    """Prints the classification report."""
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)

def plot_sample_images(X, y, num_samples=5):
    """Displays random sample images from the dataset."""
    if X.shape[0] < num_samples:
        num_samples = X.shape[0]

    indices = np.random.choice(len(X), num_samples, replace=False)
    plt.figure(figsize=(10, 2))
    
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[idx].reshape(32, 32), cmap='gray')  # Assumes images are 32x32
        plt.title("Label: {}".format(y[idx]))
        plt.axis('off')
    
    plt.show()

