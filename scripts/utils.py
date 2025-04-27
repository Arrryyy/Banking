import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f" Model saved at {filepath}")

def print_classification_metrics(y_true, y_pred):
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')  # <-- ensures clean saving
        print(f" Confusion matrix saved at {save_path}")
    
    plt.close()  