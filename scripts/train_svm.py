import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def plot_and_save_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" Confusion matrix saved to {save_path}")

def train_svm_model(processed_data_path, model_save_path, confusion_matrix_path):
    # Step 1: Load the processed data
    df = pd.read_csv(processed_data_path)

    # Step 2: Split into features and target
    X = df.drop('y', axis=1)
    y = df['y']

    # Step 3: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 4: Initialize and Train SVM
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)

    # Step 5: Save model and feature names
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model_data = {
        'model': svm_model,
        'features': list(X_train.columns)
    }
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(" SVM training complete and model saved!")

    # Step 6: Evaluate Model and Save Confusion Matrix
    y_pred = svm_model.predict(X_test)
    print("\n Classification Report:\n")
    print(classification_report(y_test, y_pred))
    plot_and_save_confusion_matrix(y_test, y_pred, labels=['No', 'Yes'], save_path=confusion_matrix_path)

if __name__ == "__main__":
    processed_data_path = "data/processed/bank_processed.csv"
    model_save_path = "models/svm_model.pkl"
    confusion_matrix_path = "images/confusion_matrix.png"
    train_svm_model(processed_data_path, model_save_path, confusion_matrix_path)