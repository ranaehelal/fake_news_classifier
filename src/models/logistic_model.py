from sklearn.linear_model import LogisticRegression

from src.eda_utils import plot_confusion_matrix
import os
from src.utils import save_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def train_logistic_model(X_train_vec, y_train, X_val_vec, y_val):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_val_vec)

    plot_confusion_matrix(y_val, y_pred, labels=["Fake", "Real"], title="Logistic Regression Confusion Matrix", save_as="logistic_cm.png")

    # Save the model
    save_model(model, os.path.join("models", "logistic_model.h5"))

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Create results file if not exists
    if not os.path.exists("experiments/results.csv"):
        with open("experiments/results.csv", "w") as f:
            f.write("Model,Accuracy,Precision,Recall,F1\n")

    # Append results
    with open("experiments/results.csv", "a") as f:
        f.write(f"Logistic Regression,{acc:.4f},{precision:.4f},{recall:.4f},{f1:.4f}\n")


    print("[INFO] Logistic Regression model trained and saved as 'logistic_model.h5'.")

    return model