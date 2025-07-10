from sklearn.linear_model import LogisticRegression

from src.evaluate import evaluate_model
from src.utils.eda_utils import plot_confusion_matrix
import os
from src.utils.utils import save_model


def train_logistic_model(X_train_vec, y_train, X_val_vec, y_val):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = evaluate_model(model, X_val_vec, y_val, model_name="Logistic Regression", is_dl=False)

    plot_confusion_matrix(y_val, y_pred, labels=["Fake", "Real"], title="Logistic Regression Confusion Matrix", save_as="logistic_cm.png")

    # Save the model
    save_model(model, os.path.join("models", "logistic_model.pkl"))



    print("[INFO] Logistic Regression model trained and saved as 'logistic_model.pkl'.")

    return model