from sklearn.linear_model import LogisticRegression

from src.eda_utils import plot_confusion_matrix, plot_training_curves
import os
from src.utils import save_model


def train_logistic_model(X_train_vec, y_train, X_val_vec, y_val):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_val_vec)
    plot_training_curves(

    )
    plot_confusion_matrix(y_val, y_pred, labels=["Fake", "Real"], title="Logistic Regression Confusion Matrix", save_as="logistic_cm.png")

    # Save the model
    save_model(model, os.path.join("models", "logistic_model.h5"))

    print("[INFO] Logistic Regression model trained and saved as 'logistic_model.pkl'.")

    return model