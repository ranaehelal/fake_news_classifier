import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(model, X_val, y_val, model_name="Model", save_path="experiments/results.csv", is_dl=False):
    # Predict
    if is_dl:
        y_pred = model.predict(X_val).round()
    else:
        y_pred = model.predict(X_val)

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f"[{model_name}] Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write("Model,Accuracy,Precision,Recall,F1\n")

    with open(save_path, "a") as f:
        f.write(f"{model_name},{acc:.4f},{precision:.4f},{recall:.4f},{f1:.4f}\n")

    print('[INFO] Evaluation results saved to:', save_path)

    return y_pred


