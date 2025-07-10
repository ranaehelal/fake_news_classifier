import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.eda_utils import plot_training_curves, plot_confusion_matrix
from src.utils import save_model

def build_lstm_model(vocab_size, embedding_dim=100, max_len=300,lstm_units=64):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    model.add(Dropout(.5))
    model.add(Bidirectional(LSTM(lstm_units)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # binary classification

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    print(f"[INFO] LSTM Model built with vocab size: {vocab_size}, embedding dim: {embedding_dim}, lstm units: {lstm_units}")

    return model

def train_lstm_model(X_train, y_train, X_val, y_val, tokenizer, max_len=300):
    vocab_size = len(tokenizer.word_index) + 1
    model = build_lstm_model(vocab_size=vocab_size, max_len=max_len)


    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )



    plot_training_curves(history,'lstm_training_curves.png')
    plot_confusion_matrix(
        y_val, model.predict(X_val).round(),
        labels=['Fake', 'Real'],
    title='LSTM Model Confusion Matrix',
        save_as='lstm_confusion_matrix.png'
    )
    save_model(model, os.path.join("models", "lstm_model.h5"))

    y_val_pred = model.predict(X_val).round()

    # Metrics
    acc = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    # Create results file if not exists
    if not os.path.exists("experiments/results.csv"):
        with open("experiments/results.csv", "w") as f:
            f.write("Model,Accuracy,Precision,Recall,F1\n")

    # Append results
    with open("experiments/results.csv", "a") as f:
        f.write(f"LSTM,{acc:.4f},{precision:.4f},{recall:.4f},{f1:.4f}\n")


    print("[INFO] LSTM model trained and saved.")

    return model, history