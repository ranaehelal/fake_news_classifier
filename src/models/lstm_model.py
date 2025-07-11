import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.evaluate import evaluate_model
from src.utils.eda_utils import plot_training_curves, plot_confusion_matrix
from src.utils.utils import save_model

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

    y_val_pred = evaluate_model(model, X_val, y_val, model_name="LSTM", is_dl=True)

    plot_confusion_matrix(
        y_val, y_val_pred,
        labels=['Fake', 'Real'],
    title='LSTM Model Confusion Matrix',
        save_as='lstm_confusion_matrix.png'
    )

    model.save("lstm_model.h5")


    print("[INFO] LSTM model trained and saved.")

    return model, history