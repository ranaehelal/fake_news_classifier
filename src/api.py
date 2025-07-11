# src/api.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.utils.utils import load_pickle
from src.preprocessing import clean_for_ml, clean_for_dl
from keras.src.utils import pad_sequences
from tensorflow.keras.models import load_model

app = FastAPI()

# load the vectorizer
tfidf_vectorizer = load_pickle("models/tfidf_vectorizer.pkl")

# load the model
logistic_model = joblib.load("models/logistic_model.pkl")

# load the Tokenizer
tokenizer = load_pickle("models/dl_tokenizer.pkl")

# load the model
lstm_model = load_model("models/lstm_model.h5")


# Define input format using Pydantic
class InputText(BaseModel):
    text: str
    model: str  # "logistic" or "lstm"


@app.post("/predict")
def predict(input_data: InputText):
    text = input_data.text
    model_name = input_data.model.lower()

    if model_name == "logistic":
        cleaned = clean_for_ml(text)
        X = tfidf_vectorizer.transform([cleaned])
        prediction = logistic_model.predict(X)[0]

    elif model_name == "lstm":
        cleaned = clean_for_dl(text)
        seq = tokenizer.texts_to_sequences([cleaned])
        X = pad_sequences(seq, maxlen=300)
        prediction = int(lstm_model.predict(X).round()[0][0])

    else:
        return {"error": "Invalid model name. Use 'logistic' or 'lstm'."}

    label = "real" if prediction == 1 else "fake"
    return {"prediction": label}
