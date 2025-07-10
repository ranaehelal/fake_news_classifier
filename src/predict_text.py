import sys
import  os

import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = info, 2 = warnings, 3 = errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from keras.src.utils import pad_sequences
from tensorflow.python.keras.saving.save import load_model

from src.preprocessing import clean_for_ml, clean_for_dl
from src.utils.utils import load_pickle, load_trained_model


def predict_text(text,model_name):
    print("[INFO] Predicting text for '{}'...".format(model_name))
    if model_name.lower() == "logistic":
        # clean text
        clean_text= clean_for_ml(text)

        # load the vectorizer
        vectorizer = load_pickle("models/tfidf_vectorizer.pkl")

        #load the model
        model = joblib.load("models/logistic_model.pkl")

        # transform the text
        X_test = vectorizer.transform([clean_text])

        prediction = model.predict(X_test)

    elif model_name.lower()== "lstm":
        # clean text
        clean_text = clean_for_dl(text)

        # load the Tokenizer
        tokenizer = load_pickle("models/dl_tokenizer.pkl")

        # load the model
        model = load_trained_model("models/lstm_model.h5")

        # padding and sequences
        seq = tokenizer.texts_to_sequences([clean_text])
        X = pad_sequences(seq, maxlen=300)

        # predict
        prediction = int(model.predict(X).round()[0][0])

    else:
        raise print("Unsupported model type")


    label = "real" if prediction == 1 else "fake"
    print(f'{text}')
    print(f"Prediction: {label}")
    return label

if __name__ == "__main__":
    sample = "President announces a new economic plan that will boost the country's growth."
    predict_text(sample, model_name="logistic")

    sample2 = "Shocking! Celebrity found hiding alien tech in secret lab!"
    predict_text(sample2, model_name="LSTM")