import sys
import  os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import load_kaggle_data
from src.features import prepare_tokenizer_and_sequences
from src.models.logistic_model import train_logistic_model
from src.models.lstm_model import train_lstm_model
from src.preprocessing import remove_duplicates_and_missing, clean_for_ml, clean_for_dl
import tqdm

from src.utils.utils import save_cleaned_data, split_data, save_pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = info, 2 = warnings, 3 = errors


def run_pipeline(model: str):

    # load data
    df_raw =load_kaggle_data()
    df_cleaned = remove_duplicates_and_missing(df_raw)

    if model in ['logistic_regression', 'svm', 'random_forest']:
        # clean data
        df_cleaned['clean_text'] = df_cleaned['text'].apply(clean_for_ml)
        save_cleaned_data(df_cleaned, "kaggle_clean_ml.csv")

        # label encoding
        df_cleaned['label'] = df_cleaned['label'].map({'fake': 0, 'real': 1})

        df_cleaned = df_cleaned.dropna(subset=['clean_text'])
        y = df_cleaned['label']

        # TF-IDF

        tfidf = TfidfVectorizer(max_features=10000)
        X_tfidf = tfidf.fit_transform(df_cleaned['clean_text'])
        save_pickle(tfidf, "models/tfidf_vectorizer.pkl")

        # split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_tfidf, y)


        if model == "logistic":
            model = train_logistic_model(X_train, y_train, X_val, y_val)

    elif model in ['lstm']:
        # clean data
        df_cleaned['clean_text'] = df_cleaned['text'].apply(clean_for_dl)
        save_cleaned_data(df_cleaned, "kaggle_clean_dl.csv")

        # label encoding
        df_cleaned['label'] = df_cleaned['label'].map({'fake': 0, 'real': 1})
        df_cleaned = df_cleaned.dropna(subset=["clean_text"])
        y = df_cleaned['label']

        # Tokenizer
        texts = df_cleaned["clean_text"].astype(str).tolist()
        tokenizer, padded_sequences = prepare_tokenizer_and_sequences(texts, max_vocab=10000, max_len=300)

        # save token
        save_pickle(tokenizer, "models/tokenizer.pkl")
        save_pickle(padded_sequences, "models/padded_sequences.pkl")

        #Split
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(padded_sequences, y)

        model, history = train_lstm_model(X_train, y_train, X_val, y_val, tokenizer)

    print(f"[INFO] Finished training and evaluation for {model.upper()} model.\n")




if __name__ == "__main__":
    run_pipeline("logistic_regression")








