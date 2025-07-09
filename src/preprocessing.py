import pandas as pd
import re
import spacy
import string

nlp = spacy.load('en_core_web_sm')

def remove_duplicates_and_missing(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].astype(str).str.strip() != ""]
    return df

def base_clean(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove links

    text = re.sub(r"\d+", " <NUM> ", text)

    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punc


    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_for_ml(text):
    """Cleaner for ML models """
    text = base_clean(text )
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(words)


def clean_for_dl(text):
    """Cleaner for deep models """
    return base_clean(text )