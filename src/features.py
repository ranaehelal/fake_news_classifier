from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def prepare_tokenizer_and_sequences(texts, max_vocab=10000, max_len=300):
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    print(f"[INFO] Tokenizer and sequences prepared. Vocab size: {len(tokenizer.word_index)}")

    return tokenizer, padded
