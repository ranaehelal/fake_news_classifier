import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


MODELS_DIR = "models"
PLOTS_DIR = "visualizations"
def generate_wordcloud(df, label, source=None, base_dir="visualizations/wordclouds"):
    """
    Generates a WordCloud for the dataframe and saves it a

    Args:
        df (pd.DataFrame): The dataframe containing the text and label columns.
        label (str): "fake" or "real".
        base_dir (str):  where the image will be saved.
        source: "kaggle", "fnn"
    """
    save_dir = os.path.join(base_dir, source if source else "")
    os.makedirs(save_dir, exist_ok=True)

    text_series = df[df['label'] == label]['text'].dropna().astype(str)
    text = ' '.join(text_series)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    save_path = os.path.join(save_dir, f"{label}_wc.png")
    plt.savefig(save_path)
    plt.close()

def plot_top_words (word_df,x_label ='Frequency', y_label='Words', title='Top Words'):
    word_df.columns = ['word', 'freq']

    plt.figure(figsize=(10, 6))

    # Create the plot
    ax = sns.barplot(
        data=word_df,
        x='freq',
        y='word',

    )


    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    plt.show()


def get_top_n_words(data, n=None):
    vec = CountVectorizer(stop_words='english').fit(data)
    bag = vec.transform(data)
    sum_words = bag.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]


def plot_confusion_matrix(y_true, y_pred, labels, title, save_as):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    os.makedirs(PLOTS_DIR + "/confusion_matrices", exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/confusion_matrices/{save_as}", bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {PLOTS_DIR}/confusion_matrices/{save_as}")

def plot_training_curves(history, save_as):
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Val")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label="Train")
    plt.plot(history.history['val_accuracy'], label="Val")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    os.makedirs(PLOTS_DIR + "/training_curves", exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/training_curves/{save_as}", bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {PLOTS_DIR}/training_curves/{save_as}")