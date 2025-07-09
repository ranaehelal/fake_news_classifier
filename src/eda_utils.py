import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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