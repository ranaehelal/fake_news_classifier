import pandas as pd
import os
def load_kaggle_data():
    fake_path = "data/raw/kaggle/Fake.csv"
    real_path = "data/raw/kaggle/True.csv"

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    fake_df['label'] ="fake"
    real_df['label'] ="real"

    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df[['title', 'text', 'subject', 'label']]

    return df
def load_fakenewsnet_data():
    base_path = "data/raw/fakenewsnet"

    #Gossip news
    gossip_fake_df = pd.read_csv(os.path.join(base_path, "gossip", "gossipcop_fake.csv"))
    gossip_real_df =  pd.read_csv(os.path.join(base_path, "gossip", "gossipcop_real.csv"))

    gossip_fake_df['label'] ="fake"
    gossip_real_df['label'] ="real"

    gossip_news= pd.concat([gossip_fake_df, gossip_real_df], ignore_index=True)
    gossip_news['subject'] = 'Gossip'

    # Political news
    political_fake_df = pd.read_csv(os.path.join(base_path, "political", "politifact_fake.csv"))
    political_real_df = pd.read_csv(os.path.join(base_path, "political", "politifact_real.csv"))

    political_fake_df['label'] ="fake"
    political_real_df['label'] ="real"

    political_news= pd.concat([political_fake_df, political_real_df], ignore_index=True)
    political_news['subject'] = 'Politics'

    df = pd.concat([gossip_news, political_news], ignore_index=True)
    df['text'] = df['title']
    df = df[['title', 'text', 'subject', 'label']]
    return df

def load_combined_data():
    kaggle = load_kaggle_data()
    fnn = load_fakenewsnet_data()
    return pd.concat([kaggle, fnn], ignore_index=True)


