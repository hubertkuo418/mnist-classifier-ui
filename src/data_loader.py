import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path="data/train.csv"):
    df = pd.read_csv(path)

    x_raw = df.drop("label", axis=1) 
    y = df["label"]

    return train_test_split(
        x_raw, y,
        test_size=0.2,
        random_state=42
    ) 