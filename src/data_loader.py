import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path="data/train.csv"):
    """
    Load MNIST digit dataset from CSV file.

    The dataset format:
    - Each row represents a 28x28 grayscale image (flattened into 784 features)
    - "label" column contains the digit class (0–9)

    Returns:
    - x_train, x_test: pixel features
    - y_train, y_test: corresponding labels
    """

    df = pd.read_csv(path)

    # Separate features (pixel values) and target label
    x_raw = df.drop("label", axis=1)
    y = df["label"]

    # Split dataset into training and testing sets
    # 20% test split ensures fair evaluation of model generalization
    return train_test_split(
        x_raw,
        y,
        test_size=0.2,
        random_state=42
    )