from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


def preprocess_lr(x_train_raw, x_test_raw):
    """
    Preprocessing pipeline for Logistic Regression.

    Steps:
    1. Standardize pixel values (zero mean, unit variance)
    2. Reduce dimensionality using PCA (retain 95% variance)

    Returns:
    - x_train_pca, x_test_pca: transformed features
    - scaler: fitted StandardScaler (for inference reuse)
    - pca: fitted PCA model (for inference reuse)
    """

    # Standardization is important for distance-based optimization convergence
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    # PCA reduces noise and speeds up Logistic Regression convergence
    pca = PCA(n_components=0.95)

    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    return x_train_pca, x_test_pca, scaler, pca


def preprocess_cnn(x_train_raw, x_test_raw):
    """
    Preprocessing pipeline for CNN.

    Steps:
    1. Normalize pixel values to [0, 1]
    2. Reshape into image format (28x28x1)

    Returns:
    - x_train_cnn, x_test_cnn: reshaped tensors for CNN input
    """

    # Convert to numpy for safety
    x_train = np.asarray(x_train_raw)
    x_test = np.asarray(x_test_raw)

    # Normalize pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape for CNN input (batch, height, width, channel)
    x_train_cnn = x_train.reshape(-1, 28, 28, 1)
    x_test_cnn = x_test.reshape(-1, 28, 28, 1)

    return x_train_cnn, x_test_cnn