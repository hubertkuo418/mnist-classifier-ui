# =========================
# Logistic Regression Pipeline
# =========================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocess_lr(x_train_raw, x_test_raw):

    # Scaling
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    # PCA
    pca = PCA(n_components=0.95)

    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    return x_train_pca, x_test_pca, scaler, pca


# =========================
# CNN Pipeline
# =========================

def preprocess_cnn(x_train_raw, x_test_raw):

    # normalize (0~255 -> 0~1)
    x_train_cnn = x_train_raw.values / 255.0
    x_test_cnn = x_test_raw.values / 255.0

    # reshape
    x_train_cnn = x_train_cnn.reshape(-1, 28, 28, 1)
    x_test_cnn = x_test_cnn.reshape(-1, 28, 28, 1)

    return x_train_cnn, x_test_cnn