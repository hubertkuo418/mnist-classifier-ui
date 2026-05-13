from sklearn.linear_model import LogisticRegression


def train_lr(x_train_pca, y_train):
    """
    Train a Logistic Regression model as a baseline classifier for MNIST digits.

    This model serves as a classical machine learning baseline to compare
    against the CNN model.

    PCA-reduced features are used as input to improve efficiency and convergence.

    Returns:
    - model: trained Logistic Regression model
    """

    # Logistic Regression is used as a baseline linear classifier
    # It is efficient on PCA-reduced data and provides a strong reference point
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        n_jobs=-1,
        random_state=42
    )

    # Train model on PCA-transformed features
    model.fit(x_train_pca, y_train)

    print("Training done")

    return model