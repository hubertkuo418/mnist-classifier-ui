from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import numpy as np


def evaluate_lr(model, x_test_pca, y_test, save_path="outputs"):

    os.makedirs(save_path, exist_ok=True)

    # Predict
    y_pred = model.predict(x_test_pca)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("LR Accuracy:", acc)

    # Confusion Matrix (explicit figure control)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)

    ax.set_title("Logistic Regression Confusion Matrix")

    plt.savefig(f"{save_path}/confusion_matrix_lr.png")
    plt.close()

    return y_pred, acc


def evaluate_cnn(model, x_test_cnn, y_test, save_path="outputs"):

    os.makedirs(save_path, exist_ok=True)

    # Predict (probabilities → class)
    y_pred_prob = model.predict(x_test_cnn)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("CNN Accuracy:", acc)

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)

    ax.set_title("CNN Confusion Matrix")

    plt.savefig(f"{save_path}/confusion_matrix_cnn.png")
    plt.close()

    return y_pred, acc