from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


def evaluate_lr(model, x_test_pca, y_test, save_path="outputs"):

    os.makedirs(save_path, exist_ok=True)

    y_pred_lr = model.predict(x_test_pca)

    acc = accuracy_score(y_test, y_pred_lr)
    print("LR Accuracy:", acc)

    # Confusion Matrix
    fig = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr)
    plt.title("Logistic Regression Confusion Matrix")

    plt.savefig(f"{save_path}/confusion_matrix_lr.png")
    plt.close()

    return y_pred_lr, acc


def evaluate_cnn(model, x_test_cnn, y_test, save_path="outputs"):

    os.makedirs(save_path, exist_ok=True)

    y_pred_cnn = model.predict(x_test_cnn)
    y_pred_cnn = y_pred_cnn.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred_cnn)
    print("CNN Accuracy:", acc)

    # Confusion Matrix
    fig = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_cnn)
    plt.title("CNN Confusion Matrix")

    plt.savefig(f"{save_path}/confusion_matrix_cnn.png")
    plt.close()

    return y_pred_cnn, acc