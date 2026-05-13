import matplotlib.pyplot as plt
import numpy as np
import os


def plot_accuracy_bar(acc_lr, acc_cnn, save_path="outputs"):
    """
    Compare model accuracy between Logistic Regression and CNN.

    This visualization helps highlight the performance gap between:
    - Classical ML model (Logistic Regression)
    - Deep Learning model (CNN)
    """

    os.makedirs(save_path, exist_ok=True)

    models = ["Logistic Regression", "CNN"]
    accs = [acc_lr, acc_cnn]

    plt.figure(figsize=(6, 4))
    plt.bar(models, accs)

    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{save_path}/accuracy_comparison.png")
    plt.close()


def plot_misclassified(x_test_raw, y_true, y_pred, save_path="outputs", model_name="model"):
    """
    Visualize misclassified samples.

    This helps understand failure cases of the model by showing:
    - What digits are being confused
    - Patterns of misclassification
    """

    # Convert to numpy for safe indexing
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    os.makedirs(save_path, exist_ok=True)

    # Get misclassified indices
    wrong_idx = np.where(y_true != y_pred)[0][:10]

    plt.figure(figsize=(10, 5))

    for i, idx in enumerate(wrong_idx):
        ax = plt.subplot(2, 5, i + 1)

        # Safe conversion (assume DataFrame input)
        img = x_test_raw.iloc[idx].values.reshape(28, 28)

        ax.imshow(img, cmap="gray")

        ax.set_title(f"P:{y_pred[idx]} / T:{y_true[idx]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_path}/misclassified_{model_name}.png")
    plt.close()