import matplotlib.pyplot as plt
import numpy as np
import os


def plot_accuracy_bar(acc_lr, acc_cnn, save_path="outputs"):

    os.makedirs(save_path, exist_ok=True)

    models = ["Logistic Regression", "CNN"]
    accs = [acc_lr, acc_cnn]

    plt.figure(figsize=(6, 4))
    plt.bar(models, accs)

    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")

    plt.ylim(0, 1)

    plt.savefig(f"{save_path}/accuracy_comparison.png")
    plt.close()


def plot_misclassified(x_test_raw, y_true, y_pred, save_path="outputs", model_name="model"):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    os.makedirs(save_path, exist_ok=True)

    wrong_idx = np.where(y_true != y_pred)[0][:10]

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):

        idx = wrong_idx[i]

        img = x_test_raw.iloc[idx].values.reshape(28, 28)

        ax.imshow(img, cmap="gray")

        ax.set_title(f"P:{y_pred[idx]} / T:{y_true[idx]}")

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(f"{save_path}/misclassified_{model_name}.png")
    plt.close()