# =========================
# Main Pipeline
# =========================

from src.data_loader import load_data
from src.preprocess import preprocess_lr, preprocess_cnn
from src.train_lr import train_lr
from src.train_cnn import train_cnn
from src.evaluate import evaluate_lr, evaluate_cnn
from src.visualize import plot_accuracy_bar, plot_misclassified


def main():

    # =========================
    # 1. Load Data
    # =========================
    x_train_raw, x_test_raw, y_train, y_test = load_data()

    # =========================
    # 2. Preprocess
    # =========================

    # Logistic Regression pipeline
    x_train_pca, x_test_pca, scaler, pca = preprocess_lr(x_train_raw, x_test_raw)

    # CNN pipeline
    x_train_cnn, x_test_cnn = preprocess_cnn(x_train_raw, x_test_raw)

    # =========================
    # 3. Train Models
    # =========================

    model_lr = train_lr(x_train_pca, y_train)
    model_cnn, history = train_cnn(x_train_cnn, y_train)

    # =========================
    # 4. Evaluate Models
    # =========================

    y_pred_lr, acc_lr = evaluate_lr(model_lr, x_test_pca, y_test)

    y_pred_cnn, acc_cnn = evaluate_cnn(model_cnn, x_test_cnn, y_test)

    # =========================
    # 5. Visualization
    # =========================

    plot_accuracy_bar(acc_lr, acc_cnn)
    
    plot_misclassified(x_test_raw, y_test, y_pred_lr, save_path="outputs", model_name="LR")
    plot_misclassified(x_test_raw, y_test, y_pred_cnn, save_path="outputs", model_name="CNN")

    # =========================
    # 6. Summary
    # =========================

    print("\n=========================")
    print("Final Results")
    print("=========================")
    print(f"Logistic Regression (PCA): {acc_lr:.4f}")
    print(f"CNN: {acc_cnn:.4f}")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()