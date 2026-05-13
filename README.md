=========================================================
🧠 HANDWRITTEN DIGIT CLASSIFICATION (MNIST)
=========================================================

This project compares Machine Learning vs Deep Learning
on handwritten digit classification using the MNIST dataset.

Models:

Logistic Regression (with PCA)
Convolutional Neural Network (CNN)

Goal:
Understand performance differences between classical ML and deep learning.

=========================================================
📊 RESULTS
=========================================================
Model Method Accuracy

Logistic Regression PCA + Linear Model ~91.6%
CNN Convolutional Neural Network ~98.3%

✔ CNN significantly outperforms Logistic Regression
✔ Because CNN learns spatial image features automatically

=========================================================
🧠 KEY CONCEPTS
=========================================================
1. ML vs DL Pipeline Comparison
Classical ML: Feature engineering (PCA) → Model
Deep Learning: Raw image → Feature learning → Model
2. Feature Engineering vs Representation Learning
PCA = manual compression
CNN = automatic feature extraction
3. Evaluation Metrics
Accuracy
Confusion Matrix
Visual prediction comparison
=========================================================
📁 PROJECT STRUCTURE
=========================================================

ml-mini-project/
│
├── src/
│ preprocess.py → Data preprocessing (LR + CNN)
│ train_lr.py → Logistic Regression model
│ train_cnn.py → CNN model
│ evaluate.py → Evaluation + confusion matrix
│ visualize.py → Visualization tools
│
├── outputs/
│ confusion_matrix_lr.png
│ confusion_matrix_cnn.png
│ sample_predictions.png
│
├── main.py → Pipeline entry point
└── README.md

=========================================================
🚀 HOW TO RUN
=========================================================

Step 1: Install dependencies

pip install numpy pandas scikit-learn matplotlib tensorflow

Step 2: Run project

python main.py

=========================================================
📦 OUTPUT FILES
=========================================================

After execution, results will be saved in:

/outputs

Files:

confusion_matrix_lr.png
confusion_matrix_cnn.png
sample_predictions.png
=========================================================
⚙️ METHODOLOGY
=========================================================

Logistic Regression Pipeline:

Standardization
PCA (95% variance retained)
Logistic Regression classifier

CNN Pipeline:

Normalize pixel values (0–1)
Reshape to (28, 28, 1)
Convolutional Neural Network
=========================================================
📈 INSIGHTS
=========================================================

✔ Logistic Regression:

Strong baseline
Limited by linear decision boundaries

✔ CNN:

Learns spatial patterns
Much higher accuracy

✔ Conclusion:
CNN > traditional ML for image tasks

=========================================================
🧠 WHAT I LEARNED
=========================================================
End-to-end ML pipeline design
Difference between classical ML and deep learning
Model evaluation and comparison
Modular project structure (src / main / outputs)
Reproducible ML workflow
=========================================================
🔮 FUTURE IMPROVEMENTS
=========================================================
Hyperparameter tuning
Data augmentation
Add more models (SVM, Random Forest)
Deploy with Streamlit / Flask
=========================================================
👤 AUTHOR
=========================================================

Student project focused on Machine Learning & Computer Vision fundamentals.