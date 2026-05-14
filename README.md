# 🧠 Handwritten Digit Classification (MNIST)

Compare Machine Learning and Deep Learning models on MNIST handwritten digit recognition.

---

# 📌 Models

- Logistic Regression + PCA
- Convolutional Neural Network (CNN)
- Streamlit Drawing App

---

# 📊 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~91.6% |
| CNN | ~98.3% |

✔ CNN learns image features automatically  
✔ CNN significantly outperforms Logistic Regression

---

# 📁 Project Structure

```text
ML-MINI-PROJECT/
│
├── data/
├── notebooks/
├── outputs/
├── saved_model/
│   └── cnn_model.keras
├── src/
│   ├── preprocess.py
│   ├── train_lr.py
│   ├── train_cnn.py
│   ├── evaluate.py
│   └── visualize.py
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Training

```bash
python main.py
```

Outputs are saved in:

```text
outputs/
```

---

# 🎨 Run Streamlit App

Launch the interactive handwritten digit recognizer:

```bash
streamlit run app.py
```

Features:

- Draw digits with mouse
- CNN real-time prediction
- Uses trained `cnn_model.keras`

---

# ⚙️ Methodology

## Logistic Regression

- Standardization
- PCA
- Logistic Regression classifier

## CNN

- Normalize images
- Convolution + Pooling + Dense layers

---

# 📈 Conclusion

CNN achieves much higher accuracy because it automatically learns spatial image features.

---

# 👤 Author

Student

