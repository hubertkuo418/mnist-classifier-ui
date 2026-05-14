import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "saved_model/cnn_model.keras",
        compile=False
    )

model = load_model()

st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) and let CNN predict it!")

# =========================
# Canvas (drawing area)
# =========================
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# =========================
# Predict Button
# =========================
if st.button("Predict"):

    if canvas_result.image_data is not None:

        # Convert canvas to image
        img = canvas_result.image_data.astype(np.uint8)
        img = Image.fromarray(img).convert("L")

        # Resize to MNIST format
        img = img.resize((28, 28))

        # Normalize
        img = np.array(img) / 255.0

        # Reshape for CNN
        img = img.reshape(1, 28, 28, 1)

        # Prediction
        pred = model.predict(img)
        result = np.argmax(pred)
        confidence = np.max(pred)

        # Output
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.2%}")