import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("cat_dog_model.h5")

# Constants
IMG_SIZE = 224
LABELS = ["Cat", "Dog"]

# Title
st.title("🐾 Cat vs Dog Classifier")
st.markdown("Upload an image **or** take one with your **webcam**, and I'll tell you if it's a **Cat** or a **Dog**!")

# Image source options
option = st.radio("Choose input method:", ["📁 Upload from computer", "📸 Use webcam"])

image = None

# Handle image upload
if option == "📁 Upload from computer":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

# Handle webcam
elif option == "📸 Use webcam":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", width=300)

# Run prediction
if image is not None:
    # Preprocess image
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]
    label = LABELS[int(pred > 0.5)]
    confidence = pred if label == "cat" else 1 - pred

    # Show prediction
    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"### 🔍 Confidence: **{confidence * 100:.2f}%**")
