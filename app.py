
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_model.h5")  # Ensure model is saved as 'cat_dog_model.h5'

model = load_model()

# Streamlit UI
st.title("Cat vs Dog Classifier")
st.write("Upload an image to classify it as a cat or a dog.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = img.resize((256, 256))  # Ensure size matches training input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img_array)
    class_name = "Dog" if prediction[0][0] > 0.5 else "Cat"
    confidence = prediction[0][0] if class_name == "Dog" else 1 - prediction[0][0]

    st.write(f"Prediction: {class_name} ({confidence:.2f} confidence)")
