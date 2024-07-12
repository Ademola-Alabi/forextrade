import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adamax
from PIL import Image
import os

# Load and compile the model
model_path = 'forex.h5'

def print_layer_configurations(model):
    for layer in model.layers:
        st.write(layer.get_config())

try:
    loaded_model = load_model(model_path, compile=False)
    print_layer_configurations(loaded_model)
    loaded_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    st.write("Model loaded and compiled successfully")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    loaded_model = None

# Function to preprocess the image
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize((400, 400))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# Function to predict the class of the image
def predict(image, model):
    img_array = preprocess_image(image)
    st.write(f"Preprocessed image shape: {img_array.shape}")
    prediction = model.predict(img_array)
    st.write(f"Raw model prediction: {prediction}")
    return prediction

# Streamlit interface
st.markdown("<h1 style='text-align: center; color: red;'>Welcome to the Forex Trading Signal Predictor!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: blue;'>Upload an image to get a prediction of BUY or SELL signals using our trained deep learning model.</p>", unsafe_allow_html=True)

# Add an image to show it is a prediction app for Forex Trading
image_path = "ForexModelimage.jpg"
if os.path.exists(image_path):
    st.image(image_path, caption="Forex Trading Signal Predictor", use_column_width=True)
else:
    st.warning(f"Image file '{image_path}' not found.")

# Upload images
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if loaded_model is not None:
        with st.spinner("Classifying..."):
            prediction = predict(image, loaded_model)
            
            # Assuming train_gen.class_indices 0 is BUY and 1 is SELL
            classes = ['BUY', 'SELL']
            buy_prob = prediction[0][0] * 100
            sell_prob = prediction[0][1] * 100
            predicted_class = classes[np.argmax(prediction)]
            
            st.markdown(f"<h2 style='text-align: center;'>Prediction: <span style='color: green;'>{predicted_class}</span></h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>Confidence: BUY {buy_prob:.2f}% | SELL {sell_prob:.2f}%</p>", unsafe_allow_html=True)
    else:
        st.error("Model could not be loaded. Please check the logs for more details.")

# Add some information about the app
st.sidebar.title("About")
st.sidebar.info("""
    This app uses a trained deep learning model to predict Forex trading signals.
    Upload an image to get a prediction of BUY or SELL.
""")
