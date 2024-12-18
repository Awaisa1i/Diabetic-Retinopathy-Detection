import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image

# Header
st.header('Diabetic Retinopathy Detection')

# Load the trained model
model = tf.keras.models.load_model('model/final_model.h5')

# Class names for prediction
class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

# Prediction function
def predict(img):
    # Convert image to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Resize the image to match the model input
    img_array = tf.image.resize(img_array, [255, 255])  # Adjust size as per your model input dimensions
    
    # Add batch dimension
    img_array = tf.expand_dims(img_array, 0)

    # Make predictions
    predictions = model.predict(img_array)

    # Get predicted class and confidence
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['jpeg', "jpg", "png"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Display file details
    st.write("File Uploaded Successfully!")

    # Open the uploaded file as an image
    image = Image.open(uploaded_file)
    st.image(image)
    
    
if st.button('Detect'):
    image = Image.open(uploaded_file)
    # Convert image to RGB (ensure compatibility)
    image = image.convert("RGB")
    
    # Predict using the uploaded image
    predicted, confidence = predict(image)
    
    # Display prediction results
    st.write(f'You are detected with: **{predicted}**')
    st.write(f'Confidence: **{confidence}%**')