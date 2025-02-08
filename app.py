import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Load the trained model
MODEL_PATH = "my_model1 .keras"
model = load_model("my_model1 .keras")

# Get input shape from model
input_shape = (222, 222)  # Updated based on model output shape

def preprocess_image(image):
    """Preprocesses the uploaded image to match model input."""
    image = image.resize(input_shape)  # Resize to model's input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Set Streamlit page layout and style
st.set_page_config(page_title="Waste Classification", layout="wide")
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 36px;
            color: #4CAF50;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>♻️ Waste Classification Using CNN</h1>", unsafe_allow_html=True)

st.sidebar.header("Upload Image")

# Upload image from file
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

col1, col2 = st.columns(2)

         # About section
st.markdown("---")
st.subheader("ℹ️ **About**")
st.markdown("""
1. This project uses a CNN model to classify waste into Organic or Recyclable.
2. Dataset Provider: Techsash (Kaggle)
3. Future Improvements: Classification based on non-recyclable plastic, recyclable plastic and 

             
Thanks for visiting!
""")

if uploaded_file:
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if st.button("🔍 Classify Image"):
            with st.spinner("Processing Image..."):
                time.sleep(2)  # Simulate processing time
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                class_index = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction) * 100
                
                # Customize class labels based on your model
                class_labels = ["Organic Waste", "Recyclable Waste", "E-Waste", "Hazardous Waste"]
                predicted_class = class_labels[class_index] if class_index < len(class_labels) else "Unknown"
                
                st.success(f"Prediction: **{predicted_class}** with **{confidence:.2f}%** confidence.")
                
                # Progress bar for effect
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                st.balloons()


       

# Footer
# Footer
st.markdown("""
    <div class="footer">
        <p>Made with ❤️ by Prathamesh | ♻️ Promoting Sustainable Waste Management</p>
    </div>
""", unsafe_allow_html=True)


