import streamlit as st
from fastai.vision.all import load_learner, PILImage
from PIL import Image
import gdown
import os

# Set the title of the app
st.title("Mongolian Foods Classifier 🥘")

# Download the model from Google Drive
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    # Google Drive file ID (replace with your file's ID)
    file_id = "1lujIhjfBh7LB0bNxvIrCiV96plymN01C"  # Replace with your file ID
    model_url = f"https://drive.google.com/file/d/1lujIhjfBh7LB0bNxvIrCiV96plymN01C/view?usp=drive_link"
    model_path = "Mongolian_foods_classifier.pkl"


    
    # Download the model if it doesn't already exist
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)

    return load_learner(model_path)

# Load the model
learn = load_model()

# Allow the user to upload an image
uploaded_file = st.file_uploader("Upload an image of Mongolian food", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to a format suitable for the model
    img = PILImage.create(uploaded_file)

    # Make a prediction
    pred_class, pred_idx, outputs = learn.predict(img)

    # Display the prediction
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {outputs[pred_idx]:.2f}")
