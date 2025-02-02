import streamlit as st
from fastai.vision.all import *
import gdown
from PIL import Image

# Function to download and load the model
@st.cache_resource
def load_model():
    file_id = "1NQ5aNhGP3mPpTgDcf0Ii9fT3X797qnuy"  # Replace with your actual Google Drive file ID
    url = f"https://drive.google.com/file/d/1NQ5aNhGP3mPpTgDcf0Ii9fT3X797qnuy/view?usp=drive_link"
    model_path = Path("Mongolian_food_classifier1.pkl")

    # Download model if it doesn't exist
    if not model_path.exists():
        with st.spinner("Downloading model..."):
            try:
                gdown.download(url, str(model_path), quiet=False)
            except Exception as e:
                st.error("❌ Failed to download the model. Check the file link.")
                return None

    # Try loading the model
    try:
        model = load_learner(model_path)
        return model
    except Exception as e:
        st.error("❌ Failed to load the model. The file may be corrupted or the path is incorrect.")
        return None

# Streamlit UI
st.title("🍲 Mongolian Food Classifier 🇲🇳")
st.write("Upload an image of Mongolian food, and the model will classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Load model
    learn = load_model()
    
    if learn:  # Only proceed if model loaded successfully
        try:
            pred, _, probs = learn.predict(PILImage.create(uploaded_file))
            st.subheader(f"🍽 Prediction: {pred}")
            st.write(f"✅ Confidence: {probs[learn.dls.vocab.o2i[pred]]:.2%}")
        except Exception as e:
            st.error("❌ Error making prediction. Ensure the image format is correct.")
    else:
        st.warning("⚠ Please check the model file and try again.")
