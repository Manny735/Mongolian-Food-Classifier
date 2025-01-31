import streamlit as st
from fastai.vision.all import load_learner, PILImage
from PIL import Image
import gdown
import os

# Define any custom functions used in model training here
# Example: def custom_splitter(model): ...

st.title("Mongolian Foods Classifier 🥘")

@st.cache_resource
def load_model():
    file_id = "1lujIhjfBh7LB0bNxvIrCiV96plymN01C"
    model_url = f"https://drive.google.com/uc?id=1lujIhjfBh7LB0bNxvIrCiV96plymN01C"
    model_path = "Mongolian_foods_classifier.pkl"
    
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)
    
    return load_learner(model_path)

learn = load_model()

uploaded_file = st.file_uploader("Upload an image of Mongolian food", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img = PILImage.create(uploaded_file)
    pred_class, _, outputs = learn.predict(img)
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {outputs[learn.dls.vocab.o2i[pred_class]]:.2f}")
