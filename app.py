import streamlit as st
from fastai.vision.all import *
import gdown

def load_model():
    url = 'https://drive.google.com/file/d/1lujIhjfBh7L0bNxvIrCiV96plymN01C/view?usp=drive_link' 
    path = Path('Mongolian_foods_classifier.pkl')
    
    if not path.exists():
        gdown.download(url, str(path), quiet=False)
    
    return load_learner(path)

st.title("Mongolian Food Classifier")
upload = st.file_uploader("Upload food image", type=['jpg', 'png', 'jpeg'])

if upload:
    img = PILImage.create(upload)
    st.image(img, width=300)
    
    learn = load_model()
    pred, _, probs = learn.predict(img)
    
    st.subheader(f"Prediction: {pred}")
    st.write(f"Confidence: {probs[learn.dls.vocab.o2i[pred]]:.2%}")
