import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from model.model import get_transfer_model
import pandas as pd


WEIGHTS_PATH = './model/default.pth' 
CLASS_NAMES = ['alien', 'predator'] 


INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


@st.cache_resource
def load_and_prepare_model(weights_path):
    # st.info("Loading ResNet50 model and custom weights... (Cached)")
    try:
        model = get_transfer_model()
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        model.eval()

        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.warning(f"Check if '{weights_path}' exists and the architecture is correct.")
        return None


model = load_and_prepare_model(WEIGHTS_PATH)




# streamlit app logic 
st.set_page_config(layout="centered", page_title="A/P Classifier")
st.title("Alien vs. Predator Classifier (ResNet50 Transfer Learning)")



if model is None:
    st.stop()


# File Uploader
uploaded_file = st.file_uploader("Upload an image (JPG, PNG) of an Alien or Predator:", type=["jpg", "jpeg", "png"])

# if the file is uploaded ...
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = INFERENCE_TRANSFORMS(image) # Apply the transformer to the image data



    # Add batch dimension meaning from 3D [C, H, W] -> 4D [1, C, H, W]
    input_batch = input_tensor.unsqueeze(0) 

    # --- Perform Inference ---
    with torch.no_grad():
        output = model(input_batch)
    
    # Get probabilities and predicted class
    probabilities = F.softmax(output, dim=1)
    conf_score, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence = conf_score.item() * 100
    


    st.info(f"Prediction: {predicted_class} | Confidence: **{confidence:.2f}%**")
    st.image(image, caption='Uploaded Image', use_container_width=True)

