import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import urllib.request

# Page config
st.set_page_config(page_title="Crop Disease Detection", layout="centered")
st.title("🌱 Crop Disease Detection")

# Model download URL
MODEL_URL = "https://github.com/sinifive/deploycdp/releases/download/v1.0/best_model.pth"
MODEL_PATH = "best_model.pth"

# Load class names
@st.cache_resource
def load_class_names(path="class_names.txt"):
    classes = []
    with open(path, "r") as f:
        for line in f:
            classes.append(line.strip().split(":", 1)[1].replace("_", " "))
    return classes

class_names = load_class_names()

# Load model
@st.cache_resource
def load_model():
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

model = load_model()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# UI
uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing image..."):
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                idx = torch.argmax(probs).item()

            disease = class_names[idx]
            confidence = probs[idx].item() * 100

        st.success(f"**Disease:** {disease}")
        st.info(f"**Confidence:** {confidence:.2f}%")
