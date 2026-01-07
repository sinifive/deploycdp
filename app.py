import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ---------- Page Config ----------
st.set_page_config(
    page_title="Crop Disease Detection",
    layout="centered"
)

st.title("🌱 Crop Disease Detection")
st.write("Upload a crop leaf image to predict disease")

# ---------- Load Class Names ----------
@st.cache_resource
def load_class_names(path="class_names.txt"):
    classes = []
    with open(path, "r") as f:
        for line in f:
            classes.append(line.strip().split(":", 1)[1].replace("_", " "))
    return classes

class_names = load_class_names()

# ---------- Load Model ----------
@st.cache_resource
def load_model(path="best_model.pth"):
    device = torch.device("cpu")
    model = torch.load(path, map_location=device)
    model.eval()
    return model

model = load_model()

# ---------- Image Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- UI ----------
uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

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
