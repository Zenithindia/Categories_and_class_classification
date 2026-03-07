import streamlit as st
from PIL import Image

st.title("Camera + Upload Test")

tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload"])

img = None

with tab1:
    cam = st.camera_input("Take a picture")
    if cam is not None:
        img = Image.open(cam)

with tab2:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if up is not None:
        img = Image.open(up)

if img is not None:
    st.image(img, caption="Selected Image", use_container_width=True)