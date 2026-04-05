import numpy as np
from PIL import Image
from ultralytics import YOLO
def load_model():
    return YOLO("best.pt")

model = load_model()

import streamlit as st
st.markdown("<h1 style='text-align: center;'>🛣️ ROAD DAMAGE DETECTOR 🛣️</h1>", unsafe_allow_html=True)
camera_img=st.camera_input("Capture an image📸") #camera input
st.write("<h5 style='text-align: center;'>Note : For better results, ensure that the image is detailed.</h5>",unsafe_allow_html=True)
uploaded_img=st.file_uploader("Upload an image",type=["jpg", "png", "jpeg"]) #uploading input

if uploaded_img:
    image = Image.open(uploaded_img).convert("RGB")
    image = np.array(image)

elif camera_img:
    image = Image.open(camera_img).convert("RGB")
    image = np.array(image)

else:
    image = None 

if image is not None:
    one, two = st.columns(2)

    with one:
        st.image(image, caption="Input Image", width=400)

    with two:
        results = model.predict(image, conf=0.18)
        st.image(results[0].plot(), caption="Result", width=400)
        boxes = results[0].boxes
        class_names = model.names
        counts = {"pothole": 0, "crack": 0, "manhole": 0}

    if boxes is not None:
        st.subheader("Details :-")

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = class_names[cls_id]

            counts[label] += 1

            st.write(f"{label} → {conf*100:.2f}%")

    st.write("<h3 style='text-align: center;'>-: Summary :-</h3>",unsafe_allow_html=True)
    st.write(f"<h5 style='text-align: center;'>Potholes: {counts['pothole']}</h5>",unsafe_allow_html=True)
    st.write(f"<h5 style='text-align: center;'>Cracks: {counts['crack']}</h5>",unsafe_allow_html=True)
    st.write(f"<h5 style='text-align: center;'>Manholes: {counts['manhole']}</h5>",unsafe_allow_html=True)
        
else:
    st.write("Upload or capture an image to proceed.")
