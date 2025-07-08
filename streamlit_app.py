import streamlit as st
import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
from keras.models import load_model
import tempfile

# Load ConvLSTM model
@st.cache_resource
def load_conv_model():
    model = load_model("convlstm_cloud_model.h5")
    return model

model = load_conv_model()

st.set_page_config(page_title="INSAT-3D Cloud Cluster Detection")
st.title("🛰️ Tropical Cloud Cluster Detection using INSAT-3D IR1")

uploaded_files = st.file_uploader("📂 Upload 6 IR1 .tif images (half-hourly sequence)", type=["tif"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) != 6:
        st.error("⚠️ Please upload exactly 6 .tif images.")
    else:
        uploaded_files.sort(key=lambda x: x.name)  # Ensure correct time order

        sequence = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                img = tiff.imread(tmp.name)
                img = cv2.resize(img, (64, 64))
                img = img / 65535.0
                sequence.append(img)

        X = np.array(sequence)[np.newaxis, ..., np.newaxis]  # shape: (1, 6, 64, 64, 1)

        # Display uploaded images
        st.subheader("🛰️ Input IR1 Frames (Last 3 Hours)")
        cols = st.columns(6)
        for i in range(6):
            cols[i].image(X[0, i, :, :, 0], use_column_width=True, caption=f"Frame {i+1}")

        # Predict cloud clusters
        y_pred = model.predict(X)[0, :, :, 0]

        # Show prediction mask
        st.subheader("☁️ Predicted Cloud Cluster Mask (Binary Output)")
        fig1, ax1 = plt.subplots()
        ax1.imshow((y_pred > 0.5).astype(np.uint8), cmap='Reds')
        ax1.axis("off")
        st.pyplot(fig1)

        # Show overlay on India bounding box
        st.subheader("📍 Prediction Overlay on India Map (Approximate)")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.set_xlim([60, 100])
        ax2.set_ylim([5, 40])
        ax2.imshow((y_pred > 0.5).astype(np.uint8), extent=[60, 100, 5, 40], cmap='Reds', alpha=0.6)
        ax2.set_title("Cloud Cluster Mask over India (Lat-Lon)")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        ax2.grid(True)
        st.pyplot(fig2)
