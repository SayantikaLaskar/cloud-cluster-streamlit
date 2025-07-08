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

uploaded_files = st.file_uploader("Upload 6 IR1 .tif images (half-hourly sequence)", type=["tif"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) != 6:
        st.error("⚠️ Please upload exactly 6 .tif images.")
    else:
        uploaded_files.sort(key=lambda x: x.name)

        sequence = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                img = tiff.imread(tmp.name)
                img = cv2.resize(img, (64, 64))
                img = img / 65535.0
                sequence.append(img)

        X = np.array(sequence)[np.newaxis, ..., np.newaxis]  # shape: (1, 6, 64, 64, 1)

        st.subheader("🛰️ Input IR1 Frames")
        cols = st.columns(6)
        for i in range(6):
            cols[i].image(X[0, i, :, :, 0], use_column_width=True, caption=f"Frame {i+1}")

        y_pred = model.predict(X)[0, :, :, 0]

        st.subheader("☁️ Predicted Cloud Cluster Mask")
        fig, ax = plt.subplots()
        ax.imshow((y_pred > 0.5).astype(np.uint8), cmap='Reds')
        ax.axis("off")
        st.pyplot(fig)
