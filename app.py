import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from io import BytesIO
from gradcam import get_gradcam_heatmap, overlay_heatmap

def preprocess_image_bytes(image_bytes, target_size=(224,224)):
    """Process image bytes to a normalized array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_normalized = img.astype(np.float32) / 255.0
    return img, img_normalized  # return both original (for overlay) and normalized

st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🩺 Pneumonia Detection from Chest X‑ray")
st.markdown("Upload a chest X‑ray image and see the AI prediction with explainable heatmap.")

uploaded_file = st.file_uploader("Choose a chest X‑ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read bytes
    file_bytes = uploaded_file.read()
    
    # Display image
    image = Image.open(BytesIO(file_bytes))
    st.image(image, caption='Uploaded X‑ray', width=500)
    st.write("")
    st.write("Classifying...")
    
    try:
        # Preprocess
        img_original, img_array = preprocess_image_bytes(file_bytes)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Load model
        model = load_model('models/best_model.h5')
        
        # Prediction
        pred = model.predict(img_batch)[0][0]
        confidence = pred if pred >= 0.5 else 1 - pred
        result = "Pneumonia" if pred >= 0.5 else "Normal"
        
        # Display result
        if pred >= 0.5:
            st.error(f"**Prediction: {result}** (confidence: {confidence:.2%})")
        else:
            st.success(f"**Prediction: {result}** (confidence: {confidence:.2%})")
        
        # Grad-CAM heatmap
        # Find the last convolutional layer (you can adjust this name if needed)
        # MobileNetV2 last conv layer is usually 'mobilenetv2_1.00_224/Conv_1'
        conv_layer_name = 'mobilenetv2_1.00_224/Conv_1'
        
        # Generate heatmap
        heatmap = get_gradcam_heatmap(model, img_batch, conv_layer_name)
        
        # Overlay on original image (original size)
        overlay = overlay_heatmap(img_original, heatmap)
        overlay_resized = cv2.resize(overlay, (500, int(500 * img_original.shape[0] / img_original.shape[1])))
        
        st.image(overlay_resized, caption='Heatmap (red regions = areas of focus)', width=500)
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.info("If you see 'layer name not found', adjust the conv_layer_name in the code.")
