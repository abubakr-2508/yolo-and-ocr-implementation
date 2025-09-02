import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import time
import urllib.request
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="YOLO Object Detection with OCR",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 YOLO Object Detection with OCR")
st.markdown("""
This application combines real-time object detection using YOLOv8 and text recognition using EasyOCR.
You can either upload an image for detection!
""")

# Try to import OpenCV and EasyOCR with error handling
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("OpenCV is not available in this environment. Some features may not work.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    st.warning("EasyOCR is not available in this environment. OCR functionality will be disabled.")

# Function to download model if not present
@st.cache_resource
def download_model():
    model_path = "yolov8n.pt"
    if not Path(model_path).exists():
        with st.spinner("Downloading YOLOv8 nano model... This may take a minute."):
            try:
                urllib.request.urlretrieve(
                    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                    model_path
                )
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                st.stop()
    return model_path

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        # Download model if needed
        model_path = download_model()
        model = YOLO(model_path)
        
        # Only load OCR reader if EasyOCR is available
        reader = None
        if EASYOCR_AVAILABLE:
            reader = easyocr.Reader(['en'])
        return model, reader
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# Function to process image
def process_image(image, model, reader):
    # Convert PIL image to OpenCV format
    if OPENCV_AVAILABLE:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        # Fallback if OpenCV is not available
        img_cv = np.array(image)
    
    # YOLO Object Detection
    results = model(img_cv)
    annotated_frame = results[0].plot()
    
    # OCR Text Detection (only if EasyOCR is available)
    text = ""
    if reader and EASYOCR_AVAILABLE:
        try:
            ocr_results = reader.readtext(img_cv)
            for detection in ocr_results:
                text += detection[1] + "\n"
        except Exception as e:
            st.warning(f"OCR failed: {e}")
            text = "OCR not available"
    else:
        text = "OCR functionality disabled"
    
    # Convert back to PIL format for display
    if OPENCV_AVAILABLE:
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    else:
        # Fallback if OpenCV is not available
        annotated_pil = Image.fromarray(annotated_frame)
    
    return annotated_pil, text

# Sidebar
st.sidebar.header("Settings")
app_mode = st.sidebar.selectbox("Choose the app mode", 
                                ["Upload Image"])

# Main content
if app_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load models
        with st.spinner("Loading models... This may take a moment."):
            model, reader = load_models()
        
        # Process image
        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file)
            annotated_image, detected_text = process_image(image, model, reader)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Object Detection + OCR")
            st.image(annotated_image, use_column_width=True)
        
        # Display detected text
        st.subheader("Detected Text")
        if detected_text.strip():
            st.text_area("Recognized Text", value=detected_text, height=200, key="detected_text")
        else:
            st.info("No text detected in the image.")
            
    else:
        st.info("Please upload an image using the sidebar to get started.")
        st.image("https://user-images.githubusercontent.com/26833433/253794510-93d75f2c-7f5a-4cbd-8d0e-6f0f5f3d1b9d.jpg", 
                 caption="Example of YOLO Object Detection", use_column_width=True)

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How to Use:
1. Select "Upload Image" mode
2. Upload an image file (JPG, PNG)
3. Wait for processing
4. View results and detected text
""")