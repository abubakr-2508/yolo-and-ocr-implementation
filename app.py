import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import time

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
You can either upload an image or use your camera for real-time detection!
""")

# Load models
@st.cache_resource
def load_models():
    model = YOLO("yolov8n.pt")
    reader = easyocr.Reader(['en'])
    return model, reader

# Function to process image
def process_image(image, model, reader):
    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # YOLO Object Detection
    results = model(img_cv)
    annotated_frame = results[0].plot()
    
    # OCR Text Detection
    ocr_results = reader.readtext(img_cv)
    text = ""
    for detection in ocr_results:
        text += detection[1] + "\n"
    
    # Convert back to PIL format for display
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    
    return annotated_pil, text

# Function to process video frame
def process_frame(frame, model, reader):
    # YOLO Object Detection
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # OCR Text Detection
    ocr_results = reader.readtext(frame)
    text = ""
    for detection in ocr_results:
        text += detection[1] + "\n"
    
    return annotated_frame, text

# Sidebar
st.sidebar.header("Settings")
app_mode = st.sidebar.selectbox("Choose the app mode", 
                                ["Upload Image", "Use Camera"])

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
            st.text_area("Recognized Text", value=detected_text, height=200)
        else:
            st.info("No text detected in the image.")
            
    else:
        st.info("Please upload an image using the sidebar to get started.")
        st.image("https://user-images.githubusercontent.com/26833433/253794510-93d75f2c-7f5a-4cbd-8d0e-6f0f5f3d1b9d.jpg", 
                 caption="Example of YOLO Object Detection", use_column_width=True)

elif app_mode == "Use Camera":
    st.info("Click 'Start Camera' to begin real-time object detection. Press 'q' or close the window to stop.")
    
    # Load models
    with st.spinner("Loading models... This may take a moment."):
        model, reader = load_models()
    
    # Camera controls
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    camera_text = st.empty()
    
    camera = None
    if run:
        camera = cv2.VideoCapture(0)
        # Set camera properties for better quality
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to access camera. Please check your camera settings.")
            break
            
        # Process frame
        annotated_frame, detected_text = process_frame(frame, model, reader)
        
        # Convert BGR to RGB for display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display frame
        FRAME_WINDOW.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
        
        # Display detected text
        if detected_text.strip():
            camera_text.text_area("Detected Text", value=detected_text, height=100)
        else:
            camera_text.info("No text detected in current frame.")
            
        # Add a small delay to control frame rate
        time.sleep(0.03)
        
    if camera:
        camera.release()

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### How to Use")
st.sidebar.markdown("""
**Upload Image Mode:**
1. Upload an image using the file uploader
2. Wait for the models to load and process the image
3. View the object detection results
4. See the detected text in the text area

**Use Camera Mode:**
1. Select "Use Camera" from the dropdown
2. Click "Start Camera" checkbox
3. Allow browser camera access when prompted
4. View real-time object detection
5. Uncheck "Start Camera" to stop
""")