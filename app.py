import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import time
import urllib.request
from pathlib import Path

# TextBlob for text cleaning and correction
try:
    from textblob import TextBlob
    TEXT_CLEANING_AVAILABLE = True
except ImportError:
    TEXT_CLEANING_AVAILABLE = False
    st.warning("Text cleaning not available. Install textblob for text correction.")

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

# Function to clean and correct OCR text using TextBlob
def clean_ocr_text(text):
    """Clean and correct OCR text using TextBlob"""
    if not TEXT_CLEANING_AVAILABLE or not text.strip():
        return text
    
    try:
        # Split text into lines and process each
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if line.strip():
                # Create a TextBlob object and correct spelling
                blob = TextBlob(line)
                corrected_line = str(blob.correct())
                cleaned_lines.append(corrected_line)
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    except Exception as e:
        st.warning(f"Text cleaning failed: {e}")
        return text

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        # Download model if needed
        model_path = download_model()
        model = YOLO(model_path)
        reader = easyocr.Reader(['en'])
        return model, reader
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

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
    
    # Clean and correct the detected text
    cleaned_text = clean_ocr_text(text)
    
    # Convert back to PIL format for display
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    
    return annotated_pil, text, cleaned_text

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
    
    # Clean and correct the detected text
    cleaned_text = clean_ocr_text(text)
    
    return annotated_frame, text, cleaned_text

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
            annotated_image, detected_text, cleaned_text = process_image(image, model, reader)
        
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
            st.text_area("Original OCR Text", value=detected_text, height=200, key="original_text")
        else:
            st.info("No text detected in the image.")
            
        # Display cleaned text if available
        if TEXT_CLEANING_AVAILABLE and cleaned_text.strip():
            st.subheader("Cleaned & Corrected Text")
            st.text_area("Corrected Text", value=cleaned_text, height=200, key="cleaned_text")
            
    else:
        st.info("Please upload an image using the sidebar to get started.")
        st.image("https://user-images.githubusercontent.com/26833433/253794510-93d75f2c-7f5a-4cbd-8d0e-6f0f5f3d1b9d.jpg", 
                 caption="Example of YOLO Object Detection", use_column_width=True)

elif app_mode == "Use Camera":
    st.info("Click 'Start Camera' to begin real-time object detection.")
    
    # Load models
    with st.spinner("Loading models... This may take a moment."):
        model, reader = load_models()
    
    # Camera controls with proper session state handling
    if 'run_camera' not in st.session_state:
        st.session_state.run_camera = False
        
    # Use a button instead of checkbox for better control
    if st.button('Start/Stop Camera'):
        st.session_state.run_camera = not st.session_state.run_camera
    
    if st.session_state.run_camera:
        # Try to access the camera
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                st.error("Failed to access camera. Please check your camera settings and ensure no other applications are using it.")
            else:
                # Set camera properties for better quality
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Create placeholders for the video feed and text
                FRAME_WINDOW = st.empty()
                camera_text = st.empty()
                cleaned_camera_text = st.empty()
                
                frame_count = 0
                while st.session_state.run_camera:
                    ret, frame = camera.read()
                    if not ret:
                        st.warning("⚠️ Camera disconnected. Please check your camera connection and click 'Start/Stop Camera' to restart.")
                        break
                    
                    # Process frame
                    annotated_frame, detected_text, cleaned_text = process_frame(frame, model, reader)
                    
                    # Convert BGR to RGB for display
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    FRAME_WINDOW.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Display detected text with a unique key
                    frame_count += 1
                    if detected_text.strip():
                        camera_text.text_area("Original OCR Text", value=detected_text, height=100, key=f"detection_text_{frame_count}")
                    else:
                        camera_text.info("No text detected in current frame.")
                    
                    # Display cleaned text if available
                    if TEXT_CLEANING_AVAILABLE and cleaned_text.strip():
                        cleaned_camera_text.text_area("Corrected Text", value=cleaned_text, height=100, key=f"cleaned_text_{frame_count}")
                    
                    # Add a small delay to control frame rate
                    time.sleep(0.03)
                
                # Release the camera when done
                camera.release()
                st.info("Camera stopped. Click 'Start/Stop Camera' to begin again.")
                
        except Exception as e:
            st.error(f"An error occurred with the camera: {str(e)}")
            if 'camera' in locals():
                camera.release()
    else:
        st.info("Camera is not running. Click 'Start/Stop Camera' to begin.")

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How to Use:
1. Select your preferred mode:
   - "Upload Image" to analyze a static image
   - "Use Camera" for real-time detection
2. For image upload:
   - Choose an image file (JPG, PNG)
   - Wait for processing
   - View results and detected text
3. For camera mode:
   - Click "Start/Stop Camera"
   - Allow camera access when prompted
   - View real-time detection
   - Click "Start/Stop Camera" again to stop
""")