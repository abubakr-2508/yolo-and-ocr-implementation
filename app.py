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
    """Clean and correct OCR text using TextBlob with enhanced handling for special cases"""
    if not TEXT_CLEANING_AVAILABLE or not text.strip():
        return text
    
    try:
        # Split text into lines and process each
        lines = text.split('\n')
        cleaned_lines = []
        
        # Check if in handwriting mode
        handwriting_mode = st.session_state.get('handwriting_mode', False)
        
        for line in lines:
            if not line.strip():
                cleaned_lines.append(line)
                continue
                
            # Extract text part from lines that might include confidence score
            # Format: "H3Lo [Conf: 0.95]"
            if "[Conf:" in line:
                parts = line.split("[Conf:")
                processed_line = parts[0].strip()
                conf_part = "[Conf:" + parts[1] if len(parts) > 1 else ""
            else:
                processed_line = line.strip()
                conf_part = ""
            
            # Handle special cases with numbers that might be letters
            # Common substitutions: 0->O, 1->I/L, 3->E, 5->S, 8->B
            substitutions = {
                '0': 'O',
                '1': 'I',
                '3': 'E',
                '5': 'S',
                '8': 'B'
            }
            
            # Check if this looks like a word with number substitutions
            has_letters = any(c.isalpha() for c in processed_line)
            has_numbers = any(c.isdigit() for c in processed_line)
            
            # Handwriting-specific processing
            if handwriting_mode:
                # For handwritten text with both letters and numbers, favor letter substitutions
                if has_letters and has_numbers:
                    # Try letter substitutions first for handwriting
                    number_replaced = processed_line
                    for num, letter in substitutions.items():
                        number_replaced = number_replaced.replace(num, letter)
                    
                    # For handwriting, we trust our substitutions more than TextBlob
                    if any(c.isdigit() for c in number_replaced):
                        # If still has digits, try TextBlob
                        blob = TextBlob(number_replaced)
                        corrected = str(blob.correct())
                        if corrected != number_replaced:
                            processed_line = corrected
                        else:
                            processed_line = number_replaced
                    else:
                        # No digits left, just use our substitutions
                        processed_line = number_replaced
                        
                        # Force uppercase if it looks like an all-caps word
                        if processed_line.isupper() or processed_line.replace(' ', '').isalpha():
                            processed_line = processed_line.upper()
                else:
                    # Regular text correction
                    blob = TextBlob(processed_line)
                    processed_line = str(blob.correct())
            else:
                # Non-handwriting mode - more standard approach
                # If it has both letters and numbers, try different variations
                if has_letters and has_numbers and len(processed_line) <= 15:
                    # Try with TextBlob first
                    blob = TextBlob(processed_line)
                    corrected = str(blob.correct())
                    
                    # If TextBlob didn't change it much, try our substitutions
                    if corrected == processed_line or corrected.lower() == processed_line.lower():
                        # Create variations with common number-to-letter substitutions
                        variations = [processed_line]
                        
                        # Generate variation with number substitutions
                        number_replaced = processed_line
                        for num, letter in substitutions.items():
                            number_replaced = number_replaced.replace(num, letter)
                        
                        if number_replaced != processed_line:
                            variations.append(number_replaced)
                        
                        # Also try all uppercase version if it has lowercase letters
                        if not processed_line.isupper() and any(c.islower() for c in processed_line):
                            variations.append(processed_line.upper())
                        
                        # Try TextBlob correction on each variation
                        best_correction = processed_line
                        
                        for var in variations:
                            try:
                                blob_var = TextBlob(var)
                                corrected_var = str(blob_var.correct())
                                
                                # Simple heuristic: prefer corrections with more real words
                                if corrected_var != var:
                                    # If correction changed something, it's likely better
                                    best_correction = corrected_var
                                    break
                            except:
                                continue
                        
                        processed_line = best_correction
                    else:
                        processed_line = corrected
                else:
                    # Regular TextBlob correction for normal text
                    blob = TextBlob(processed_line)
                    processed_line = str(blob.correct())
            
            # Reattach confidence information if present
            if conf_part:
                cleaned_lines.append(f"{processed_line} {conf_part}")
            else:
                cleaned_lines.append(processed_line)
        
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
        
        # Check if handwriting mode is enabled
        handwriting_mode = st.session_state.get('handwriting_mode', False)
        
        # Load OCR model with appropriate settings
        if handwriting_mode:
            # For handwriting, we'll include English and use a specific model
            reader = easyocr.Reader(['en'], recog_network='best', 
                                  gpu=False, 
                                  download_enabled=True)
        else:
            # Default OCR reader
            reader = easyocr.Reader(['en'])
            
        return model, reader
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# Function to preprocess image for better OCR
def preprocess_image_for_ocr(image, options):
    """Apply preprocessing techniques to improve OCR accuracy"""
    # Make a copy of the image
    processed = image.copy()
    
    if "Thresholding" in options:
        # Convert to grayscale if not already
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed
            
        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    if "Noise Removal" in options:
        # Apply median blur to remove noise
        processed = cv2.medianBlur(processed, 3)
    
    if "Sharpening" in options:
        # Create kernel for sharpening
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        # Apply kernel
        processed = cv2.filter2D(processed, -1, kernel)
    
    if "Dilation" in options:
        # Create kernel for dilation
        kernel = np.ones((2,2), np.uint8)
        # Apply dilation to make text thicker
        processed = cv2.dilate(processed, kernel, iterations=1)
    
    return processed

# Function to process image
def process_image(image, model, reader):
    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # YOLO Object Detection
    results = model(img_cv)
    annotated_frame = results[0].plot()
    
    # Get OCR preprocessing options from session state
    ocr_enhance = st.session_state.get('ocr_enhance', False)
    preprocessing_options = st.session_state.get('preprocessing_options', [])
    
    # Preprocess image for OCR if enabled
    if ocr_enhance and preprocessing_options:
        ocr_image = preprocess_image_for_ocr(img_cv, preprocessing_options)
        
        # Display preprocessed image in sidebar for debugging if requested
        if st.session_state.get('show_preprocessed', False):
            st.sidebar.image(ocr_image, caption="Preprocessed Image", use_column_width=True)
    else:
        ocr_image = img_cv
    
    # OCR Text Detection
    ocr_results = reader.readtext(ocr_image)
    text = ""
    
    # Create a copy for annotation with text
    text_annotated = annotated_frame.copy()
    
    # Draw OCR results on image for better visualization
    for detection in ocr_results:
        # Get bounding box, text and confidence
        box = detection[0]
        detected_text = detection[1]
        conf = detection[2]
        
        # Convert box to proper format
        box = np.array(box, dtype=np.int32)
        
        # Draw rectangle around text
        cv2.polylines(text_annotated, [box], True, (0, 255, 0), 2)
        
        # Add text and confidence above the box
        text_position = (min(box[:, 0]), min(box[:, 1]) - 10)
        text_with_conf = f"{detected_text} ({conf:.2f})"
        cv2.putText(text_annotated, text_with_conf, text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add to plain text output
        text += f"{detected_text} [Conf: {conf:.2f}]\n"
    
    # Clean and correct the detected text
    cleaned_text = clean_ocr_text(text)
    
    # Convert back to PIL format for display
    annotated_pil = Image.fromarray(cv2.cvtColor(text_annotated, cv2.COLOR_BGR2RGB))
    
    return annotated_pil, text, cleaned_text

# Function to process video frame
def process_frame(frame, model, reader):
    # YOLO Object Detection
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # Get OCR preprocessing options from session state
    ocr_enhance = st.session_state.get('ocr_enhance', False)
    preprocessing_options = st.session_state.get('preprocessing_options', [])
    
    # Preprocess image for OCR if enabled
    if ocr_enhance and preprocessing_options:
        ocr_image = preprocess_image_for_ocr(frame, preprocessing_options)
    else:
        ocr_image = frame
    
    # OCR Text Detection
    ocr_results = reader.readtext(ocr_image)
    text = ""
    
    # Create a copy for annotation with text
    text_annotated = annotated_frame.copy()
    
    # Draw OCR results on image for better visualization
    for detection in ocr_results:
        # Get bounding box, text and confidence
        box = detection[0]
        detected_text = detection[1]
        conf = detection[2]
        
        # Convert box to proper format
        box = np.array(box, dtype=np.int32)
        
        # Draw rectangle around text
        cv2.polylines(text_annotated, [box], True, (0, 255, 0), 2)
        
        # Add text and confidence above the box
        text_position = (min(box[:, 0]), min(box[:, 1]) - 10)
        text_with_conf = f"{detected_text} ({conf:.2f})"
        cv2.putText(text_annotated, text_with_conf, text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add to plain text output
        text += f"{detected_text} [Conf: {conf:.2f}]\n"
    
    # Clean and correct the detected text
    cleaned_text = clean_ocr_text(text)
    
    return text_annotated, text, cleaned_text

# Sidebar
st.sidebar.header("Settings")
app_mode = st.sidebar.selectbox("Choose the app mode", 
                                ["Upload Image", "Use Camera"])

# OCR settings
st.sidebar.subheader("OCR Settings")

# Handwriting mode
handwriting_mode = st.sidebar.checkbox("Handwriting Mode", value=True,
                                help="Optimize OCR for handwritten text")
# Store in session state
st.session_state['handwriting_mode'] = handwriting_mode

ocr_enhance = st.sidebar.checkbox("Enhance OCR", value=True, 
                                help="Apply image preprocessing to improve text recognition")

# Store OCR enhance setting in session state
st.session_state['ocr_enhance'] = ocr_enhance

if ocr_enhance:
    preprocessing_options = st.sidebar.multiselect(
        "Preprocessing Options",
        ["Thresholding", "Noise Removal", "Sharpening", "Dilation"],
        default=["Thresholding", "Noise Removal"]
    )
    
    # Store preprocessing options in session state
    st.session_state['preprocessing_options'] = preprocessing_options
    
    # Option to show preprocessed image (helpful for debugging)
    show_preprocessed = st.sidebar.checkbox("Show Preprocessed Image", value=False,
                                          help="Display the preprocessed image used for OCR")
    st.session_state['show_preprocessed'] = show_preprocessed

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
            
            # Add explanation for confidence scores
            st.caption("Note: [Conf: X.XX] indicates the confidence score of the OCR detection.")
        else:
            st.info("No text detected in the image.")
            
        # Display cleaned text if available
        if TEXT_CLEANING_AVAILABLE and cleaned_text.strip():
            st.subheader("Cleaned & Corrected Text")
            st.text_area("Corrected Text", value=cleaned_text, height=200, key="cleaned_text")
            
            # Add information about the correction process
            if st.session_state.get('handwriting_mode', False):
                st.info("✨ Handwriting mode is enabled: Special text processing optimized for handwritten text has been applied.")
            
            # Show advanced options
            if st.checkbox("Show Text Processing Details", value=False):
                st.markdown("""### How Text Correction Works:
                1. **OCR Detection**: EasyOCR detects text and provides confidence scores
                2. **Number to Letter Conversion**: Common substitutions (e.g., 3→E, 0→O)
                3. **Spelling Correction**: TextBlob analyzes and corrects detected text
                4. **Handwriting Optimization**: Special handling for handwritten characters""")
                
                # Show substitution dictionary
                st.json({
                    "Number to Letter Substitutions": {
                        "0": "O",
                        "1": "I",
                        "3": "E",
                        "5": "S",
                        "8": "B"
                    }
                })
            
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

### OCR Enhancement Tips:
- **Handwriting Mode**: Best for handwritten notes like your "H3LLO" example
- **Thresholding**: Improves contrast for better text detection
- **Noise Removal**: Removes small artifacts that might confuse OCR
- **Sharpening**: Makes text edges clearer
- **Dilation**: Makes thin text strokes thicker for better recognition
""")