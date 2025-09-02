# YOLO Object Detection with OCR

This project combines real-time object detection using YOLOv8 and text recognition using EasyOCR. It can process uploaded images to detect objects and recognize text.

## Features

- Object detection with YOLOv8
- Text recognition with EasyOCR
- Image upload functionality
- Combined display of detection results and recognized text

## Requirements

- Python 3.6+
- OpenCV (opencv-python-headless for cloud deployment)
- NumPy
- Ultralytics YOLO
- EasyOCR
- Streamlit

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web Application

Run the Streamlit web app:

```bash
streamlit run app.py
```

## Deployment

### Streamlit Cloud

This application is optimized for Streamlit Cloud deployment:

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your forked repository
4. Set the main file path to `app.py`
5. Click "Deploy"

The app will automatically install dependencies from `requirements.txt` and start the application.

### Hugging Face Spaces

This application can also be deployed to Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Choose "Streamlit" as the SDK
3. Connect to your GitHub repository or upload files directly
4. The Space will automatically detect and use the `requirements.txt` file

## Model Files

Models are automatically downloaded at runtime from the official Ultralytics repository. This approach reduces the repository size and makes deployment easier.

## Note on Camera Functionality

The camera functionality has been disabled for cloud deployment due to browser security restrictions and resource limitations. For real-time camera detection, please run the application locally and use the desktop version (`detect.py`).