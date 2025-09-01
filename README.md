# YOLO Object Detection with OCR

This project combines real-time object detection using YOLOv8 and text recognition using EasyOCR. It captures video from the webcam, detects objects, recognizes text, and displays both results side-by-side.

## Features

- Real-time object detection with YOLOv8
- Text recognition with EasyOCR
- Combined display of detection results and recognized text
- High-resolution camera feed (1280x720)

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Ultralytics YOLO
- EasyOCR

## Installation

```bash
pip install ultralytics opencv-python easyocr numpy
```

## Usage

Run the detection script:

```bash
python detect.py
```

Press 'q' to quit the application.

## Model Files

- `yolov8n.pt` - Nano model (fastest, least accurate)
- `yolov8s.pt` - Small model (balanced speed and accuracy)
- `yolov8x.pt` - Extra large model (slowest, most accurate)

To use a different model, modify the model loading line in `detect.py`:

```python
model = YOLO("yolov8s.pt")  # or yolov8x.pt
```