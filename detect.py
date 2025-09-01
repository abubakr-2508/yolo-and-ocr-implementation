# Install required libraries first if not already installed:
# pip install ultralytics opencv-python easyocr numpy

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

# Load YOLOv8 model (you can use 'yolov8n.pt' for speed, 'yolov8s.pt' for better accuracy)
model = YOLO("yolov8n.pt")

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# --- Increase camera resolution ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- YOLO Object Detection ---
    results = model(frame)
    annotated_frame = results[0].plot()

    # --- OCR Text Detection ---
    ocr_results = reader.readtext(frame)
    text = ""
    for detection in ocr_results:
        text += detection[1] + "\n"

    # --- Create a white panel for OCR text ---
    panel = 255 * np.ones((frame.shape[0], 300, 3), dtype=np.uint8)

    y0, dy = 30, 30
    for i, line in enumerate(text.splitlines()):
        y = y0 + i*dy
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # --- Combine object detection + OCR panel ---
    combined = np.hstack((annotated_frame, panel))

    # --- Display final output ---
    cv2.imshow("Object + OCR Detection", combined)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
