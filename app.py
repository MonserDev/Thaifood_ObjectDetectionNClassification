import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLOv11 detection and classification models
detection_model = YOLO("C:/Users/niwat maneewong/OneDrive/Desktop/First/homwork/CS/Thaifood_ObjectDetectionNClassification/datasets/food/runs/detect/train14/weights/best.pt")       # e.g. train14/best.pt
classification_model = YOLO("path/to/classification_model.pt")  # e.g. class/best.pt

# Function to process image
def detect_and_classify(image):
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = detection_model(image_bgr)

    cropped_images = []
    food_labels = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = image_bgr[y1:y2, x1:x2]

            # Convert back to PIL format for classifier
            crop_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)

            # Classify with YOLOv11 classification model
            class_result = classification_model(pil_crop)
            class_name = class_result[0].names[int(class_result[0].probs.top1)]
            cropped_images.append(pil_crop)
            food_labels.append(class_name)

    # Combine results: [(label1, img1), (label2, img2), ...]
    return list(zip(food_labels, cropped_images))

# Gradio UI setup
iface = gr.Interface(
    fn=detect_and_classify,
    inputs=gr.Image(type="pil", label="📷 Upload Thai Food Image"),
    outputs=gr.Gallery(label="🍛 Detected & Classified Dishes").style(grid=[2], height="auto"),
    title="🍽️ Thai Food Detector & Classifier",
    description="Upload an image containing Thai dishes. The system will detect each dish, crop it, and classify it using YOLOv11."
)

iface.launch()
