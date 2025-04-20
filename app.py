import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import datasets
import os

import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score
)
#########################################################################################


##### Prepare class for classification model #####
data_dir = r'datasets\THFOOD50-v1'
train_ds = datasets.ImageFolder(
    root=os.path.join(data_dir, "train")
)


## Path to the model directories
class_model_dir = r"models\classification_models"
detection_model_dir = r"models\detection_models"


### Load the ResNet34 model with pretrained weights
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = ResNet34_Weights.DEFAULT
class_model = resnet34(weights=weights)
num_ftrs = class_model.fc.in_features

# Add Dropout and an intermediate layer for better regularization
class_model.fc = nn.Sequential(
    nn.Dropout(p=0.3),                   
    nn.Linear(num_ftrs, len(train_ds.classes))  
)
class_model = class_model.to(device)

# Load the best model weights
class_model.load_state_dict(torch.load(os.path.join(class_model_dir, "model_best_resnet34new.pth")))
class_model.eval()

class_names = train_ds.classes


# Define the preprocessing pipeline for the classification model
preprocess = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize(256),
    transforms_v2.CenterCrop(224),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load YOLOv11 detection and classification models
detection_model = YOLO(os.path.join(detection_model_dir,"best_modell.pt"))       # e.g. train14/best.pt


### prepare calorie map ###

def load_calorie_map(file_path):
    calorie_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                name, cal = line.strip().split('=')
                calorie_map[name.strip()] = int(cal.strip())
    return calorie_map

# Load the calorie mapping
menu_calories = load_calorie_map("calories.txt")


# Function to process image
def draw_label(pil_img, text, *, pad=4):
    w, h = pil_img.size
    draw = ImageDraw.Draw(pil_img)

    # pick a font (default PIL bitmap font is fine & shipped with Pillow)
    try:
        font = ImageFont.truetype("arial.ttf", size=int(h * 0.07))
    except OSError:
        font = ImageFont.load_default()

    text_w, text_h = draw.textsize(text, font=font)
    # black translucent box behind text for readability
    box_coords = [0, 0, text_w + 2 * pad, text_h + 2 * pad]
    draw.rectangle(box_coords, fill=(0, 0, 0, 160))
    draw.text((pad, pad), text, fill="white", font=font)
    return pil_img

# ─── callback ─────────────────────────────────────────────────────────────────
def detect_and_classify(image):
    CONF_THRES = 0.48
    count = 0
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = detection_model(img_bgr)

    annotated_crops = []
    menu = []

    for r in results:
        for xyxy, conf, cls in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
        ):
            print(f"Detected {cls} with confidence {conf:.2f}")
            
            if conf < CONF_THRES:
                 print("=================skipping=====================")
                 continue
            
            x1, y1, x2, y2 = map(int, xyxy)
            crop_bgr = img_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)

            # Handle RGBA to RGB
            if pil_crop.mode != "RGB":
                pil_crop = pil_crop.convert("RGB")

            # Classification step (ResNet)
            input_tensor = preprocess(pil_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = class_model(input_tensor)
                probas = logits.softmax(1)
                class_id = probas.argmax(1).item()
                score = probas[0][class_id].item()
                category_name = class_names[class_id]

            if score < 0.3: #theshold
                continue
            # Final label
            label_txt = f"{category_name} ({score:.1%})"
            pil_crop = draw_label(pil_crop, label_txt)

            annotated_crops.append(pil_crop)
            menu.append(category_name)
    
    calories = []
    total_calories = 0
    if menu:
        conut = 1
        for e in menu:
            calories.append(f"🍔 {count}.{e} : {menu_calories.get(e, 'Unknown')} cal")
            total_calories += menu_calories.get(e, 0)
            count+=1

    text_output = f"List of detected dishes:\n\n" + "\n".join(calories)+ f"\n\nDetected {count-1} menu" + f"\n\n📜 Total Calories: {total_calories} cal"

    return annotated_crops if annotated_crops else None, text_output if text_output else ["No objects detected."]

# Gradio UI setup
iface = gr.Interface(
    fn=detect_and_classify,
    inputs=gr.Image(type="pil", label="📷 Upload Thai Food Image"),

    outputs=[
    gr.Gallery(label="🍛 Detected & Classified Dishes"),
    gr.Textbox(label="📝 Detection Summary", lines=5)
    ],

    title="🍽️ Thai Food Detector & Classifier",
    description="Upload an image containing Thai dishes. The system will detect each dish, crop it, and classify it using YOLOv11."
)

iface.launch()