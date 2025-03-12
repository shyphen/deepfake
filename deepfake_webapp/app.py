import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)  # Initialize model
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)
)
model.load_state_dict(torch.load("best_deepfake_model.pth", map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# CSV file setup
CSV_FILE = "predictions.csv"
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["Timestamp", "Filename", "Real_Prob", "Fake_Prob"])
    df.to_csv(CSV_FILE, index=False)

# Prediction function
def predict_image(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
        fake_prob, real_prob = probabilities.squeeze().tolist()  # Extract probabilities
    
    # Log prediction to CSV
    df = pd.read_csv(CSV_FILE)
    new_entry = pd.DataFrame({
        "Timestamp": [datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
        "Filename": ["Uploaded Image"],
        "Real_Prob": [real_prob * 100],
        "Fake_Prob": [fake_prob * 100]
    })
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    
    # Create bar chart visualization
    fig, ax = plt.subplots()
    ax.bar(["Fake", "Real"], [fake_prob * 100, real_prob * 100], color=["red", "green"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Deepfake Detection Confidence")
    
    # Format textual output
    result_text = f"Real: {real_prob * 100:.2f}% | Fake: {fake_prob * 100:.2f}%"
    
    return result_text, fig

# Gradio Interface
iface = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(type="pil"), 
    outputs=["text", gr.Plot()],
    title="Deepfake Detector",
    description="Upload an image to detect if it is real or fake with probability scores and visualization."
)

# Launch app
if __name__ == "__main__":
    iface.launch()
