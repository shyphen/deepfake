import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gradio as gr
import pandas as pd
import os
import datetime

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
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
        probabilities = F.softmax(output, dim=1)
        fake_prob, real_prob = probabilities.squeeze().tolist()
    
    # Log prediction to CSV
    df = pd.read_csv(CSV_FILE)
    new_entry = pd.DataFrame({
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Filename": ["Uploaded Image"],
        "Real_Prob": [real_prob * 100],
        "Fake_Prob": [fake_prob * 100]
    })
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    
    return {"Real": f"{real_prob * 100:.2f}%", "Fake": f"{fake_prob * 100:.2f}%"}

# Gradio Interface
iface = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(type="pil"), 
    outputs="label",
    title="Deepfake Detector",
    description="Upload an image to detect if it is real or fake with probability scores."
)

# Launch app
if __name__ == "__main__":
    iface.launch()
