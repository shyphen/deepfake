import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import io

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

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
def predict_image(image, filename):
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
        "Filename": [filename],
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
    
    # Save plot to a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()
    
    return {"Real": f"{real_prob * 100:.2f}%", "Fake": f"{fake_prob * 100:.2f}%"}, img_io

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files["image"]
    image = Image.open(image_file)
    filename = image_file.filename
    result, img_io = predict_image(image, filename)
    
    return jsonify({"result": result, "chart": "/chart"})

@app.route("/chart")
def get_chart():
    _, img_io = predict_image(Image.open("static/temp_image.png"), "temp")
    return send_file(img_io, mimetype='image/png')

@app.route("/get_history", methods=["GET"])
def get_history():
    df = pd.read_csv(CSV_FILE)
    return jsonify(df.to_dict(orient="records"))

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)