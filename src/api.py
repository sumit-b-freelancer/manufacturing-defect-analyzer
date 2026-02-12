from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from src.models.model import SentinelModel
from src.utils.gradcam import ExplainableAI
import uvicorn
import numpy as np
import base64

app = FastAPI(title="Sentinel-Vision API", version="1.0.0")

# Global variables for model
model = None
device = None
explainable_ai = None
labels = {0: "No Defect", 1: "Defect"} # Simple mapping, can be expanded

@app.on_event("startup")
async def startup_event():
    global model, device, explainable_ai
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    # Load model structure
    model = SentinelModel(model_name='tf_efficientnetv2_s', num_classes=2)
    
    # Load weights if available, otherwise just init (for demo purposes)
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("Loaded trained weights.")
    else:
        print("Warning: No weights found, using random initialization.")
    
    model.to(device)
    model.eval()
    
    explainable_ai = ExplainableAI(model)
    print("Model loaded successfully.")

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0), np.array(image.resize((224, 224))).astype(np.float32) / 255.0

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        input_tensor, original_image = transform_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data")

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        score, predicted_idx = torch.max(probabilities, 1)

    predicted_label = labels.get(predicted_idx.item(), "Unknown")
    confidence = score.item()

    # Generate Grad-CAM
    # We want to explain the predicted class
    _, heatmap = explainable_ai.generate_heatmap(input_tensor, original_image, target_class=predicted_idx.item())
    
    heatmap_b64 = None
    if heatmap is not None:
        # Convert heatmap (numpy) to base64 string
        # heatmap is RGB float, need to convert to uint8 image
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        buffered = io.BytesIO()
        heatmap_img.save(buffered, format="PNG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "heatmap_base64": heatmap_b64
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

import os

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
