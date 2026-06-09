from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

app = FastAPI(title="AI Leaf Doctor Inference Engine", version="1.0.0")

# Setup runtime computing target
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Safe runtime verification for weight matrices map
if os.path.exists('plant_model.pth'):
    try:
        checkpoint = torch.load('plant_model.pth', map_location=device)
        class_names = checkpoint['classes']

        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print("🧠 PyTorch Model loaded and compiled successfully into API backend!")
    except Exception as e:
        print(f"❌ Critical error configuring model layers: {e}")
        model = None
else:
    print("⚠️ Warning: 'plant_model.pth' checkpoint missing! Inference routing will remain locked.")
    model = None

# Production valuation transform sequence
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def read_root():
    return {"status": "Online", "service": "AI Plant Disease Diagnostics Engine"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model runtime weights uninitialized.")
        
    try:
        # Buffer input binary structures asynchronously
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Deploy matrix transformations pipeline
        tensor = inference_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, class_idx = torch.max(probabilities, 0)
            
        return {
            "prediction": class_names[class_idx].upper(),
            "confidence": round(float(confidence.item()) * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference process error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Dynamically query system ports to accommodate remote hosting environments
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)