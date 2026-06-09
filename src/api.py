from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title="AI Leaf Doctor Inference Engine", version="1.0.0")

# Determine active runtime target environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verify model checkpoint state
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
    print(f"⚠️ Warning: Model checkpoint initialization failed. Ensure train.py is executed. Details: {e}")
    model = None

# Pure evaluation transformation pipeline
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
        # Stream image data from processing buffer
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Apply transformation matrices
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
        raise HTTPException(status_code=500, detail=f"Inference error execution breakdown: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)