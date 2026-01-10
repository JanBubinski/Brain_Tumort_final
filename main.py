from gc import freeze

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.transforms as T
from PIL import Image
import io
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
import uvicorn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "My_UNet_Arch.pth"
NUM_CLASSES = 4
NAZWY_KLAS = ["notumor", "pituitary", "meningioma", "glioma"]


unet = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None)
encoder = unet.encoder

class SelectLastItem(nn.Module):
    def forward(self, x):
        return x[-1]

model = nn.Sequential(
    encoder,
    SelectLastItem(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(encoder.out_channels[-1], NUM_CLASSES)
).to(DEVICE)


state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict.pop("4.weight", None)
state_dict.pop("4.bias", None)
model.load_state_dict(state_dict, strict=False)
model.eval()
model.to(DEVICE)
print(f"Model załadowany na {DEVICE}. Gotowy do pracy.")


transforms=T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
def przygotuj_obraz(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transforms(img)
    return img_tensor.unsqueeze(0).to(DEVICE)
app = FastAPI(title="Interface for medical USG(mri patients images)")
@app.get("/")
def root():
    return {"message": "Welcome to My U-Net Interface"}
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_transformed = przygotuj_obraz(img_bytes)
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        return {"error": "Unsupported file type"};
    with torch.no_grad():
        raw = model(img_transformed)
        probabilities = F.softmax(raw, dim=1)
        class_id = torch.argmax(probabilities, 1).item()
        confidence = float(torch.max(probabilities).item())
        class_name = NAZWY_KLAS[class_id]

    return {
        "Klasa": class_name,
        "Pewność": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

