import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.transforms as T
from PIL import Image
import io
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
import uvicorn
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os
from dotenv import load_dotenv
import boto3
load_dotenv()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
def upload_to_s3(file_bytes, s3_fi_name):
    try:
        s3_client.put_object(
            Bucket=os.getenv("S3_BUCKET"),
            Key=s3_fi_name,
            Body=file_bytes,
            ContentType= "image/jpeg"
        )
        return  True
    except Exception as e:
        print(f"Błąd s3:{e}")
        return False

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
ref_df = pd.read_csv("reference_stats.csv")
production_buffer = []
@app.get("/")
def root():
    return {"message": "Welcome to My U-Net Interface"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        return {"error": "Unsupported file type"}

    img_bytes = await file.read()


    image = Image.open(io.BytesIO(img_bytes)).convert('L')
    img_array = np.array(image)
    current_mean = np.mean(img_array)
    production_buffer.append(current_mean)

    drift_alert = False
    p_val = 1.0

    # Sprawdzanie dryfu co 20 zapytań
    if len(production_buffer) >= 20:
        stat, p_val = ks_2samp(ref_df['mean_brightness'], production_buffer)
        if p_val < 0.05:
            drift_alert = True
        production_buffer.clear()

    # 2. Klasyfikacja (Twoja istniejąca logika) [cite: 19, 20, 32]
    img_transformed = przygotuj_obraz(img_bytes)
    with torch.no_grad():
        raw = model(img_transformed)
        probabilities = F.softmax(raw, dim=1)
        class_id = torch.argmax(probabilities, 1).item()
        confidence = float(torch.max(probabilities).item())
        class_name = NAZWY_KLAS[class_id]
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"logs/{timestamp}_{file.filename}"

    upload_success = upload_to_s3(img_bytes, file_name)
    # 3. Zwracamy wszystko w jednym JSONie
    return {
        "Klasa": class_name,
        "Pewność": confidence,
        "S3 logged":upload_success,
        "Monitoring": {
            "Drift_Detected": drift_alert,
            "P_Value": round(p_val, 4),
            "Status": "Data distribution stable" if not drift_alert else "Data drift warning"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

