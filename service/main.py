"""
Tuya AI Detection — YOLOv8n inference service.
POST /analyze  — accepts a JPEG image, returns human detection result + annotated image.
GET  /health   — liveness check used by the HA options flow validator.
"""

import base64
import io

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO

CONFIDENCE = 0.60
PERSON_CLASS = 0  # COCO class 0 = person

app = FastAPI(title="Tuya AI Detection")
model = YOLO("yolov8n.pt")


@app.get("/health")
def health():
    return {"status": "ok", "model": "yolov8n"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    results = model(img, verbose=False)[0]

    human = False
    confidence = 0.0
    for box in results.boxes:
        if int(box.cls[0]) == PERSON_CLASS:
            conf = float(box.conf[0])
            if conf >= CONFIDENCE and conf > confidence:
                human = True
                confidence = conf

    annotated = Image.fromarray(results.plot()[..., ::-1])  # BGR → RGB
    buf = io.BytesIO()
    annotated.save(buf, "JPEG", quality=85)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "human": human,
        "confidence": round(confidence, 3),
        "annotated_image_b64": annotated_b64,
    }
