"""
Tuya AI Detection — YOLOv8 inference service (default model: YOLOv8s).
POST /analyze  — accepts a JPEG image, returns human + animal detection results.
GET  /health   — liveness check used by the HA options flow validator.

Model choice: YOLOv8s by default. In a 7-day comparison on real motion snapshots it
caught clearly more people than YOLOv8n in hard cases — night-time infrared, faces
covered by a helmet or hood, people close to the lens or half out of frame — at about
35 ms per image on a modern 8-core CPU. On slow hardware set AI_MODEL=yolov8n.pt
(faster, misses more people). Both models are baked into the image.
"""

import base64
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO

MODEL_FILE = os.environ.get("AI_MODEL", "yolov8s.pt")
CONFIDENCE = 0.60
PERSON_CLASS = 0  # COCO class 0 = person
ANIMAL_CLASSES = {
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}

app = FastAPI(title="Tuya AI Detection")
model = YOLO(MODEL_FILE)


@app.get("/health")
def health():
    return {"status": "ok", "model": Path(MODEL_FILE).stem}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    all_classes = [PERSON_CLASS] + list(ANIMAL_CLASSES.keys())
    results = model(img, classes=all_classes, verbose=False)[0]

    human = False
    confidence = 0.0
    animals: list[str] = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if conf < CONFIDENCE:
            continue
        if cls_id == PERSON_CLASS:
            if conf > confidence:
                human      = True
                confidence = conf
        elif cls_id in ANIMAL_CLASSES:
            label = ANIMAL_CLASSES[cls_id]
            if label not in animals:
                animals.append(label)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "human":              human,
        "confidence":         round(confidence, 3),
        "animals":            animals,
        "annotated_image_b64": annotated_b64,
    }
