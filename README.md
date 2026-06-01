# ha-tuya-ai-detection

Human presence detection service for the [ha-tuya-cameras](https://github.com/a000a999a/ha-tuya-cameras) integration.

Runs a local YOLOv8n inference service that receives motion event images from `tuya_cameras`, classifies them, and signals the integration to email only when a human is detected.

---

## How it works

```
Tuya MQTT motion event (with image)
        │
        ▼
tuya_cameras (HA integration)
        │  POST image
        ▼
ai-detection service  ──── YOLOv8n inference ────▶  {human: true/false, confidence, annotated_image}
        │
        ▼
tuya_cameras:
  human detected  → send email with annotated image (bounding boxes drawn)
  no human        → discard silently
  service error   → send email with original image (fail-open)
  no image        → discard silently
```

---

## Requirements

- Docker + Docker Compose
- ~2 GB disk (Python image + PyTorch + YOLOv8n model)
- ~500 MB RAM at runtime
- CPU-only — no GPU required (YOLOv8n runs in ~250–400 ms on a modern dual-core)

---

## Setup

### 1. Clone and start the service

```bash
git clone https://github.com/a000a999a/ha-tuya-ai-detection
cd ha-tuya-ai-detection
docker compose up -d
```

The first build downloads the YOLOv8n model (~6 MB). Subsequent starts are fast.

### 2. Verify it is running

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"yolov8n"}
```

### 3. Configure tuya_cameras

In Home Assistant:  
**Settings → Devices & Services → Tuya Cameras → Configure → AI Detection**

- Enable AI detection: ✓
- Inference service URL: `http://localhost:8000`

The options flow validates `/health` before saving.

---

## API

### `GET /health`

Returns `{"status": "ok", "model": "yolov8n"}` when the service is ready.

### `POST /analyze`

**Request:** multipart form with field `file` — a JPEG image.

**Response:**

```json
{
  "human": true,
  "confidence": 0.874,
  "annotated_image_b64": "<base64-encoded JPEG with bounding boxes>"
}
```

| Field | Type | Description |
|---|---|---|
| `human` | bool | `true` if a person was detected above the 0.60 confidence threshold |
| `confidence` | float | Highest person detection confidence (0–1) |
| `annotated_image_b64` | string | Base64 JPEG with YOLO bounding boxes drawn |

---

## Model

- **Model:** YOLOv8n (nano) — optimised for CPU inference
- **Confidence threshold:** 0.60 (hardcoded)
- **Detection class:** COCO class 0 (person)

---

## Extending

To swap to a larger model (e.g. `yolov8s.pt`), edit the `YOLO(...)` line in `service/main.py` and rebuild. Note that `yolov8s` is ~3× slower on CPU.

To run on GPU, install the CUDA-enabled PyTorch variant and uncomment the `device="cuda"` parameter in the `model()` call.
