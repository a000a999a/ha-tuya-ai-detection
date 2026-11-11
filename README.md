# ha-tuya-ai-detection

Human presence detection service for the [ha-tuya-cameras](https://github.com/a000a999a/ha-tuya-cameras) integration.

Runs a local YOLOv8 inference service (YOLOv8s by default) that receives motion event images from `tuya_cameras`, classifies them, and signals the integration to email only when a human is detected.

---

## How it works

```
Tuya MQTT motion event (with image)
        │
        ▼
tuya_cameras (HA integration)
        │  POST image
        ▼
ai-detection service  ──── YOLOv8s inference ────▶  {human: true/false, confidence, annotated_image}
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
- **~10 GB disk** for the image. On x86-64, `pip install ultralytics` pulls the standard PyTorch
  build including its NVIDIA/CUDA libraries (~5.5 GB of the total), even when no GPU is present.
- ~0.5 GB RAM at runtime
- CPU-only — no GPU required (see [Hardware recommendations](#hardware-recommendations))

**This repo does not install Docker for you** — it assumes Docker + Docker Compose are already
available on the machine you're running it on. Whether that's a 1-step or 2-step setup for you
depends on what you're starting from:

- **Already have Docker/Docker Compose on this machine?** (common if you also run Home Assistant
  itself in Docker) → **1 step**: clone this repo and `docker compose up -d` below.
- **Linux machine without Docker** (bare install, or Home Assistant Supervised on Debian)? →
  **2 steps**: install Docker + Docker Compose first
  ([docs.docker.com/engine/install](https://docs.docker.com/engine/install/)), *then* the same
  clone + `docker compose up -d` below.
- **Home Assistant OS** (e.g. Home Assistant Green / Yellow, or a Raspberry Pi running HA OS)?
  HA OS does not let you install Docker or run your own containers next to it, so the service
  **cannot run on that box**. Run it on a second machine on the same network and point the
  integration at it (see *Network exposure* below).

---

## Hardware recommendations

The service checks up to 4 images per motion event, so speed per image matters more than
anything else. Measured and estimated speed per image:

| Machine | YOLOv8s (default) | YOLOv8n (`AI_MODEL=yolov8n.pt`) | Recommendation |
|---|---|---|---|
| Modern 8-core mini PC (measured: Ryzen 9 7940HS, 8 cores) | ~35 ms | ~15 ms | Plenty of headroom |
| Intel N100-class mini PC, 4 cores | ~200–300 ms (estimate) | ~70–100 ms (estimate) | **Recommended minimum** |
| Raspberry Pi 5 | ~0.5–1 s (estimate, untested) | faster, untested | Use YOLOv8n, or run the service on another machine |

- **RAM:** 8 GB total is comfortable for Home Assistant plus this service.
- **Disk:** an SSD with 128 GB+, of which ~10 GB is this image.
- **GPU:** not needed. An NVIDIA GPU can be used (see [Extending](#extending)) but brings little
  for a few images per motion event.
- **Timeout:** the integration waits up to 5 s per image. On slow hardware, several cameras
  firing at once can queue up past that; the alert is then sent without AI filtering
  (fail-open), so nothing is lost, but it may be a false alarm.

For the whole suite (Home Assistant, cameras, watering, network) see
[Hardware recommendations in Tuya Home Core](https://github.com/a000a999a/ha-tuya-home-core#hardware-recommendations).

`ha-tuya-cameras` (the main integration) never requires Docker — this AI service is the only
piece of the solution that does, and only if you want human-detection filtering. Everything else
(motion alerts, SD monitoring, email delivery) works fully without it.

---

## Setup

### 1. Clone and start the service

```bash
git clone https://github.com/a000a999a/ha-tuya-ai-detection
cd ha-tuya-ai-detection
docker compose up -d
```

The first build downloads PyTorch and both YOLOv8 models (YOLOv8s ~22 MB, YOLOv8n ~6 MB) and
takes a while. Subsequent starts are fast.

To use the faster, less accurate YOLOv8n on slow hardware:

```bash
AI_MODEL=yolov8n.pt docker compose up -d   # or put AI_MODEL=yolov8n.pt in a .env file
```

### 2. Verify it is running

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"yolov8s"}
```

### 3. Configure tuya_cameras

In Home Assistant:  
**Settings → Devices & Services → Tuya Cameras → Configure → AI Detection**

- Enable AI detection: ✓
- Inference service URL: `http://localhost:8000`

The options flow validates `/health` before saving.

**Network exposure:** the service has no authentication, so by default it listens on
`127.0.0.1` only — which works when Home Assistant runs on the same host (host
networking or HA OS add-on style setups). If Home Assistant runs on a different
machine, bind it to this host's LAN address instead and use that address as the
inference service URL:

```bash
AI_BIND_ADDR=192.168.x.y docker compose up -d   # or put AI_BIND_ADDR=... in a .env file
```

Never forward port 8000 on your router or expose it outside your LAN.

---

## API

### `GET /health`

Returns `{"status": "ok", "model": "yolov8s"}` when the service is ready (the model name follows
`AI_MODEL`).

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

- **Model:** YOLOv8s (small) by default; YOLOv8n (nano) via `AI_MODEL=yolov8n.pt`
- **Confidence threshold:** 0.60 (hardcoded)
- **Detection class:** COCO class 0 (person)

### Why YOLOv8s

YOLOv8n was the original default. A comparison over seven days of real motion snapshots
(~2,700 motion events, 7 cameras) showed YOLOv8n regularly scoring real people just below the
0.60 threshold in hard conditions: night-time infrared images, faces covered by a motorcycle
helmet or hood, people very close to the lens or only half in frame. Counted per visit,
YOLOv8n missed 8 visits by real people that YOLOv8s caught; YOLOv8s missed 3 that YOLOv8n
caught. YOLOv8s is about 2–3× slower, which on a modern CPU is still well under a second per
motion event.

Neither model is perfect: both occasionally score static objects in empty night-time scenes
as a person, and YOLOv8s adds a few such false alarms per week on some cameras.

---

## Extending

To try another model (e.g. `yolov8m.pt`), set `AI_MODEL=yolov8m.pt`; models not baked into the
image are downloaded on first start, which needs internet access. Larger models are
noticeably slower on CPU.

To run on an NVIDIA GPU, give the container GPU access (NVIDIA Container Toolkit plus a
`deploy.resources.reservations.devices` entry in `docker-compose.yml`) and pass `device="cuda"`
in the `model(...)` call in `service/main.py`.
