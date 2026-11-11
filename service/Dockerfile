FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download both models so the service starts without internet access and
# switching via AI_MODEL needs no rebuild (yolov8s ~22 MB default, yolov8n ~6 MB)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); YOLO('yolov8n.pt')"

COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
