# backend/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import os
from datetime import datetime

app = FastAPI(title="Mit Detection API")

MODEL_PATH = os.path.join("..", "yolov8", "best.pt")  # tương đối từ backend/ -> yolov8/best.pt
if not os.path.exists(MODEL_PATH):
    # Thử đường dẫn trực tiếp nếu bạn chạy từ root
    MODEL_PATH = os.path.join("yolov8", "best.pt")

model = YOLO(MODEL_PATH)

LATEST_RESULTS = os.path.join(os.path.dirname(__file__), "..", "frontend", "latest_results.json")

@app.get("/")
def root():
    return {"message": "API YOLOv8 nhận dạng độ chín trái mít đang hoạt động"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.5):
    # Đọc dữ liệu ảnh từ request
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)

    # Chạy mô hình
    results = model.predict(source=img_array, conf=conf)

    # Lấy kết quả nhận dạng
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf_score = float(box.conf[0])
        # Lấy bbox coords
        xyxy = box.xyxy[0].tolist() if hasattr(box, "xyxy") else None
        detections.append({
            "label": label,
            "confidence": round(conf_score, 3),
            "bbox": xyxy
        })

    # Xuất ảnh có khung
    annotated_img = results[0].plot()  # numpy BGR
    # Nếu là RGB numpy, đảm bảo chuyển; ultralytics vẽ BGR
    _, buffer = cv2.imencode(".jpg", annotated_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Lưu latest_results.json cho frontend đọc vào AI Insight
    try:
        latest = {
            "timestamp": datetime.utcnow().isoformat(),
            "detections": detections,
            "counts": {}
        }
        # tổng hợp counts
        for d in detections:
            latest["counts"][d["label"]] = latest["counts"].get(d["label"], 0) + 1
        os.makedirs(os.path.dirname(LATEST_RESULTS), exist_ok=True)
        with open(LATEST_RESULTS, "w", encoding="utf-8") as f:
            json.dump(latest, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # không crash nếu lưu thất bại

    return JSONResponse(content={
        "detections": detections,
        "image": img_base64
    })
