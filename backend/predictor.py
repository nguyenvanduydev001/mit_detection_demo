from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
import uuid
import pandas as pd
import json

from .model_loader import load_model
from .utils import encode_image_to_base64, decode_uploaded_file
from .config import SAVE_PATH

router = APIRouter(prefix="/predict", tags=["Predict"])
model = load_model()

@router.post("/")
async def predict_image(file: UploadFile, conf: float = 0.5):
    try:
        img = decode_uploaded_file(await file.read())
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh tải lên.")

    try:
        results = model.predict(source=img, conf=conf, save=False, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi YOLO: {e}")

    boxes = results[0].boxes
    predictions = []
    detections = []

    for b in boxes:
        xywh = b.xywh[0].tolist()
        class_id = int(b.cls[0])
        label = results[0].names[class_id]
        conf_score = float(b.conf[0])

        detections.append({
            "label": label,
            "confidence": round(conf_score, 3)
        })

        # JSON format 
        predictions.append({
            "x": float(xywh[0]),
            "y": float(xywh[1]),
            "width": float(xywh[2]),
            "height": float(xywh[3]),
            "confidence": conf_score,
            "class": label,
            "class_id": class_id,
            "detection_id": str(uuid.uuid4())
        })

    # tạo ảnh annotate
    annotated_np = results[0].plot()
    annotated = Image.fromarray(annotated_np)
    annotated_b64 = encode_image_to_base64(annotated)

    # lưu JSON 
    result_json = {"predictions": predictions}
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    return {"image": annotated_b64, "detections": detections}
