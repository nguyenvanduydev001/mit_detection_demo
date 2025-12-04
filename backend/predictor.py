from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
import uuid

from .model_loader import load_model
from .utils import encode_image_to_base64, decode_uploaded_file

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = load_model()

# ---------------------------------------------------------
# ROUTER WEB ONLY
# ---------------------------------------------------------
router_web = APIRouter(prefix="/predict", tags=["Web Predict"])

# =========================================================
# 🟩 API WEBSITE (STREAMLIT)
# =========================================================
@router_web.post("/")
async def website_detect_image(file: UploadFile, conf: float = 0.5):
    """
    API duy nhất còn lại: nhận ảnh từ Streamlit/Web để detect trái mít.
    """
    # Decode ảnh
    try:
        img = decode_uploaded_file(await file.read())
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh tải lên.")

    # YOLO Predict
    try:
        results = model.predict(source=img, conf=conf, save=False, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi YOLO: {e}")

    # Lấy kết quả
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

    # Vẽ bounding box
    annotated_np = results[0].plot()
    annotated = Image.fromarray(annotated_np)
    annotated_b64 = encode_image_to_base64(annotated)

    return {
        "image": annotated_b64,
        "detections": detections,
        "predictions": predictions,
    }
