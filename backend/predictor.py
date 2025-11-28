from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
import uuid
import tempfile
import cv2
import numpy as np

from .model_loader import load_model
from .utils import encode_image_to_base64, decode_uploaded_file

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = load_model()

# ---------------------------------------------------------
# ROUTER WEB
# ---------------------------------------------------------
router_web = APIRouter(prefix="/predict", tags=["Web Predict"])

# ---------------------------------------------------------
# ROUTER MOBILE (APP FLUTTER)
# ---------------------------------------------------------
router_mobile = APIRouter(prefix="/mobile", tags=["Mobile Predict"])


# =========================================================
# 🟩 API WEBSITE (STREAMLIT)
# =========================================================
@router_web.post("/")
async def website_detect_image(file: UploadFile, conf: float = 0.5):
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

    annotated_np = results[0].plot()
    annotated = Image.fromarray(annotated_np)
    annotated_b64 = encode_image_to_base64(annotated)

    return {"image": annotated_b64, "detections": detections}



# =========================================================
# LABEL MAPPING TIẾNG VIỆT
# =========================================================
LABEL_MAP = {
    "mit_non": {"label": "Mít non", "status": "Cần chăm thêm"},
    "mit_chin": {"label": "Mít chín", "status": "Có thể thu hoạch"},
    "mit_saubenh": {"label": "Mít sâu bệnh", "status": "Cần xử lý sớm"},
    "mit_hu": {"label": "Mít hư", "status": "Loại bỏ"}
}



# =========================================================
# 🟦 MOBILE API — PHÂN TÍCH ẢNH
# =========================================================
@router_mobile.post("/image")
async def mobile_detect_image(file: UploadFile, conf: float = 0.5):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, "Ảnh phải là JPG hoặc PNG")

    img = decode_uploaded_file(await file.read())
    results = model.predict(img, conf=conf, verbose=False)
    result = results[0]
    boxes = result.boxes
    names = result.names

    if len(boxes) == 0:
        return {
            "label": "Không phát hiện",
            "status": "Hãy thử lại ảnh rõ hơn",
            "confidence": 0,
            "count": 0,
            "image_annotated": None
        }

    # Lấy trái có confidence cao nhất
    best_conf = -1
    best_info = None

    for b in boxes:
        cls_id = int(b.cls[0])
        raw_name = names[cls_id]
        conf_score = float(b.conf[0])

        info = LABEL_MAP.get(raw_name, {"label": raw_name, "status": "Không rõ"})
        if conf_score > best_conf:
            best_conf = conf_score
            best_info = info

    # Vẽ label tiếng Việt
    orig_names = result.names
    vn_names = {}
    for idx, raw in orig_names.items():
        mapped = LABEL_MAP.get(raw)
        vn_names[idx] = mapped["label"] if mapped else raw

    result.names = vn_names
    annotated = Image.fromarray(result.plot())
    annotated_b64 = encode_image_to_base64(annotated)
    result.names = orig_names

    return {
        "label": best_info["label"],
        "status": best_info["status"],
        "confidence": round(best_conf, 2),
        "count": len(boxes),
        "image_annotated": annotated_b64
    }



# =========================================================
# 🟦 MOBILE API — PHÂN TÍCH VIDEO
# =========================================================
@router_mobile.post("/video")
async def mobile_detect_video(file: UploadFile, conf: float = 0.5):

    if file.content_type not in ["video/mp4", "video/avi", "video/mov"]:
        raise HTTPException(400, "Video phải là MP4/AVI/MOV")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(await file.read())
    tmp_path = tmp.name
    tmp.close()

    cap = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(400, "Không đọc được video")

    results = model.predict(frame, conf=conf, verbose=False)
    result = results[0]
    boxes = result.boxes
    names = result.names

    if len(boxes) == 0:
        return {
            "label": "Không phát hiện",
            "status": "Không có trái nào",
            "confidence": 0,
            "count": 0,
            "image_annotated": None
        }

    best_conf = -1
    best_info = None

    for b in boxes:
        cls_id = int(b.cls[0])
        raw = names[cls_id]
        conf_score = float(b.conf[0])

        info = LABEL_MAP.get(raw, {"label": raw, "status": "Không rõ"})

        if conf_score > best_conf:
            best_conf = conf_score
            best_info = info

    # Vẽ tiếng Việt
    orig_names = result.names
    vn_names = {}

    for idx, raw in orig_names.items():
        mapped = LABEL_MAP.get(raw)
        vn_names[idx] = mapped["label"] if mapped else raw

    result.names = vn_names
    annotated = Image.fromarray(result.plot())
    annotated_b64 = encode_image_to_base64(annotated)

    result.names = orig_names

    return {
        "label": best_info["label"],
        "status": best_info["status"],
        "confidence": round(best_conf, 2),
        "count": len(boxes),
        "image_annotated": annotated_b64
    }



# =========================================================
# 🟦 MOBILE API — REALTIME CAMERA
# =========================================================
@router_mobile.post("/realtime")
async def mobile_realtime(file: UploadFile):

    frame_bytes = await file.read()

    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Không decode được frame")

    results = model.predict(frame, conf=0.4, verbose=False)
    result = results[0]

    # Đổi label sang tiếng Việt
    orig_names = result.names
    vn_names = {}
    for idx, raw in orig_names.items():
        mapped = LABEL_MAP.get(raw)
        vn_names[idx] = mapped["label"] if mapped else raw

    result.names = vn_names

    annotated = Image.fromarray(result.plot())
    annotated_b64 = encode_image_to_base64(annotated)

    result.names = orig_names

    return {"image_annotated": annotated_b64}
