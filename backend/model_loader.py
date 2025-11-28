from ultralytics import YOLO
from .config import MODEL_PATH

def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Không thể load YOLO model từ {MODEL_PATH}: {e}")
