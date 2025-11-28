import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "yolov8", "best.pt")

API_TITLE = "YOLOv8 Mit Detection API"
API_DESCRIPTION = "API nhận dạng & phân loại độ chín trái mít 🌾"
API_VERSION = "1.0.0"
