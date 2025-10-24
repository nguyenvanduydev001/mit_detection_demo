from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import API_TITLE, API_DESCRIPTION, API_VERSION
from .predictor import router as predict_router

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)

@app.get("/")
def root():
    return {"message": "🚀 YOLOv8 Mit Detection API hoạt động!"}
