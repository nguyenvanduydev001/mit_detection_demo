from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import API_TITLE, API_DESCRIPTION, API_VERSION
from .predictor import router_web, router_mobile
from .auth import router as auth_router


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

app.include_router(router_web)      # API cho website
app.include_router(router_mobile)   # API cho app mobile
app.include_router(auth_router)

@app.get("/")
def root():
    return {"message": "🚀 YOLOv8 Mit Detection API hoạt động!"}
