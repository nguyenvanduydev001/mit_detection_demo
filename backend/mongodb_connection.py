from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Tải các biến môi trường từ file .env
load_dotenv()

def get_database():
    """
    Kết nối tới MongoDB Atlas và trả về database 'mit_detection'.
    """
    MONGO_URI = os.getenv("MONGO_URI")

    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        print("✅ Kết nối MongoDB Atlas thành công!")
        return client["mit_detection"]
    except Exception as e:
        print("❌ Lỗi kết nối MongoDB:", e)
        raise e
