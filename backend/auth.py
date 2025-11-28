from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, EmailStr
from backend.mongodb_connection import get_database
from datetime import datetime
import hashlib

router = APIRouter(prefix="/auth", tags=["Auth"])

# ------------------------------
# Mô hình dữ liệu người dùng
# ------------------------------
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    confirm_password: str

class UserLogin(BaseModel):
    username: str
    password: str

# ------------------------------
# Kết nối MongoDB
# ------------------------------
db = get_database()
users_collection = db["users"]

# ------------------------------
# Hàm tiện ích
# ------------------------------
def hash_password(pw: str):
    return hashlib.sha256(pw.encode()).hexdigest()

# ------------------------------
# API Đăng ký
# ------------------------------
@router.post("/register")
def register(user: UserRegister):
    # Kiểm tra trùng username hoặc email
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Tên người dùng đã tồn tại.")
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email này đã được sử dụng.")
    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="Mật khẩu xác nhận không khớp.")

    hashed_pw = hash_password(user.password)
    new_user = {
        "username": user.username,
        "email": user.email,
        "password": hashed_pw,
        "created_at": datetime.now(),
        "last_login": None
    }

    result = users_collection.insert_one(new_user)
    if not result.inserted_id:
        raise HTTPException(status_code=500, detail="Không thể lưu người dùng vào cơ sở dữ liệu.")

    return {"message": f"Đăng ký thành công! Tài khoản '{user.username}' đã được lưu."}


@router.post("/login")
def login(user: UserLogin):
    db_user = users_collection.find_one({"username": user.username})
    if not db_user:
        raise HTTPException(status_code=404, detail="Vui lòng nhập đầy đủ thông tin.")
    if db_user["password"] != hash_password(user.password):
        raise HTTPException(status_code=401, detail="Sai mật khẩu.")

    # Cập nhật thời gian đăng nhập
    users_collection.update_one(
        {"_id": db_user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )

    return {
        "message": "Đăng nhập thành công!",
        "username": db_user["username"],
        "email": db_user["email"],
        "created_at": db_user["created_at"],
        "last_login": db_user["last_login"],
    }
# ------------------------------
# API: LẤY THÔNG TIN NGƯỜI DÙNG
# ------------------------------
@router.get("/info")
def get_user_info(username: str):
    user = users_collection.find_one({"username": username}, {"_id": 0, "password": 0})
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")

    if user.get("created_at"):
        user["created_at"] = user["created_at"].strftime("%d/%m/%Y %H:%M")
    if user.get("last_login"):
        user["last_login"] = user["last_login"].strftime("%d/%m/%Y %H:%M")

    return user

# ------------------------------
# API: CẬP NHẬT THÔNG TIN NGƯỜI DÙNG
# ------------------------------
@router.patch("/update")
def update_user_info(
    username: str,
    new_username: str = Body(None),
    email: str = Body(None),
    password: str = Body(None),
    confirm_password: str = Body(None)
):
    user = users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")

    update_data = {}

    # Đổi tên người dùng
    if new_username:
        existing_user = users_collection.find_one({"username": new_username})
        if existing_user and new_username != username:
            raise HTTPException(status_code=400, detail="Tên người dùng này đã tồn tại.")
        update_data["username"] = new_username

    # Cập nhật email (nếu có)
    if email:
        existing_email = users_collection.find_one({"email": email, "username": {"$ne": username}})
        if existing_email:
            raise HTTPException(status_code=400, detail="Email này đã được sử dụng bởi tài khoản khác.")
        update_data["email"] = email

    # Cập nhật mật khẩu (nếu có)
    if password:
        if password != confirm_password:
            raise HTTPException(status_code=400, detail="Mật khẩu xác nhận không khớp.")
        update_data["password"] = hash_password(password)

    if not update_data:
        raise HTTPException(status_code=400, detail="Không có thông tin nào để cập nhật.")

    update_data["updated_at"] = datetime.utcnow()
    result = users_collection.update_one({"username": username}, {"$set": update_data})

    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Không thể cập nhật thông tin người dùng.")

    # Nếu đổi username → trả về tên mới
    return {"message": "Cập nhật thông tin thành công.", "new_username": update_data.get("username", username)}