import streamlit as st
import os
import io
import time
import base64
import requests
import pandas as pd
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import google.generativeai as genai

# === KHỞI TẠO & CẤU HÌNH ===
load_dotenv()
API_PREDICT = os.getenv("API_PREDICT", "http://127.0.0.1:8000/predict")
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception:
        pass


def show():
    # ✅ Kiểm tra đăng nhập
    if "user" not in st.session_state or not st.session_state["user"]:
        st.warning("⚠️ Bạn cần đăng nhập để sử dụng tính năng này.")
        st.info("Vui lòng chuyển sang tab **Đăng nhập** để tiếp tục.")
        st.stop()

    st.header("📸 Phân tích ảnh")

    # --- Kết nối MongoDB ---
    client = MongoClient(MONGO_URI)
    db = client["mit_detection"]
    logs_col = db["analysis_logs"]

    # === Upload ảnh ===
    with st.container():
        st.markdown("### 🖼️ Chọn ảnh trái mít cần phân tích")
        uploaded_file = st.file_uploader("📁 Tải ảnh lên (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        confidence = st.slider(
            "Ngưỡng Confidence", 0.1, 1.0, 0.5, 0.05,
            help="Giá trị này xác định mức độ chắc chắn của mô hình khi nhận dạng."
        )
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Bắt đầu phân tích ảnh", use_container_width=True)

    # === Hiển thị ảnh gốc ===
    if uploaded_file:
        col1, col2 = st.columns(2)
        img = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.markdown("**Ảnh gốc**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("**Ảnh kết quả nhận dạng**")
            out_image = st.empty()

    # === Gửi ảnh tới API ===
    if analyze_btn:
        if not uploaded_file:
            st.warning("⚠️ Vui lòng tải ảnh lên trước khi phân tích.")
        else:
            status_placeholder = st.empty()
            status_placeholder.info("⏳ Đang xử lý ảnh, vui lòng chờ trong giây lát...")
            progress = st.progress(0)
            files = {"file": uploaded_file.getvalue()}

            try:
                for percent in range(0, 80, 10):
                    time.sleep(0.1)
                    progress.progress(percent)

                resp = requests.post(API_PREDICT, files=files, params={"conf": confidence}, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                for percent in range(80, 101, 10):
                    time.sleep(0.1)
                    progress.progress(percent)

            except Exception as e:
                st.error(f"Lỗi gọi API: {e}")
                data = None

            progress.empty()
            status_placeholder.empty()
            st.toast("✨ Phân tích hoàn tất!", duration="short")

            # --- Hiển thị kết quả ---
            if data:
                img_data = base64.b64decode(data["image"])
                annotated = Image.open(io.BytesIO(img_data)).convert("RGB")
                st.session_state.last_data = data
                st.session_state.last_img = annotated

                # --- Lưu log Mongo ---
                preds = data.get("detections", []) or data.get("predictions", [])
                counts = {}
                for p in preds:
                    cls = p.get("class") or p.get("label")
                    if cls:
                        counts[cls] = counts.get(cls, 0) + 1

                username = st.session_state["user"]
                for p in preds:
                    p.pop("_id", None)

                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "username": username,
                    "counts": counts,
                    "total": sum(counts.values()),
                    "confidence": confidence,
                    "file_name": uploaded_file.name,
                    "raw": preds
                }

                try:
                    logs_col.insert_one(log_entry)
                    st.toast("Đã lưu kết quả vào lịch sử.", icon="✅", duration="short")
                except Exception as e:
                    st.warning(f"⚠️ Không thể lưu log vào MongoDB: {e}")

                out_image.image(annotated, use_container_width=True)

                detections = data.get("detections", [])
                if not detections:
                    st.warning("⚠️ Không phát hiện được trái mít nào.")
                else:
                    df = pd.DataFrame(detections)
                    row_df = df[["label", "confidence"]].rename(columns={"label": "Loại", "confidence": "Độ tin cậy"})
                    row_df["Độ tin cậy"] = row_df["Độ tin cậy"].map(lambda x: f"{x:.2f}")

                    st.markdown("---")
                    st.markdown("### 📊 Kết quả nhận dạng")
                    st.dataframe(row_df, use_container_width=True)

                    st.download_button(
                        "⬇️ Tải ảnh kết quả",
                        data=io.BytesIO(img_data),
                        file_name=f"ket_qua_{uploaded_file.name}",
                        mime="image/jpeg"
                    )

    # === Phân tích AI chuyên sâu ===
    if "last_data" in st.session_state:
        st.markdown("---")
        st.markdown("""
        <div style='background-color:#F9FBE7; padding:15px; border-radius:10px; margin-bottom: 10px;'>
            <h4 style='color:#33691E;'>🧠 Phân tích ảnh chuyên sâu bởi AgriVision</h4>
            <p style='color:#4E342E;'>AI hỗ trợ đánh giá độ chín, sâu bệnh và khuyến nghị thu hoạch.</p>
        </div>
        """, unsafe_allow_html=True)

        latest = st.session_state.last_data
        preds = latest.get("detections", []) or latest.get("predictions", [])
        counts = {}
        for p in preds:
            cls = p.get("class") or p.get("label")
            if cls:
                counts[cls] = counts.get(cls, 0) + 1
        total = sum(counts.values())

        if st.button("📊 Yêu cầu AgriVision phân tích ảnh", use_container_width=True):
            status_placeholder = st.empty()
            status_placeholder.info("🤖 AgriVision đang phân tích dữ liệu từ hình ảnh, vui lòng chờ...")
            progress = st.progress(0)
            for p in range(0, 100, 10):
                time.sleep(0.1)
                progress.progress(p)

            prompt = f"""
               Bạn là hệ thống AgriVision — nền tảng AI ứng dụng YOLOv8 trong nhận dạng và phân loại độ chín trái mít.Sau mỗi lần xử lý hình ảnh, bạn sẽ tự động tạo Kết quả phân tích tổng hợp kết quả phân tích.  
               Dữ liệu đầu vào bạn vừa xử lý:
               counts={counts}, total={total}.
               Hãy viết **Kết quả phân tích  tự nhiên, gần gũi nhưng chuyên nghiệp**, thể hiện được năng lực công nghệ của hệ thống AgriVision.  
               Giọng văn giống như một kỹ sư nông nghiệp đang chia sẻ lại kết quả mà AgriVision vừa quan sát được.
               Bố cục yêu cầu:
               1) Tổng quan tình hình nhận dạng (kết quả phát hiện, tỉ lệ mít chín, non, sâu bệnh).  
               2️) Nhận xét & khuyến nghị thu hoạch (nêu rõ nên thu hay chưa, lý do, lợi ích).  
               3️) Biện pháp xử lý nếu có mít sâu bệnh (đưa hướng dẫn thực tế, dễ hiểu).  
               4️) Hỗ trợ kỹ thuật & tính năng thông minh của hệ thống (mô tả cách AgriVision giúp người dùng quản lý và chăm sóc vườn hiệu quả hơn).   
               5) Giới thiệu ngắn về vai trò của AgriVision trong việc hỗ trợ bạn theo dõi vườn qua hình ảnh.
                
               Phong cách viết:
               - Mở đầu bằng lời chào: “Chào bạn, tôi là AgriVision – người bạn đồng hành trong vườn mít.”  
               - Ngôn từ thân thiện, rõ ràng, không rườm rà.  
            """

            ai_text = None
            try:
                if GEMINI_KEY:
                    model = genai.GenerativeModel("models/gemini-2.5-flash")
                    resp = model.generate_content(prompt)
                    ai_text = getattr(resp, "text", None) or str(resp)
                else:
                    raise RuntimeError("Không có khóa Gemini API")
            except Exception as e:
                ai_text = f"Lỗi khi gọi Gemini: {e}"

            progress.empty()
            status_placeholder.empty()
            st.toast("✨ Phân tích hoàn tất!", duration="short")

            st.markdown("### 📑 Kết quả phân tích AI")
            st.markdown(
                f"<div style='background-color:#FAFAFA; padding:15px; border-radius:10px; color:#212121;'>{ai_text}</div>",
                unsafe_allow_html=True
            )
