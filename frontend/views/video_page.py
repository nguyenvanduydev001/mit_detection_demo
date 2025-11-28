import streamlit as st
import os
import time
import cv2 
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from ultralytics import YOLO
import google.generativeai as genai

# --- Tải biến môi trường ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception:
        pass


def show():
    # --- Kiểm tra đăng nhập ---
    if "user" not in st.session_state or not st.session_state["user"]:
        st.warning("⚠️ Bạn cần đăng nhập để sử dụng tính năng này.")
        st.info("Vui lòng chuyển sang tab **Đăng nhập** để tiếp tục.")
        st.stop()

    username = st.session_state["user"]

    # --- Kết nối MongoDB ---
    try:
        client = MongoClient(MONGO_URI)
        db = client["mit_detection"]
        logs_col = db["video_logs"]
    except Exception as e:
        st.warning(f"⚠️ Không thể kết nối MongoDB: {e}")

    st.markdown("## 🎥 Phân tích Video / Webcam")
    st.info(
        "🤖 **AgriVision** nhận dạng độ chín trái mít trực tiếp từ video hoặc webcam. "
        "Video được xử lý bằng mô hình YOLOv8, hiển thị bounding box, label và JSON realtime bên cạnh."
    )

    # --- Tải model YOLO ---
    @st.cache_resource(show_spinner="🚀 Đang tải mô hình YOLOv8...")
    def load_model():
        model_path = os.path.join(os.path.dirname(__file__), "..", "..","yolov8", "best.pt")
        return YOLO(model_path)

    model = load_model()

    # --- Cấu hình ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        source = st.radio("Nguồn dữ liệu:", ["🎞️ Video file", "📷 Webcam"], horizontal=True)
    with col2:
        conf_v = st.slider(
            "Ngưỡng Confidence", 0.1, 1.0, 0.5, 0.05,
            help="Giá trị này xác định mức độ chắc chắn của mô hình khi nhận dạng. "
                 "Càng cao thì mô hình chỉ hiển thị các đối tượng mà nó tin tưởng mạnh, "
                 "càng thấp thì mô hình hiển thị nhiều hơn nhưng dễ nhiễu."
        )

    st.markdown("---")
    if source == "📷 Webcam":
        st.session_state["video_done"] = False
        st.session_state.pop("video_json", None)
        st.session_state["video_loaded"] = False

    # ------------------- VIDEO FILE -------------------
    if source == "🎞️ Video file":
        uploaded = st.file_uploader("📁 Tải video lên (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

        if uploaded:
            if "video_loaded" not in st.session_state or not st.session_state["video_loaded"]:
                st.toast("✅ Video đã tải xong! Bấm nút dưới để bắt đầu phân tích.")
                st.session_state["video_loaded"] = True

            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(uploaded.read())
            video_path = temp_input.name
            st.video(video_path)

            if st.button("▶️ Bắt đầu phân tích video"):
                st.toast("🚀 Đang xử lý video, vui lòng đợi...")

                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()

                if not frames:
                    st.error("❌ Không thể đọc video.")
                else:
                    mid = len(frames) // 2
                    frame = cv2.resize(frames[mid], (640, 640))

                    results = model.predict(frame, conf=conf_v)
                    predictions_json = {"predictions": []}

                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        labels = results[0].names
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            label = labels.get(cls_id, "mít")
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy().astype(float)
                            x, y, w, h = xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]
                            predictions_json["predictions"].append({
                                "class": label,
                                "confidence": round(conf, 3),
                                "bbox": {"x": round(x, 3), "y": round(y, 3),
                                         "width": round(w, 3), "height": round(h, 3)}
                            })
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                                          (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)
                            label_text = f"{label} {conf:.0%}"
                            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1]-th-6)),
                                          (int(xyxy[0]+tw+4), int(xyxy[1])), (0,255,0), -1)
                            cv2.putText(frame, label_text, (int(xyxy[0]+2), int(xyxy[1]-4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="📈 Khung giữa video sau nhận dạng", use_container_width=True)

                    with st.expander("📦 Xem dữ liệu đầu vào từ hệ thống nhận dạng"):
                        st.json(predictions_json)

                    # ✅ Lưu log MongoDB
                    try:
                        counts = {}
                        for p in predictions_json["predictions"]:
                            counts[p["class"]] = counts.get(p["class"], 0) + 1

                        log_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "username": username,
                            "video_name": uploaded.name,
                            "counts": counts,
                            "total": sum(counts.values()),
                            "confidence": conf_v,
                            "source": "video",
                            "raw": predictions_json
                        }
                        logs_col.insert_one(log_entry)
                        st.toast("Đã lưu lịch sử phân tích video.", icon="✅")
                    except Exception as e:
                        st.warning(f"⚠️ Không thể lưu log vào MongoDB: {e}")

                    st.session_state["video_done"] = True
                    st.session_state["video_json"] = predictions_json
                    st.toast("✨ Phân tích hoàn tất!", duration="short")

    # ---------------- SAU KHI XỬ LÝ VIDEO ----------------
    if st.session_state.get("video_done", False):
        latest = st.session_state.get("video_json", {})
        st.markdown("---")
        st.markdown("""
        <div style='background-color:#FCFCE3; padding:15px; border-radius:10px; margin-bottom:10px;'>
            <h4 style='color:#33691E;'>💬 Phân tích video chuyên sâu bởi AgriVision</h4>
            <p style='color:#4E342E;'>AgriVision tổng hợp và đánh giá kết quả nhận dạng từ video bạn gửi.</p>
        </div>
        """, unsafe_allow_html=True)

        preds = latest.get("predictions", [])
        counts = {}
        for p in preds:
            cls = p.get("class")
            if cls:
                counts[cls] = counts.get(cls, 0) + 1
        total = sum(counts.values())

        if st.button("📊 Yêu cầu AgriVision phân tích video", use_container_width=True):
            progress = st.progress(0)
            for p in range(0, 100, 10):
                time.sleep(0.1)
                progress.progress(p)
            progress.empty()

            prompt = f"""
            Bạn là hệ thống AgriVision — nền tảng AI ứng dụng YOLOv8 trong nhận dạng và phân loại độ chín trái mít.Sau mỗi lần xử lý video, bạn sẽ tự động tạo Kết quả phân tích tổng hợp kết quả phân tích.  
            Dữ liệu đầu vào bạn vừa xử lý:
            counts={counts}, total={total}.
            Hãy viết **Kết quả phân tích  tự nhiên, gần gũi nhưng chuyên nghiệp**, thể hiện được năng lực công nghệ của hệ thống AgriVision.  
            Giọng văn giống như một kỹ sư nông nghiệp đang chia sẻ lại kết quả mà AgriVision vừa quan sát được.
            Bố cục yêu cầu:
            1) Tổng quan tình hình nhận dạng (kết quả phát hiện, tỉ lệ mít chín, non, sâu bệnh).  
            2️) Nhận xét & khuyến nghị thu hoạch (nêu rõ nên thu hay chưa, lý do, lợi ích).  
            3️) Biện pháp xử lý nếu có mít sâu bệnh (đưa hướng dẫn thực tế, dễ hiểu).  
            4️) Hỗ trợ kỹ thuật & tính năng thông minh của hệ thống (mô tả cách AgriVision giúp người dùng quản lý và chăm sóc vườn hiệu quả hơn).   
            5) Giới thiệu ngắn về vai trò của AgriVision trong việc hỗ trợ bạn theo dõi vườn qua video.  
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
                    ai_text = "⚠️ Chưa thiết lập Gemini API key."
            except Exception as e:
                ai_text = f"Lỗi khi gọi Gemini: {e}"

            st.markdown("### 🧠 Kết quả phân tích video")
            st.markdown(
                f"<div style='background-color:#FAFAFA; padding:15px; border-radius:10px; color:#212121;'>{ai_text}</div>",
                unsafe_allow_html=True
            )

    # ------------------- WEBCAM -------------------
    if source == "📷 Webcam":
        st.info("Bật webcam để AgriVision nhận dạng trực tiếp theo thời gian thực.")
        start_btn = st.button("▶️ Bắt đầu nhận dạng qua Webcam")
        stop_btn = st.button("⛔ Tắt video")

        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False

        if start_btn:
            st.session_state.webcam_running = True
            st.toast("📸 Webcam đang hoạt động!", duration="short")

        if stop_btn:
            st.session_state.webcam_running = False
            st.toast("🟥 Đã tắt webcam.", duration="short")

        frame_slot = st.empty()
        detections_all = []
        cap = cv2.VideoCapture(0)

        try:
            while st.session_state.webcam_running:
                ok, frame = cap.read()
                if not ok:
                    st.warning("⚠️ Không thể đọc khung hình từ webcam.")
                    break

                results = model.predict(frame, conf=conf_v, verbose=False)
                predictions_json = {"predictions": []}
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    labels = results[0].names
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        label = labels.get(cls_id, "mít")
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().astype(float)
                        x, y, w, h = xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]
                        predictions_json["predictions"].append({
                            "class": label,
                            "confidence": round(conf, 3),
                            "bbox": {"x": round(x, 3), "y": round(y, 3),
                                     "width": round(w, 3), "height": round(h, 3)}
                        })
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                                      (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)
                        label_text = f"{label} {conf:.0%}"
                        cv2.putText(frame, label_text, (int(xyxy[0]), int(xyxy[1]-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slot.image(frame_rgb, use_container_width=True)
                detections_all.append(predictions_json)
                time.sleep(0.05)

        finally:
            cap.release()
            frame_slot.empty()
            try:
                counts = {}
                for d in detections_all:
                    for p in d["predictions"]:
                        cls = p["class"]
                        counts[cls] = counts.get(cls, 0) + 1
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "username": username,
                    "counts": counts,
                    "total": sum(counts.values()),
                    "confidence": conf_v,
                    "source": "webcam"
                }
                logs_col.insert_one(log_entry)
            except Exception as e:
                st.warning(f"⚠️ Không thể lưu log webcam: {e}")