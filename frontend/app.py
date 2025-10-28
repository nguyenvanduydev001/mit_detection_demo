import streamlit as st
import requests
import base64
import io
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import tempfile
import base64
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception:
        # proceed: genai may still error at call time
        pass

st.set_page_config(page_title="Agri Vision - Hệ Thống Nhận Dạng Vfa Phân Loại Độ Chín Trái Mít", layout="wide")
st.markdown(
    """
    <style>
    .main-title {
        font-size: 20px;
        font-weight: 800;
        text-align: center;
        color: #2E7D32;
        margin-bottom: 0.3em;
        letter-spacing: 0.5px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #555;
        font-style: italic;
        margin-bottom: 1.5em;
    }
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, #8BC34A, #558B2F);
        margin-bottom: 1.5em;
    }
    </style>
    
    <div class="main-title">AGRI VISION — HỆ THỐNG NHẬN DẠNG VÀ PHÂN LOẠI ĐỘ CHÍN TRÁI MÍT</div>
    <p class="sub-title">Ứng dụng AI phục vụ Nông nghiệp Thông minh</p>
    <hr>
    """,
    unsafe_allow_html=True
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
API_URL = "http://127.0.0.1:8000/predict"
LATEST_RESULTS = os.path.join(os.path.dirname(__file__), "latest_results.json")

# --- Chuyển ảnh logo sang base64 để hiển thị ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
if os.path.exists(logo_path):
    logo_base64 = get_base64_of_bin_file(logo_path)
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="140" style="border-radius:10px; margin-bottom:10px"/>'
else:
    logo_html = "<div style='font-size:40px'>🍈</div>"


theme = st.get_option("theme.base")  # trả về 'dark' hoặc 'light'

if theme == "dark":
    menu_styles = {
        "container": {
            "background-color": "#1C1E24",
            "padding": "1rem",
            "border-radius": "12px",
        },
        "icon": {"color": "#FFFFFF", "font-size": "20px"},
        "nav-link": {
            "font-size": "16px",
            "color": "#FFFFFFCC",
            "text-align": "left",
            "margin": "6px 0",
            "--hover-color": "#292B33",
            "border-radius": "8px",
        },
        "nav-link-selected": {
            "background-color": "#6DBE45",
            "color": "#FFFFFF",
            "font-weight": "600",
        },
    }
else:
    menu_styles = {
        "container": {
            "background-color": "#FFFFFF",
            "padding": "1rem",
            "border-radius": "12px",
            "box-shadow": "0 2px 8px rgba(0,0,0,0.05)",
        },
        "icon": {"color": "#8EEB60", "font-size": "20px"},
        "nav-link": {
            "font-size": "16px",
            "color": "#000000CC",
            "text-align": "left",
            "margin": "6px 0",
            "--hover-color": "#E8F5E9",
            "border-radius": "8px",
        },
        "nav-link-selected": {
            "background-color": "#6DBE45",
            "color": "#FFFFFF",
            "font-weight": "600",
        },
    }
# Sidebar logo + menu
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding-bottom:10px">
             {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    choice = option_menu(
        None,
        ["Trang chủ", "Phân tích ảnh", "Video/Webcam",
         "Thống kê", "So sánh YOLOv8", "Chat AgriVision "],
        icons=["house", "camera", "camera-video", "bar-chart", "activity", "chat-dots"],
        default_index=1,
        styles=menu_styles,
    )
    #Reset session_state khi chuyển tab để tránh lỗi hiển thị chồng
    if "last_tab" not in st.session_state:
        st.session_state["last_tab"] = choice
    elif st.session_state["last_tab"] != choice:
    # Reset trạng thái chỉ khi đổi tab
        st.session_state["last_tab"] = choice
        st.session_state.pop("video_done", None)
        st.session_state.pop("video_json", None)
        st.session_state.pop("last_data", None)

if choice == "Trang chủ":
    st.markdown("""
    ### 🎯 Mục tiêu dự án
    Ứng dụng AI giúp nông dân nhận biết **độ chín của trái mít** qua hình ảnh, 
    hỗ trợ **ra quyết định thu hoạch chính xác**, giảm thất thoát, 
    hướng đến **nông nghiệp thông minh**.
    """)
    st.info("Chọn mục trong menu bên trái để bắt đầu 👉")

# ---------------- TAB 1: ẢNH ----------------
elif choice == "Phân tích ảnh":
    st.header("Phân tích ảnh")

    # === Khu vực upload và chọn ngưỡng ===
    with st.container():
        st.markdown("### 🖼️ Chọn ảnh trái mít cần phân tích")
        uploaded_file = st.file_uploader("📁 Tải ảnh lên (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        confidence = st.slider("Ngưỡng Confidence",0.1, 1.0, 0.5, 0.05,help="Giá trị này xác định mức độ chắc chắn của mô hình khi nhận dạng. "
         "Càng cao thì mô hình chỉ hiển thị các đối tượng mà nó tin tưởng mạnh, "
         "càng thấp thì mô hình hiển thị nhiều hơn nhưng dễ nhiễu.")
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Bắt đầu phân tích ảnh", use_container_width=True)

    # --- Hiển thị ảnh gốc và ảnh kết quả ngang hàng ---
    if uploaded_file:
        col1, col2 = st.columns(2)
        img = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.markdown("**Ảnh gốc**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("**Ảnh kết quả nhận dạng**")
            out_image = st.empty()

    # === Khi nhấn nút "Phân tích ảnh" ===
    if analyze_btn and uploaded_file:
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Đang xử lý ảnh, vui lòng chờ trong giây lát...")
        progress = st.progress(0)
        files = {"file": uploaded_file.getvalue()}

        try:
            for percent in range(0, 80, 10):
                time.sleep(0.1)
                progress.progress(percent)

            resp = requests.post(API_URL, files=files, params={"conf": confidence}, timeout=30)
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
        st.success("✨ Phân tích hoàn tất!")

        # --- Hiển thị kết quả ---
        if data:
            img_data = base64.b64decode(data["image"])
            annotated = Image.open(io.BytesIO(img_data)).convert("RGB")
            st.session_state.last_data = data
            st.session_state.last_img = annotated
            from datetime import datetime
            log_path = os.path.join(os.path.dirname(__file__), "analysis_log.json")

            # Đếm số lượng từng loại mít trong lần xử lý này
            preds = data.get("detections", []) or data.get("predictions", [])
            counts = {}
            for p in preds:
                cls = p.get("class") or p.get("label")
                if cls:
                    counts[cls] = counts.get(cls, 0) + 1

            # Đọc log cũ nếu có
            logs = []
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                except Exception:
                    logs = []

            # Thêm dữ liệu mới
            for cls_name, cnt in counts.items():
                logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "class": cls_name,
                    "count": cnt
                })

            # Lưu lại file
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)


            # cập nhật ảnh kết quả bên phải
            out_image.image(annotated, use_container_width=True)

            detections = data.get("detections", [])
            if not detections:
                st.warning("⚠️ Không phát hiện được trái mít nào.")
            else:
                df = pd.DataFrame(detections)
                row_df = (
                    df[["label", "confidence"]]
                    .rename(columns={"label": "Loại", "confidence": "Độ tin cậy"})
                    .copy()
                )

                row_df["Độ tin cậy"] = row_df["Độ tin cậy"].map(lambda x: f"{x:.2f}")


                st.markdown("---")
                st.markdown("### 📊 Kết quả nhận dạng")
                st.dataframe(
                    row_df.style.set_properties(**{
                        'text-align': 'center',
                        'font-size': '16px'
                    })
                )

    # === PHẦN 2: Phân tích AI chuyên sâu ===
    if "last_data" in st.session_state:
        st.markdown("---")
        st.markdown("""
        <div style='background-color:#F9FBE7; padding:15px; border-radius:10px;'>
            <h4 style='color:#33691E;'>🧠 Phân tích ảnh chuyên sâu bởi AgriVision</h4>
            <p style='color:#4E342E;'>AI hỗ trợ đánh giá độ chín, sâu bệnh và khuyến nghị thu hoạch.</p>
        </div>
        """, unsafe_allow_html=True)

        def summarize_counts_from_latest(latest: dict):
            preds = latest.get("predictions")
            if isinstance(preds, list):
                counts = {}
                for p in preds:
                    cls = p.get("class")
                    if cls:
                        counts[cls] = counts.get(cls, 0) + 1
                total = sum(counts.values())
                return counts, total
            counts = latest.get("counts", {}) or {}
            total = latest.get("total", sum(counts.values()))
            return counts, total

        if os.path.exists(LATEST_RESULTS):
            with open(LATEST_RESULTS, "r", encoding="utf-8") as f:
                last = json.load(f)
            
            st.markdown("<div style='margin-top:15px'></div>", unsafe_allow_html=True)
            with st.expander("📦 Xem dữ liệu đầu vào từ hệ thống nhận dạng"):
                st.json(last)

            counts, total = summarize_counts_from_latest(last)

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
                        raise RuntimeError("No GEMINI key")
                except Exception as e:
                    ai_text = None
                    st.error(f"Không gọi được Gemini (fallback). Lỗi: {e}")

                progress.empty()
                status_placeholder.empty()
                st.success("✨ Phân tích hoàn tất!")

                if not ai_text:
                    lines = ["Báo cáo phân tích (fallback):"]
                    if total == 0:
                        lines.append("- Không phát hiện trái mít nào trong ảnh.")
                    else:
                        for k, v in counts.items():
                            pct = (v / total) * 100 if total > 0 else 0
                            lines.append(f"- {k}: {v} trái ({pct:.1f}%)")
                    ai_text = "\n".join(lines)

                st.markdown("### 📑 Kết quả phân tích AI")
                st.markdown(
                    f"<div style='background-color:#FAFAFA; padding:15px; border-radius:10px; color:#212121;'>{ai_text}</div>",
                    unsafe_allow_html=True
                )

# ---------------- TAB 2: VIDEO / WEBCAM ----------------
elif choice == "Video/Webcam":
    import time, json, tempfile, os, cv2
    from ultralytics import YOLO

    st.markdown("## 🎥 Phân tích Video / Webcam")
    st.info(
        "🤖 **AgriVision** nhận dạng độ chín trái mít trực tiếp từ video hoặc webcam. "
        "Video được xử lý bằng mô hình YOLOv8, hiển thị bounding box, label và JSON realtime bên cạnh."
    )

    # --- Tải model ---
    @st.cache_resource(show_spinner="🚀 Đang tải mô hình YOLOv8...")
    def load_model():
        model_path = os.path.join(os.path.dirname(__file__), "..", "yolov8", "best.pt")
        return YOLO(model_path)

    model = load_model()

    # --- Cấu hình ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        source = st.radio("Nguồn dữ liệu:", ["🎞️ Video file", "📷 Webcam"], horizontal=True)
    with col2:
        conf_v = st.slider("Ngưỡng Confidence",0.1, 1.0, 0.5, 0.05,
        help="Giá trị này xác định mức độ chắc chắn của mô hình khi nhận dạng. "
         "Càng cao thì mô hình chỉ hiển thị các đối tượng mà nó tin tưởng mạnh, "
         "càng thấp thì mô hình hiển thị nhiều hơn nhưng dễ nhiễu."
)

    st.markdown("---")
    if source == "📷 Webcam":
        st.session_state["video_done"] = False
        st.session_state.pop("video_json", None)


    # ------------------- VIDEO FILE -------------------
    if source == "🎞️ Video file":
        uploaded = st.file_uploader("📁 Tải video lên (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

        if uploaded:
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(uploaded.read())
            video_path = temp_input.name

            st.video(video_path)
            st.success("✅ Video đã tải xong! Bấm nút dưới để bắt đầu phân tích.")

            if st.button("▶️ Bắt đầu phân tích video"):
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24

                video_col, json_col = st.columns([3, 2])
                frame_slot = video_col.empty()
                json_box = json_col.empty()
                detections_all = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.track(frame, conf=conf_v, persist=True, tracker="bytetrack.yaml")
                    predictions_json = {"predictions": []}

                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        labels = results[0].names
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            label = labels.get(cls_id, "mít")
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy().astype(float)
                            x, y, w, h = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]

                            predictions_json["predictions"].append({
                                "class": label,
                                "confidence": round(conf, 3),
                                "bbox": {
                                    "x": round(x, 3),
                                    "y": round(y, 3),
                                    "width": round(w, 3),
                                    "height": round(h, 3)
                                }
                            })

                            # Vẽ khung và nhãn
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                                          (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                            label_text = f"{label} {conf:.0%}"
                            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1] - th - 6)),
                                          (int(xyxy[0] + tw + 4), int(xyxy[1])), (0, 255, 0), -1)
                            cv2.putText(frame, label_text, (int(xyxy[0] + 2), int(xyxy[1] - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # Hiển thị realtime
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_slot.image(frame_rgb, use_container_width=True)

                    # JSON realtime bên phải
                    json_html = f"""
                    <div style='background-color:#f9fafb;padding:10px;border-radius:10px;
                    border:1px solid #e3e3e3;height:308px;overflow-y:auto;
                    font-family:monospace;font-size:13px;white-space:pre;'>
                        {json.dumps(predictions_json, indent=2, ensure_ascii=False)}
                    </div>
                    """
                    json_box.markdown(json_html, unsafe_allow_html=True)
                    detections_all.append(predictions_json)

                cap.release()

                # Lưu kết quả cuối cùng
                st.session_state["video_done"] = True
                st.session_state["video_json"] = detections_all[-1] if detections_all else {}
    # ---------------- SAU KHI XỬ LÝ XONG VIDEO ----------------
    if st.session_state.get("video_done", False):
        latest = st.session_state.get("video_json", {})

        st.markdown("---")
        st.markdown("""
        <div style='background-color:#FCFCE3; padding:15px; border-radius:10px; margin-bottom:10px;'>
            <h4 style='color:#33691E;'>💬 Phân tích video chuyên sâu bởi AgriVision</h4>
            <p style='color:#4E342E;'>AgriVision tổng hợp và đánh giá kết quả nhận dạng từ video bạn gửi.</p>
        </div>
        """, unsafe_allow_html=True)

        def summarize_video_data(data):
            preds = data.get("predictions", [])
            counts = {}
            for p in preds:
                cls = p.get("class")
                if cls:
                    counts[cls] = counts.get(cls, 0) + 1
            total = sum(counts.values())
            return counts, total

        counts, total = summarize_video_data(latest)

        if st.button("📊 Yêu cầu AgriVision phân tích video", use_container_width=True):
            status = st.empty()
            progress = st.progress(0)
            status.info("🤖 AgriVision đang phân tích dữ liệu video...")
            for p in range(0, 100, 10):
                time.sleep(0.1)
                progress.progress(p)
            progress.empty()
            status.empty()

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
                    ai_text = "Phân tích thủ công: AgriVision chưa kích hoạt Gemini API."
            except Exception:
                ai_text = "Không thể gọi Gemini API, hiển thị kết quả tóm tắt thay thế."

            st.markdown("### 🧠 Kết quả phân tích video")
            st.markdown(
                f"<div style='background-color:#FAFAFA; padding:15px; border-radius:10px; color:#212121;'>{ai_text}</div>",
                unsafe_allow_html=True
            )
    
    # ------------------- WEBCAM (CHUẨN HIỂN THỊ) -------------------
    if source == "📷 Webcam":
        st.info("Bật webcam của bạn để AgriVision nhận dạng trái mít theo thời gian thực.")
        run = st.checkbox("▶️ Bắt đầu nhận dạng qua Webcam", value=False)

        video_col, json_col = st.columns([3, 2])
        frame_slot = video_col.empty()
        json_box = json_col.empty()

        detections_all = []
        cap = cv2.VideoCapture(0)

        if run:
            st.warning("⏹ Dừng nhận dạng bằng cách bỏ chọn checkbox.")
            while True:
                ret, frame = cap.read()
                if not ret or not run:
                    break

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
                        x, y, w, h = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]

                        predictions_json["predictions"].append({
                            "class": label,
                            "confidence": round(conf, 3),
                            "bbox": {
                                "x": round(x, 3),
                                "y": round(y, 3),
                                "width": round(w, 3),
                                "height": round(h, 3)
                            }
                        })

                        # Vẽ bounding box và label
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                                    (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        label_text = f"{label} {conf:.0%}"
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1] - th - 6)),
                                    (int(xyxy[0] + tw + 4), int(xyxy[1])), (0, 255, 0), -1)
                        cv2.putText(frame, label_text, (int(xyxy[0] + 2), int(xyxy[1] - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slot.image(frame_rgb, use_container_width=True)

                # Hiển thị JSON realtime bên phải
                formatted_json = json.dumps(predictions_json, indent=2, ensure_ascii=False)
                json_box.markdown(
                    f"""
                    <div style="background-color:#f9fafb;padding:10px;border-radius:10px;
                    border:1px solid #e3e3e3;height:411px;overflow-y:auto;
                    font-family:monospace;font-size:13px;white-space:pre;">
                    {formatted_json}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                detections_all.append(predictions_json)
                time.sleep(0.05)  # để Streamlit kịp render lại UI

            cap.release()
            st.success("🟢 Webcam đã dừng.")

# ---------------- TAB 3: THỐNG KÊ & QUẢN LÝ VƯỜN ----------------
elif choice == "Thống kê":

    st.markdown("## AgriVision – Thống kê & Theo dõi vườn mít")
    st.markdown("""
    Xin chào 👋  
    Đây là bảng điều khiển thông minh của **AgriVision**, nơi bạn có thể xem lại tình hình vườn mít của mình.  
    Hệ thống tổng hợp kết quả nhận dạng, phân tích tỷ lệ mít **chín – non – sâu bệnh**,  
    và đưa ra **gợi ý hành động thực tế** giúp bạn quản lý vườn hiệu quả hơn 🌱
    """)
    st.divider()

    # ======================= TỔNG QUAN NHẬN DẠNG =========================
    st.subheader("Tổng quan nhận dạng mới nhất")

    latest_path = os.path.join(os.path.dirname(__file__), "latest_results.json")
    counts, total = {}, 0

    if os.path.exists(latest_path):
        with open(latest_path, "r", encoding="utf-8") as f:
            latest = json.load(f)
        preds = latest.get("predictions", [])
        for p in preds:
            cls = p.get("class")
            if cls:
                counts[cls] = counts.get(cls, 0) + 1
        total = sum(counts.values())

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số trái phát hiện", total)
    col2.metric("✅ Mít chín", counts.get("mit_chin", 0))
    col3.metric("⚠️ Mít sâu bệnh", counts.get("mit_saubenh", 0))

    if total > 0:
        st.caption(f"Cập nhật lúc {datetime.now().strftime('%H:%M – %d/%m/%Y')}")
        df_counts = pd.DataFrame(list(counts.items()), columns=["Loại", "Số lượng"])
        fig, ax = plt.subplots()
        colors = ["#7FC97F", "#FDBF6F", "#E31A1C"]
        ax.pie(df_counts["Số lượng"], labels=df_counts["Loại"], autopct="%1.1f%%", startangle=90, colors=colors)
        ax.set_title("Tỷ lệ các loại mít trong vườn", fontsize=12)
        fig.set_size_inches(4, 4)  
        st.pyplot(fig)
    else:
        st.info("💡 Chưa có dữ liệu nhận dạng gần đây. Hãy tải video hoặc bật webcam để cập nhật vườn nhé.")

    st.divider()

    # ======================= HOẠT ĐỘNG GẦN ĐÂY =========================
    st.subheader("📅 Nhật ký hoạt động nhận dạng")

    history_path = os.path.join(os.path.dirname(__file__), "analysis_log.json")
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        if isinstance(history, list) and len(history) > 0:
            df_hist = pd.DataFrame(history)
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])

            fig, ax = plt.subplots(figsize=(4, 2.5))  # 🔹 thu nhỏ biểu đồ

            for cls_name, group in df_hist.groupby("class"):
                ax.plot(group["timestamp"], group["count"], marker="o", label=cls_name, linewidth=2)

            ax.legend(fontsize=8, loc="upper left", frameon=False)
            ax.set_ylabel("Số lượng phát hiện", fontsize=9)
            ax.set_title("Xu hướng nhận dạng mít theo thời gian", fontsize=11)

            # ❌ Ẩn trục thời gian
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

            st.pyplot(fig)
        else:
            st.caption("📂 Chưa có lịch sử nhận dạng được lưu.")
    else:
        st.caption("📁 Hệ thống chưa ghi nhận dữ liệu trước đó.")

    st.divider()

    # ======================= GỢI Ý & CẢNH BÁO =========================
    st.subheader("🧠 Gợi ý & cảnh báo từ AgriVision")

    if total > 0:
        chin = counts.get("mit_chin", 0)
        non = counts.get("mit_non", 0)
        sau = counts.get("mit_saubenh", 0)
        ratio_chin = chin / total if total else 0
        ratio_non = non / total if total else 0

        if ratio_chin >= 0.6:
            st.success("🌤️ **Thu hoạch sắp tới!** Tỷ lệ mít chín cao, bạn nên chuẩn bị bao trái và lên kế hoạch thu trong vài ngày tới.")
        elif ratio_non >= 0.6:
            st.info("🕓 **Chưa vội thu hoạch:** Phần lớn trái vẫn còn non, hãy chờ thêm 3–5 ngày để đạt chất lượng tốt nhất.")
        elif sau > 0:
            st.warning("🚨 **Phát hiện sâu bệnh:** Có một số trái bị hư hại, nên tách riêng và xử lý sớm để tránh lây lan sang cây khác.")
        else:
            st.info("📊 Hệ thống chưa đủ dữ liệu để đưa ra khuyến nghị chi tiết.")
    else:
        st.caption("Vui lòng chạy nhận dạng trước để kích hoạt phân tích tự động.")

    st.divider()

    # ======================= BÁO CÁO NHANH =========================
    st.subheader("📋 Báo cáo nhanh vườn hôm nay")

    if total > 0:
        ratio_chin = (counts.get("mit_chin", 0) / total) * 100 if total else 0
        ratio_non = (counts.get("mit_non", 0) / total) * 100 if total else 0
        ratio_sau = (counts.get("mit_saubenh", 0) / total) * 100 if total else 0

        st.markdown(f"""
        **Tổng hợp:**  
        - Tổng trái được phát hiện: **{total}**  
        - 🍈 Mít chín: **{ratio_chin:.1f}%**  
        - 🌱 Mít non: **{ratio_non:.1f}%**  
        - 🐛 Mít sâu bệnh: **{ratio_sau:.1f}%**

        **Đánh giá chung:**  
        - Vườn đang ở giai đoạn **{'chín rộ' if ratio_chin > 50 else 'phát triển'}**.  
        - AgriVision sẽ tiếp tục theo dõi để gợi ý thời điểm thu hoạch phù hợp nhất.  
        """)
    else:
        st.caption("Chưa có dữ liệu đủ để lập báo cáo nhanh.")


# ---------------- TAB 4: So sánh ----------------
elif choice == "So sánh YOLOv8":
    st.header("So sánh YOLOv8n vs YOLOv8s")
    path_n = os.path.join(os.path.dirname(__file__), "..", "yolov8", "results_n.csv")
    path_s = os.path.join(os.path.dirname(__file__), "..", "yolov8", "results_s.csv")
    if os.path.exists(path_n) and os.path.exists(path_s):
        df_n = pd.read_csv(path_n)
        df_s = pd.read_csv(path_s)
        metrics = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)"]
        for m in metrics:
            if m in df_n.columns and m in df_s.columns:
                fig, ax = plt.subplots()
                ax.plot(df_n.index, df_n[m], label="n")
                ax.plot(df_s.index, df_s[m], label="s")
                ax.set_title(m)
                ax.legend()
                st.pyplot(fig)
        st.dataframe({
            "YOLOv8n (last)": df_n.iloc[-1].to_dict(),
            "YOLOv8s (last)": df_s.iloc[-1].to_dict()
        })
    else:
        st.warning("Thiếu results_n.csv / results_s.csv trong yolov8/ — export từ quá trình training.")

# ---------------- TAB 5: CHAT (Gemini) ----------------
elif choice == "Chat AgriVision ":
    st.header("Trợ lý nông nghiệp thông minh - AgriVision")
    st.subheader("Trao đổi về mô hình YOLOv8, độ chín trái mít, hoặc kỹ thuật nông nghiệp.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        if role=="user":
            st.write(f"**Bạn:** {content}")
        else:
            st.write(f"**AI:** {content}")

    user_q = st.text_input("Nhập câu hỏi", "")
    if st.button("Gửi câu hỏi"):
        if user_q.strip():
            st.session_state.chat_history.append({"role":"user","content":user_q})
            try:
                if GEMINI_KEY:
                    model = genai.GenerativeModel("models/gemini-2.5-flash")
                    resp = model.generate_content(user_q)
                    answer = getattr(resp, "text", None) or str(resp)
                else:
                    raise RuntimeError("No GEMINI key")
            except Exception as e:
                answer = f"[Fallback trả lời tự động] Không thể gọi Gemini: {e}"
            st.session_state.chat_history.append({"role":"assistant","content":answer})
