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
import google.generativeai as genai
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

st.set_page_config(page_title="Nhận dạng và phân loại độ chín trái mít", layout="wide")
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
    <p class="sub-title">Ứng dụng AI phục vụ Nông nghiệp Thông minh 🌾</p>
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
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="100" style="border-radius:10px; margin-bottom:10px"/>'
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
            <h3 style="margin:0; color:#6DBE45;">🌿 AgriVision</h3>
            <p style="font-size:13px; color:gray;">Hệ thống nhận dạng & phân loại độ chín trái mít</p>
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
        uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", "jpeg", "png"])
        confidence = st.slider("Ngưỡng confidence", 0.1, 1.0, 0.5, 0.05)
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
            <h4 style='color:#33691E;'>🧠 Phân tích chuyên sâu bởi AgriVision</h4>
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

            if st.button("📊 Yêu cầu AgriVision phân tích", use_container_width=True):
                status_placeholder = st.empty()
                status_placeholder.info("🤖 AgriVision đang phân tích dữ liệu, vui lòng chờ...")
                progress = st.progress(0)

                for p in range(0, 100, 10):
                    time.sleep(0.1)
                    progress.progress(p)

                prompt = f"""
                Bạn là hệ thống AgriVision — nền tảng AI ứng dụng YOLOv8 trong nhận dạng và phân loại độ chín trái mít.  
                Sau mỗi lần xử lý hình ảnh, bạn sẽ tự động tạo Kết quả phân tích tổng hợp kết quả phân tích.  
                Dữ liệu đầu vào bạn vừa xử lý:
                counts={counts}, total={total}.
                Hãy viết **Kết quả phân tích  tự nhiên, gần gũi nhưng chuyên nghiệp**, thể hiện được năng lực công nghệ của hệ thống AgriVision.  
                Giọng văn giống như một kỹ sư nông nghiệp đang chia sẻ lại kết quả mà AgriVision vừa quan sát được.
                Bố cục yêu cầu:
                1) Tổng quan tình hình nhận dạng (kết quả phát hiện, tỉ lệ mít chín, non, sâu bệnh).  
                2️) Nhận xét & khuyến nghị thu hoạch (nêu rõ nên thu hay chưa, lý do, lợi ích).  
                3️) Biện pháp xử lý nếu có mít sâu bệnh (đưa hướng dẫn thực tế, dễ hiểu).  
                4️) Hỗ trợ kỹ thuật & tính năng thông minh của hệ thống (mô tả cách AgriVision giúp người dùng quản lý và chăm sóc vườn hiệu quả hơn).   

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
    st.header("Video/Webcam (chạy model local)")
    st.markdown(
        "**Lưu ý:** tab này dùng model local (yolov8) để demo video/webcam. "
        "Đảm bảo `yolov8/best.pt` có sẵn nếu muốn chạy local inference."
    )

    # Checkbox chọn chạy local inference
    use_local = st.checkbox("Chạy inference local (không qua API)", value=True, key="local_inference_toggle")

    if use_local:
        from ultralytics import YOLO

        @st.cache_resource
        def load_local_model():
            try:
                model_path = os.path.join(os.path.dirname(__file__), "..", "yolov8", "best.pt")
                return YOLO(model_path)
            except Exception as e:
                st.error(f"Không load được model local: {e}")
                return None

        local_model = load_local_model()
        if local_model is None:
            st.error("Không load được model local. Kiểm tra yolov8/best.pt")
            use_local = False
    else:
        st.info("Sử dụng FastAPI backend để inference video (không khả dụng cho webcam realtime).")

    source = st.radio("Nguồn:", ["Video file", "Webcam"], horizontal=True, key="video_source")
    conf_v = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05, key="confidence_slider")

    frame_slot = st.empty()  # slot hiển thị frame

    # ---------------- Video File ----------------
    if source == "Video file":
        up = st.file_uploader("Upload video", type=["mp4", "mov", "avi"], key="video_upload")
        if up:
            t = tempfile.NamedTemporaryFile(delete=False)
            t.write(up.read())
            cap = cv2.VideoCapture(t.name)

            stop_video = st.checkbox("Dừng video", value=False, key="stop_video_toggle")
            while cap.isOpened() and not stop_video:
                ret, frame = cap.read()
                if not ret:
                    break
                if use_local and local_model:
                    res = local_model.track(frame, conf=conf_v, persist=True, tracker="bytetrack.yaml")
                    frame = res[0].plot()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slot.image(frame_rgb, use_container_width=True)

                stop_video = st.session_state.get("stop_video_toggle", False)

            cap.release()

    # ---------------- Webcam ----------------
    else:
        if use_local:
            # Checkbox bật/tắt webcam (không tạo nhiều lần)
            if "webcam_running" not in st.session_state:
                st.session_state.webcam_running = False
            st.session_state.webcam_running = st.checkbox(
                "Bật webcam", value=False, key="webcam_toggle"
            )

            if st.session_state.webcam_running:
                cap = cv2.VideoCapture(0)
                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Không đọc được webcam.")
                        break
                    res = local_model.track(frame, conf=conf_v, persist=True, tracker="bytetrack.yaml")
                    frame = res[0].plot()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_slot.image(frame_rgb, use_container_width=True)

                    # Kiểm tra trạng thái checkbox
                    if not st.session_state.webcam_running:
                        break

                cap.release()
        else:
            st.error("Webcam realtime yêu cầu inference local (Model local). Bật 'Chạy inference local' trước.")

# ---------------- TAB 3: MÔ HÌNH & THỐNG KÊ ----------------
elif choice == "Thống kê":
    st.header("Mô hình & Thống kê")
    st.markdown("""
    **Mô hình hiển thị:** Mô hình YOLOv8 nhận dạng mít chín  
    **Tập dữ liệu:** ~1700 ảnh, 3 nhãn: mit_non, mit_chin, mit_saubenh  
    **Triển khai:** FastAPI (inference) + Streamlit (frontend)  
    """)
    st.markdown("Bạn có thể mở tab So sánh để xem kết quả training (results_n.csv / results_s.csv).")

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
    st.header("Trợ lý nông nghiệp thông minh - AgriVision 🌾")
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
