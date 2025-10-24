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
    
    <div class="main-title">ỨNG DỤNG YOLOv8 TRONG NHẬN DẠNG VÀ PHÂN LOẠI ĐỘ CHÍN TRÁI MÍT</div>
    <p class="sub-title">Phục vụ Nông nghiệp thông minh 🌾</p>
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
            <h3 style="margin:0; color:#6DBE45;">🌾 Nông nghiệp thông minh</h3>
            <p style="font-size:13px; color:gray;">Ứng dụng AI nhận dạng độ chín trái mít</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    choice = option_menu(
        None,
        ["Trang chủ", "Nhận dạng ảnh", "Video/Webcam",
         "Thống kê", "So sánh YOLOv8", "AI Insight", "Chat Gemini"],
        icons=["house", "camera", "camera-video", "bar-chart", "activity", "cpu", "chat-dots"],
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
elif choice == "Nhận dạng ảnh":
    st.header("Nhận dạng ảnh tĩnh")
    col1, col2 = st.columns([1,1])
    with col1:
        uploaded_file = st.file_uploader("Chọn ảnh trái mít...", type=["jpg","jpeg","png"])
        confidence = st.slider("Ngưỡng confidence", 0.1, 1.0, 0.5, 0.05)
        enable_alert = st.checkbox("🔔 Bật cảnh báo thu hoạch qua email (giả lập)", value=False)
        analyze_btn = st.button("🔍 Phân tích ảnh")

    with col2:
        st.markdown("**Ảnh gốc**")
        preview = st.empty()
        st.markdown("**Ảnh kết quả**")
        out_image = st.empty()
        st.markdown("**Thống kê nhanh**")
        stats_box = st.empty()

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        preview.image(img, use_container_width=True)

    if analyze_btn and uploaded_file:
        with st.spinner("Gửi ảnh tới API, chờ kết quả..."):
            files = {"file": uploaded_file.getvalue()}
            try:
                resp = requests.post(API_URL, files=files, params={"conf": confidence}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"Lỗi gọi API: {e}")
                data = None

        if data:
            # show annotated image
            img_data = base64.b64decode(data["image"])
            annotated = Image.open(io.BytesIO(img_data)).convert("RGB")
            out_image.image(annotated, use_container_width=True)

            detections = data.get("detections", [])
            if not detections:
                stats_box.warning("Không phát hiện được trái mít nào.")
            else:
                # bảng detections
                df = pd.DataFrame(detections)
                stats_box.dataframe(df)

                # counts + pie chart
                labels = df["label"].tolist()
                count_df = pd.Series(labels).value_counts().rename_axis('Loại').reset_index(name='Số lượng')
                st.subheader("Thống kê")
                st.dataframe(count_df)

                fig, ax = plt.subplots()
                ax.pie(count_df["Số lượng"], labels=count_df["Loại"], autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

                # gợi ý thu hoạch
                total = count_df["Số lượng"].sum()
                chin = int(count_df[count_df["Loại"]=="mit_chin"]["Số lượng"].sum()) if "mit_chin" in count_df["Loại"].values else 0
                ratio = chin / total if total>0 else 0
                if ratio > 0.7:
                    st.success("🌾 Gợi ý: Hơn 70% trái chín — nên thu hoạch sớm.")
                    if enable_alert:
                        st.info("📤 [Giả lập] Đã gửi email cảnh báo tới user@example.com")
                elif ratio > 0.4:
                    st.info("🍈 Nhiều trái đang chín — theo dõi thêm.")
                else:
                    st.warning("🟢 Chủ yếu là mít non, chưa nên thu hoạch.")

            # write latest_results.json was created by backend; just show
            if os.path.exists(LATEST_RESULTS):
                st.info("✅ Kết quả lưu sẵn cho AI Insight.")
            else:
                st.info("⚠️ latest_results.json chưa được lưu (kiểm tra backend).")

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

# ---------------- TAB 5: AI INSIGHT (Gemini phân tích kết quả) ----------------
elif choice == "AI Insight":
    st.header("AI Insight — Phân tích kết quả")
    st.markdown("Tab này đọc file `latest_results.json` (do backend lưu) và dùng Gemini để sinh báo cáo kỹ thuật. Nếu Gemini không sẵn, app dùng fallback phân tích tự động ngắn.")

    if os.path.exists(LATEST_RESULTS):
        with open(LATEST_RESULTS, "r", encoding="utf-8") as f:
            last = json.load(f)
        st.subheader("Kết quả mới nhất")
        st.json(last)
        if st.button("Yêu cầu AI phân tích (Gemini)"):
            # build prompt
            counts = last.get("counts", {})
            total = sum(counts.values()) if counts else 0
            prompt = f"""Bạn là chuyên gia nông nghiệp + kỹ sư AI. Dữ liệu đầu vào từ hệ thống nhận dạng trái mít:
counts={counts}, total={total}.
Hãy viết báo cáo kỹ thuật (tiếng Việt, formal) gồm:
1) Tóm tắt tình trạng (tỉ lệ chín/non/sâu bệnh)
2) Khuyến nghị thu hoạch (khi nào, vì sao)
3) Biện pháp xử lý nếu thấy sâu bệnh
4) Gợi ý kỹ thuật cho triển khai tiếp (sampling, thu thập thêm ảnh)
5) Ngắn gọn kết luận.
Trả lời chi tiết, ngôn ngữ chuyên môn kỹ thuật nông nghiệp và AI."""
            st.info("Đang gửi prompt đến Gemini (nếu key hợp lệ)...")
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

            if not ai_text:
                # fallback local analysis
                lines = []
                lines.append("Báo cáo phân tích (fallback):")
                if total == 0:
                    lines.append("- Không phát hiện trái mít nào trong ảnh.")
                else:
                    for k, v in counts.items():
                        pct = (v/total)*100 if total>0 else 0
                        lines.append(f"- {k}: {v} trái ({pct:.1f}%)")
                    if counts.get("mit_chin",0)/total > 0.7:
                        lines.append("- Khuyến nghị: Thu hoạch sớm trong 1-3 ngày.")
                    elif counts.get("mit_chin",0)/total > 0.4:
                        lines.append("- Khuyến nghị: Theo dõi, có thể thu hoạch lứa nhỏ.")
                    else:
                        lines.append("- Khuyến nghị: Chưa thu hoạch; tiếp tục theo dõi.")
                ai_text = "\n".join(lines)
            st.subheader("Kết quả phân tích AI")
            st.markdown(ai_text)
    else:
        st.info("Chưa có kết quả mới (latest_results.json). Hãy chạy Tab Ảnh để tạo.")

# ---------------- TAB 6: CHAT (Gemini) ----------------
elif choice == "Chat Gemini":
    st.header("Chat tự do với Gemini")
    st.header("Chat tự do với Gemini (hỏi về mô hình, nông nghiệp...)")
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
