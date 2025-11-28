import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import google.generativeai as genai


def show():
    # --- Kiểm tra đăng nhập ---
    if "user" not in st.session_state or not st.session_state["user"]:
        st.warning("⚠️ Bạn cần đăng nhập để sử dụng Chat AgriVision.")
        st.info("Vui lòng chuyển sang tab **Đăng nhập** để tiếp tục.")
        st.stop()

    username = st.session_state["user"]

    # --- Kết nối MongoDB ---
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    try:
        client = MongoClient(MONGO_URI)
        db = client["mit_detection"]
        chat_logs = db["chat_logs"]
    except Exception as e:
        st.warning(f"⚠️ Không thể kết nối MongoDB: {e}")
        st.stop()

    # --- Cấu hình Gemini ---
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_KEY:
        try:
            genai.configure(api_key=GEMINI_KEY)
        except Exception:
            pass

    # --- Header ---
    st.markdown(f"""
        <h1 style='text-align: center; color: #2E7D32; font-weight: 700;'>Chat AgriVision</h1>
        <p style='text-align: center; color: #555; font-size: 16px;'>
            Xin chào, <b>{username}</b> 👋<br>
            Hỏi tôi về <b>YOLOv8</b>, <b>độ chín của mít</b>, hoặc <b>kỹ thuật nông nghiệp thông minh</b>.
        </p>
    """, unsafe_allow_html=True)

    # --- Khởi tạo lịch sử người dùng ---
    if "chat_user" not in st.session_state or st.session_state.get("chat_user") != username:
        st.session_state.chat_history = []
        st.session_state.chat_user = username

        chats = list(chat_logs.find({"username": username}).sort("timestamp", -1).limit(10))
        chats.reverse()
        for c in chats:
            st.session_state.chat_history.append({"role": "user", "content": c["user_message"]})
            st.session_state.chat_history.append({"role": "assistant", "content": c["assistant_reply"]})

    # --- CSS ---
    st.markdown("""
        <style>
        .chat-box {
            border-radius: 10px;
            padding: 8px 14px;
            margin: 6px 0;
            width: fit-content;
            max-width: 75%;
            word-wrap: break-word;
            font-size: 15px;
            line-height: 1.4;
        }
        .user-msg {
            background-color: #DCF8C6;
            margin-left: auto;
            margin-right: 10px;
            text-align: right;
            border: 1px solid #C8E6C9;
        }
        .assistant-msg {
            background-color: #ffffff;
            margin-right: auto;
            margin-left: 10px;
            border: 1px solid #E0E0E0;
        }
        div[data-testid="stButton"][key="floating_clear"] {
            position: fixed;
            bottom: 110px;
            right: 35px;
            z-index: 999;
        }
        div[data-testid="stButton"][key="floating_clear"] button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 50%;
            width: 45px;
            height: 45px;
            font-size: 20px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.2);
            transition: all 0.25s ease-in-out;
        }
        div[data-testid="stButton"][key="floating_clear"] button:hover {
            background-color: #388E3C;
            transform: scale(1.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Hiển thị lịch sử ---
    for msg in st.session_state.chat_history:
        role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
        st.markdown(f"<div class='chat-box {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    # --- Ô nhập liệu ---
    user_input = st.chat_input("💬 Gõ câu hỏi của bạn...")

    # --- Nút xóa hội thoại ---
    clear_btn = st.button("🗑️", key="floating_clear", help="Xóa hội thoại", use_container_width=False)
    if clear_btn:
        st.session_state.chat_history = []
        chat_logs.delete_many({"username": username})
        st.toast("Đã xóa toàn bộ hội thoại", icon="🗑️")
        st.rerun()

    # --- Xử lý gửi tin ---
    if user_input:
        current_time = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")

        st.markdown(f"<div class='chat-box user-msg'>{user_input}</div>", unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "user", "content": user_input, "time": current_time})

        try:
            if GEMINI_KEY:
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                context = """
                Bạn là AgriVision — trợ lý nông nghiệp thông minh hỗ trợ người trồng mít.
                Bạn giao tiếp như một kỹ sư nông nghiệp thân thiện, dễ hiểu.

                Quy tắc xưng hô:
                - Luôn bắt đầu câu trả lời bằng: “Chào bạn,”
                - Không dùng bất kỳ cách xưng hô nào khác (không dùng anh/chị, bà con, quý khách,...)

                Lưu ý:
                - Bạn KHÔNG phân tích hình ảnh. Việc nhận diện đã do mô hình YOLOv8 xử lý.
                - Nếu người dùng hỏi về ảnh, bạn chỉ hướng dẫn cách hiểu kết quả nhận diện, không tự suy diễn từ ảnh.

                Nhiệm vụ chính:
                1. Giải thích kết quả nhận diện mít (chín, non, sâu bệnh…).  
                2. Hướng dẫn kiểm tra trái mít bằng mắt thường.  
                3. Tư vấn kỹ thuật: chăm sóc, bón phân, phòng trừ sâu bệnh.  
                4. Đưa lời khuyên thực tế ngoài vườn, dễ hiểu, dễ làm theo.  
                5. Giữ câu trả lời gọn gàng, thân thiện, đúng vai trò kỹ sư hỗ trợ.

                Phong cách:
                - Mở đầu bằng “Chào bạn,”
                - Thân thiện, rõ ràng, không dùng từ ngữ phức tạp.
                - Tập trung vào giải pháp thực tế cho nông dân.
                """
                resp = model.generate_content(f"{context}\n\nNgười dùng hỏi: {user_input}")
                answer = getattr(resp, "text", None) or str(resp)
            else:
                raise RuntimeError("Thiếu GEMINI_API_KEY trong môi trường")
        except Exception as e:
            answer = f"⚠️ Không thể gọi Gemini API: {e}"

        st.markdown(f"<div class='chat-box assistant-msg'>{answer}</div>", unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "time": current_time})

        # --- Lưu vào MongoDB ---
        try:
            chat_logs.insert_one({
                "timestamp": datetime.now().isoformat(),
                "username": username,
                "user_message": user_input,
                "assistant_reply": answer,
                "model": "Gemini 2.5 Flash"
            })
        except Exception as e:
            st.warning(f"⚠️ Không thể lưu chat vào MongoDB: {e}")

        st.rerun()
