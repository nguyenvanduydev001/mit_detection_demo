import streamlit as st
import requests
import os
import time
from pymongo import MongoClient
from dotenv import load_dotenv


def show():
    st.markdown("""
        <div style="text-align:center; margin-top:-1.5em;">
            <h2 style="font-size:22px; font-weight:800; color:#2E7D32;">🌿 TÀI KHOẢN NGƯỜI DÙNG</h2>
            <p style="color:#555;">Quản lý hồ sơ và hoạt động của bạn trong hệ thống AgriVision</p>
        </div>
        <hr style="margin:0.5em 0 1em 0; border-color:#A5D6A7;">
    """, unsafe_allow_html=True)

    if "user" not in st.session_state:
        st.warning("⚠️ Bạn cần đăng nhập để xem thông tin tài khoản.")
        st.info("Vui lòng đăng nhập tại tab **Đăng nhập**.")
        st.stop()

    username = st.session_state["user"]

    # --- Kết nối MongoDB ---
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    try:
        client = MongoClient(MONGO_URI)
        db = client["mit_detection"]
    except Exception as e:
        st.error(f"⚠️ Lỗi kết nối MongoDB: {e}")
        st.stop()

    # --- Lấy thông tin người dùng ---
    try:
        res = requests.get(f"http://127.0.0.1:8000/auth/info?username={username}")
        user = res.json() if res.status_code == 200 else None
    except Exception:
        user = None

    if not user:
        st.error("⚠️ Không thể tải thông tin người dùng.")
        return

    # --- CSS tổng thể ---
    st.markdown("""
        <style>
        .user-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 1.5em;
        }
        .user-card img {
            border-radius: 50%;
            margin-bottom: 10px;
            border: 2px solid #A5D6A7;
        }
        .metric-box {
            background-color: #F1F8E9;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 1px solid #C8E6C9;
        }
        </style>
    """, unsafe_allow_html=True)

    avatar_url = f"https://api.dicebear.com/9.x/identicon/svg?seed={user['username']}"
    st.markdown(f"""
        <div class="user-card">
            <img src="{avatar_url}" width="110">
            <h3 style="color:#2E7D32;">Xin chào, <b>{user['username']}</b> 👋</h3>
            <p><b>Email:</b> {user.get('email', '—')}</p>
            <p><b>Ngày tạo:</b> {user.get('created_at', '—')}</p>
            <p><b>Đăng nhập gần nhất:</b> {user.get('last_login', '—')}</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Thống kê hoạt động ---
    try:
        detections_count = db["analysis_logs"].count_documents({"username": username})
        reports_count = db["compare_logs"].count_documents({"username": username})
        chats_count = db["chat_logs"].count_documents({"username": username})
    except Exception as e:
        st.warning(f"Lỗi tải thống kê: {e}")
        detections_count = reports_count = chats_count = 0

    st.markdown("### 📊 Hoạt động gần đây")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-box"><h4>Ảnh đã phân tích</h4><p style="font-size:18px;font-weight:700;">{detections_count}</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><h4>Báo cáo đã xuất</h4><p style="font-size:18px;font-weight:700;">{reports_count}</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><h4>Lượt chat trợ lý</h4><p style="font-size:18px;font-weight:700;">{chats_count}</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Cập nhật thông tin ---
    st.markdown("### ⚙️ Cập nhật thông tin cá nhân")
    if "form_counter" not in st.session_state:
        st.session_state.form_counter = 0
    if st.session_state.get("refresh_flag", False):
        st.toast("✅ Cập nhật thông tin thành công!", icon="🎉")
        st.session_state.form_counter += 1
        st.session_state.pop("refresh_flag", None)
        st.rerun()

    unique_key = f"upd_form_{st.session_state.form_counter}"

    with st.form(f"update_info_{unique_key}"):
        st.markdown("<div style='padding:10px 0 5px 0;color:#555;'>Bạn có thể cập nhật tên, email hoặc mật khẩu của mình tại đây.</div>", unsafe_allow_html=True)
        new_username = st.text_input("👤 Tên người dùng mới", placeholder="Đổi tên hiển thị...", key=f"upd_username_{unique_key}")
        new_email = st.text_input("📧 Địa chỉ email mới", placeholder="Nhập email mới...", key=f"upd_email_{unique_key}")
        new_password = st.text_input("🔑 Mật khẩu mới", type="password", placeholder="Nhập mật khẩu mới...", key=f"upd_pw_{unique_key}")
        confirm_password = st.text_input("🔁 Xác nhận mật khẩu", type="password", placeholder="Nhập lại mật khẩu mới...", key=f"upd_pw2_{unique_key}")

        submitted = st.form_submit_button("💾 Lưu thay đổi")

        if submitted:
            if new_password and new_password != confirm_password:
                st.error("❌ Mật khẩu xác nhận không khớp.")
            else:
                payload = {}
                if new_username.strip():
                    payload["new_username"] = new_username.strip()
                if new_email.strip():
                    payload["email"] = new_email.strip()
                if new_password:
                    payload["password"] = new_password
                    payload["confirm_password"] = confirm_password

                if not payload:
                    st.warning("⚠️ Bạn chưa thay đổi gì cả.")
                else:
                    with st.spinner("⏳ Đang cập nhật..."):
                        try:
                            res = requests.patch(
                                f"http://127.0.0.1:8000/auth/update?username={username}",
                                json=payload,
                            )
                            if res.status_code == 200:
                                data = res.json()
                                if "new_username" in data:
                                    st.session_state["user"] = data["new_username"]
                                st.session_state["refresh_flag"] = True
                                st.rerun()
                            else:
                                st.toast(res.json().get("detail", "Cập nhật thất bại."), icon="⚠️")
                        except Exception as e:
                            st.toast(f"Lỗi kết nối tới API: {e}", icon="⚠️")

    # --- Nút Đăng xuất ---
    st.markdown("---")
    st.markdown("""
        <style>
        div[data-testid="stButton"][key="logout_btn"] button {
            background-color: #ffccbc;
            color: #5d4037;
            font-weight: 600;
            border-radius: 8px;
            border: 1px solid #ffab91;
            transition: all 0.3s ease;
        }
        div[data-testid="stButton"][key="logout_btn"] button:hover {
            background-color: #ffab91;
            color: white;
            transform: scale(1.03);
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("🚪 Đăng xuất", key="logout_btn", use_container_width=True):
        st.session_state.pop("user", None)
        st.toast("Đã đăng xuất khỏi AgriVision!", icon="✅")
        time.sleep(2)
        st.rerun()
