import streamlit as st
import requests
import os
import time

# --- Hàm hiển thị trang đăng nhập/đăng ký ---
def show():
    # Ẩn tiêu đề chính của app (đã render sẵn ở app.py)
    st.markdown("""
        <style>
            div.main-title, p.sub-title, hr {display: none;}
        </style>
    """, unsafe_allow_html=True)

    # Tiêu đề cục bộ của trang
    st.markdown("""
        <div style="text-align:center; margin-top: -3.1em;">
            <h2 style="font-size: 20px; font-weight: 800; color: #2E7D32; margin-bottom: 0.3em; letter-spacing: 0.5px;">
                AGRI VISION — HỆ THỐNG NHẬN DẠNG VÀ PHÂN LOẠI ĐỘ CHÍN TRÁI MÍT
            </h2>
            <p style="font-style:italic; color:#555;">
                Ứng dụng AI phục vụ Nông nghiệp Thông minh
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Tạo 2 tab: Đăng nhập / Đăng ký
    tab_login, tab_register = st.tabs(["🔑 Đăng nhập", "🧾 Đăng ký"])

    # ---------------- CSS tùy chỉnh ----------------
    st.markdown("""
        <style>
        div[data-testid="stTextInput"] input {
            background-color: #fff !important;
            border: 1.2px solid #cfd8dc !important;
            border-radius: 8px !important;
            padding: 10px 12px !important;
            font-size: 15px !important;
            color: #333 !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #4CAF50 !important;
            box-shadow: 0 0 0 3px rgba(76,175,80,0.15) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ---------------- Cấu hình API ----------------
    API_URL = "http://127.0.0.1:8000"
    base_path = os.path.dirname(__file__)
    sign_in = os.path.join(base_path, ".." , "assets" , "sign-in.svg")
    sign_up = os.path.join(base_path, ".." , "assets" , "sign-up.svg")

    # TAB ĐĂNG NHẬP
    with tab_login:
        st.markdown("<h3 style='text-align:center; margin-bottom:20px;'>Đăng nhập tài khoản</h3>", unsafe_allow_html=True)
        left, right = st.columns([1, 1])

        with left:
            if os.path.exists(sign_in):
                st.image(sign_in, use_container_width=False, width=380)

        with right:
            # Kiểm tra trạng thái đăng nhập
            if "user" in st.session_state:
                st.success(f"👋 Xin chào **{st.session_state['user']}**")
                if st.button("Đăng xuất", use_container_width=True):
                    st.session_state.pop("user", None)
                    st.toast("Đã đăng xuất!", icon="✅")
                    st.rerun()
            else:
                username = st.text_input("👤 Tên người dùng", placeholder="Nhập tên đăng nhập...")
                password = st.text_input("🔒 Mật khẩu", type="password", placeholder="Nhập mật khẩu...")

                if st.button("Đăng nhập", use_container_width=True):
                    if not username or not password:
                        st.warning("⚠️ Vui lòng nhập đầy đủ thông tin.")
                    else:
                        with st.spinner("Đang xác thực..."):
                            try:
                                res = requests.post(f"{API_URL}/auth/login", json={"username": username, "password": password})
                                if res.status_code == 200:
                                    st.session_state["user"] = username
                                    st.toast("Đăng nhập thành công!", icon="🎉")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(res.json().get("detail", "Sai thông tin đăng nhập."))
                            except Exception as e:
                                st.error(f"Lỗi đăng nhập: {e}")
                st.divider()

    # 🟣 TAB ĐĂNG KÝ
    with tab_register:
        st.markdown("<h3 style='text-align:center; margin-bottom:20px;'>Đăng ký tài khoản mới</h3>", unsafe_allow_html=True)
        left, right = st.columns([1, 1])

        with left:
            if os.path.exists(sign_up):
                st.image(sign_up, use_container_width=False, width=450)

        with right:
            # Tạo khóa session riêng để reset form khi cần
            if "register_key" not in st.session_state:
                st.session_state.register_key = str(time.time())

            unique = st.session_state.register_key  # key động để render lại form
            username = st.text_input("👤 Tên người dùng", key=f"user_{unique}", placeholder="Tên hiển thị hoặc nickname...")
            email = st.text_input("📧 Địa chỉ email", key=f"email_{unique}", placeholder="example@gmail.com")
            password = st.text_input("🔑 Mật khẩu", key=f"pass_{unique}", type="password", placeholder="Nhập mật khẩu...")
            confirm_password = st.text_input("🔁 Xác nhận mật khẩu", key=f"confirm_{unique}", type="password", placeholder="Nhập lại mật khẩu...")

            if st.button("Tạo tài khoản", use_container_width=True):
                if not username or not email or not password:
                    st.warning("⚠️ Vui lòng nhập đầy đủ thông tin.")
                elif password != confirm_password:
                    st.error("❌ Mật khẩu xác nhận không khớp.")
                else:
                    with st.spinner("🛠️ Đang tạo tài khoản..."):
                        try:
                            res = requests.post(f"{API_URL}/auth/register", json={
                                "username": username,
                                "email": email,
                                "password": password,
                                "confirm_password": confirm_password
                            })
                            if res.status_code == 200:
                                st.success("Tài khoản đã được tạo. Vui lòng chuyển sang tab **Đăng nhập** để tiếp tục.", icon="👋")
                                time.sleep(2)
                                # Reset form bằng key mới
                                st.session_state.register_key = str(time.time())
                                st.rerun()
                            else:
                                st.toast(res.json().get("detail", "Lỗi đăng ký không xác định."), icon="⚠️")
                        except Exception as e:
                            st.toast(f"Lỗi kết nối tới API: {e}", icon="⚠️")
