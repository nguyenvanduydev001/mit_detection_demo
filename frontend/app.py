import streamlit as st
from streamlit_option_menu import option_menu
import os

# Import các trang
from views import (
    home_page,
    login_page,
    analysis_page,
    video_page,
    stats_page,
    compare_page,
    chat_page,
    account_page
)

# Import các hàm tiện ích
from utils.helpers import get_logo_html, get_menu_style, render_header


# ==================== CẤU HÌNH CHUNG ====================
st.set_page_config(
    page_title="AgriVision - Hệ Thống Nhận Dạng Và Phân Loại Độ Chín Trái Mít",
    layout="wide"
)

# Hiển thị header chung
render_header()

# ==================== LOGO VÀ MENU ====================
assets_dir = os.path.join(os.path.dirname(__file__), "assets")
logo_html = get_logo_html(assets_dir)
menu_styles = get_menu_style()

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding-bottom:10px">
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Menu chính
    choice = option_menu(
        None,
        ["Trang chủ", "Đăng nhập", "Phân tích ảnh", "Video/Webcam",
         "Thống kê", "So sánh YOLOv8", "Chat AgriVision", "Tài khoản"],
        icons=["house", "box-arrow-right", "camera", "camera-video",
               "bar-chart", "activity", "chat-dots", "person-circle"],
        default_index=1,
        styles=menu_styles,
    )

    # Reset session_state khi đổi tab
    if "last_tab" not in st.session_state:
        st.session_state["last_tab"] = choice
    elif st.session_state["last_tab"] != choice:
        st.session_state["last_tab"] = choice
        st.session_state.pop("video_done", None)
        st.session_state.pop("video_json", None)
        st.session_state.pop("last_data", None)


# ==================== ĐIỀU HƯỚNG TRANG ====================
if choice == "Trang chủ":
    home_page.show()
elif choice == "Đăng nhập":
    login_page.show()
elif choice == "Phân tích ảnh":
    analysis_page.show()
elif choice == "Video/Webcam":
    video_page.show()
elif choice == "Thống kê":
    stats_page.show()
elif choice == "So sánh YOLOv8":
    compare_page.show()
elif choice == "Chat AgriVision":
    chat_page.show()
elif choice == "Tài khoản":
    account_page.show()
