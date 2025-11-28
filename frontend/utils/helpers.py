import os
import base64
import streamlit as st


# ==================== ĐỌC FILE ẢNH BASE64 ====================
def get_base64_of_bin_file(bin_file: str) -> str:
    """Đọc file nhị phân và trả về chuỗi base64."""
    try:
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""


# ==================== LẤY LOGO HTML ====================
def get_logo_html(assets_dir: str) -> str:
    """Tạo HTML hiển thị logo từ thư mục assets."""
    logo_path = os.path.join(assets_dir, "logo.png")
    if os.path.exists(logo_path):
        logo_base64 = get_base64_of_bin_file(logo_path)
        return f'<img src="data:image/png;base64,{logo_base64}" width="140" style="border-radius:10px; margin-bottom:10px"/>'
    return "<div style='font-size:40px'>🍈</div>"


# ==================== TẢI STYLE THEO THEME ====================
def get_menu_style() -> dict:
    """Tùy chỉnh style của menu theo theme hiện tại."""
    theme = st.get_option("theme.base")

    if theme == "dark":
        return {
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

    return {
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


# ==================== HIỂN THỊ HEADER CHUNG ====================
def render_header():
    """Render phần tiêu đề chung."""
    st.markdown("""
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
        margin-top: 0.5em;
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
    """, unsafe_allow_html=True)
