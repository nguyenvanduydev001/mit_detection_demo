import streamlit as st
import os
import base64

def show():
    # --- Đường dẫn tài nguyên ---
    base_path = os.path.dirname(__file__)
    hero_img = os.path.join(base_path, ".." , "assets" , "hero-agriculture.svg")
    data_img = os.path.join(base_path, ".." , "assets" , "data-analysis.svg")
    realtime_img = os.path.join(base_path, ".." , "assets" , "realtime-detection.svg")
    stats_img = os.path.join(base_path, ".." , "assets" , "stats-graph.svg")
    compare_img = os.path.join(base_path, ".." , "assets" , "compare-yolov8.svg")
    chat_img = os.path.join(base_path, ".." , "assets" , "chat-ai.svg")
    logo_path = os.path.join(base_path, ".." , "assets" , "logo.png")

    # --- Hàm tiện ích ---
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    if os.path.exists(logo_path):
        logo_base64 = get_base64_of_bin_file(logo_path)
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="140" style="border-radius:10px; margin-bottom:10px"/>'
    else:
        logo_html = "<div style='font-size:40px'>🍈</div>"

    # # --- Phần đầu trang ---
    # st.markdown(
    #     """
    #     <style>
    #     .main-title {
    #         font-size: 20px;
    #         font-weight: 800;
    #         text-align: center;
    #         color: #2E7D32;
    #         margin-bottom: 0.3em;
    #         letter-spacing: 0.5px;
    #     }
    #     .sub-title {
    #         text-align: center;
    #         font-size: 18px;
    #         color: #555;
    #         font-style: italic;
    #         margin-top: 0.5em;
    #         margin-bottom: 1.5em;
    #     }
    #     hr {
    #         border: none;
    #         height: 2px;
    #         background: linear-gradient(to right, #8BC34A, #558B2F);
    #         margin-bottom: 1.5em;
    #     }
    #     </style>

    #     <div class="main-title">AGRI VISION — HỆ THỐNG NHẬN DẠNG VÀ PHÂN LOẠI ĐỘ CHÍN TRÁI MÍT</div>
    #     <p class="sub-title">Ứng dụng AI phục vụ Nông nghiệp Thông minh</p>
    #     <hr>
    #     """,
    #     unsafe_allow_html=True
    # )

    # --- GIỚI THIỆU TỔNG QUAN ---
    st.markdown(f"""
    <div style='text-align:center; margin:10px 0 40px 0;'>
        {logo_html}
        <h1 style="color:#2E7D32; font-size:30px; font-weight:800;">
            AgriVision — Trợ lý AI cho Nông nghiệp Việt
        </h1>
        <p style="color:#555; font-size:18px; max-width:780px; margin:10px auto; line-height:1.6;">
            Nền tảng ứng dụng <b>AI</b> và <b>phân tích hình ảnh</b> giúp nông dân <b>quan sát, đánh giá</b> và <b>chăm sóc vườn mít</b> chính xác, dễ dùng và hiệu quả.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- GIỚI THIỆU NHANH ---
    col1, col2 = st.columns([1.1, 1], vertical_alignment="center")
    with col1:
        if os.path.exists(hero_img):
            st.image(hero_img, use_container_width=True)
    with col2:
        st.markdown("""
        <div style='margin-top:10px;'>
            <h3 style='color:#2E7D32;'>AgriVision giúp bạn:</h3>
            <ul style='font-size:16px; color:#444; line-height:1.8;'>
                <li>Phân tích hình ảnh để <b>nhận biết độ chín</b>, sâu bệnh và chất lượng trái mít.</li>
                <li>Nhận dạng thời gian thực qua <b>camera</b> hoặc <b>video</b> tại vườn.</li>
                <li>Thống kê dữ liệu phát triển, dự báo <b>xu hướng cây trồng</b>.</li>
                <li>So sánh mô hình <b>YOLOv8</b> để chọn kết quả tối ưu.</li>
                <li>Tương tác với <b>Chat AgriVision</b> để được gợi ý kỹ thuật và thời điểm thu hoạch phù hợp.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:60px 0;'></div>", unsafe_allow_html=True)

    # --- TÍNH NĂNG NỔI BẬT ---
    st.markdown("""
    <div style='text-align:center; margin-bottom:50px;'>
        <h2 style='color:#33691E; font-size:24px;'>Tính năng nổi bật</h2>
        <p style='color:#555; font-size:16px; max-width:720px; margin:auto;'>
            AgriVision mang đến bộ công cụ toàn diện cho người trồng mít — từ hình ảnh, video đến dữ liệu, tất cả được xử lý bằng AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Feature 1 ---
    colA, colB = st.columns([1, 1.1], vertical_alignment="center")
    with colA:
        if os.path.exists(data_img):
            st.image(data_img, use_container_width=True)
    with colB:
        st.markdown("""
        <h4 style='color:#2E7D32;'>Phân tích ảnh & dữ liệu</h4>
        <p style='color:#444; font-size:16px; line-height:1.7;'>
            AgriVision tự động nhận dạng độ chín, tình trạng sâu bệnh và đánh giá chất lượng trái mít.  
            Kết quả hiển thị trực quan, giúp người trồng dễ theo dõi và so sánh qua từng thời điểm phát triển.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:50px 0;'></div>", unsafe_allow_html=True)

    # --- Feature 2 ---
    colC, colD = st.columns([1.1, 1], vertical_alignment="center")
    with colC:
        st.markdown("""
        <h4 style='color:#2E7D32;'>Nhận dạng thời gian thực</h4>
        <p style='color:#444; font-size:16px; line-height:1.7;'>
            Theo dõi trực tiếp qua <b>camera</b> hoặc <b>video</b> — nhận biết mít chín, non, sâu bệnh trong thời gian thực.
            AgriVision đưa ra gợi ý <b>thu hoạch, xử lý sâu bệnh</b> hoặc điều chỉnh chăm sóc cây non.
        </p>
        """, unsafe_allow_html=True)
    with colD:
        if os.path.exists(realtime_img):
            st.image(realtime_img, use_container_width=True)

    st.markdown("<div style='margin:50px 0;'></div>", unsafe_allow_html=True)

    # --- Feature 3 ---
    colE, colF = st.columns([1, 1.1], vertical_alignment="center")
    with colE:
        if os.path.exists(stats_img):
            st.image(stats_img, use_container_width=True)
    with colF:
        st.markdown("""
        <h4 style='color:#2E7D32;'>Thống kê & gợi ý thông minh</h4>
        <p style='color:#444; font-size:16px; line-height:1.7;'>
            Lưu trữ dữ liệu theo từng đợt kiểm tra và phân tích <b>xu hướng phát triển</b> vườn mít.  
            Hệ thống gợi ý thời điểm thu hoạch, cảnh báo sớm sâu bệnh và đưa ra <b>khuyến nghị kỹ thuật</b> tối ưu.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:70px 0 40px 0;'></div>", unsafe_allow_html=True)

    # --- MỞ RỘNG HỆ THỐNG ---
    st.markdown("""
    <div style='text-align:center; margin-bottom:40px;'>
        <h2 style='color:#33691E; font-size:24px;'>🧠 Mở rộng hệ thống AgriVision</h2>
        <p style='color:#555; font-size:16px; max-width:720px; margin:auto;'>
            Kết hợp AI, dữ liệu và tương tác thông minh để hỗ trợ người nông dân toàn diện — từ so sánh mô hình đến trò chuyện cùng trợ lý ảo.
        </p>
    </div>
    """, unsafe_allow_html=True)

    colX, colY = st.columns([1.1, 1], vertical_alignment="center")
    with colX:
        st.markdown("""
        <h4 style='color:#2E7D32;'>So sánh mô hình YOLOv8</h4>
        <p style='color:#444; font-size:16px; line-height:1.7;'>
            Đánh giá hiệu năng giữa các phiên bản YOLOv8, chọn mô hình có độ chính xác và tốc độ phù hợp nhất cho điều kiện thực tế.
        </p>
        """, unsafe_allow_html=True)
    with colY:
        if os.path.exists(compare_img):
            st.image(compare_img, use_container_width=True)

    st.markdown("<div style='margin:40px 0;'></div>", unsafe_allow_html=True)

    colZ1, colZ2 = st.columns([1, 1.1], vertical_alignment="center")
    with colZ1:
        if os.path.exists(chat_img):
            st.image(chat_img, use_container_width=True)
    with colZ2:
        st.markdown("""
        <h4 style='color:#2E7D32;'>Chat AgriVision</h4>
        <p style='color:#444; font-size:16px; line-height:1.7;'>
            Trợ lý AI tương tác trực tuyến — giải đáp kỹ thuật, đưa ra gợi ý chăm sóc và hướng dẫn phân tích kết quả trực tiếp trong ứng dụng.
        </p>
        """, unsafe_allow_html=True)

    # --- Footer ---
    st.markdown("""
    <p style='text-align:center; color:#888; font-size:14px; margin-top:50px;'>
        © 2025 AgriVision. Nền tảng AI đồng hành cùng Nông nghiệp Việt Nam.
    </p>
    """, unsafe_allow_html=True)
