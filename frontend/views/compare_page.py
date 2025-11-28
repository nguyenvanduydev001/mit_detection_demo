import streamlit as st
import io, os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont


def show():
    # --- Kiểm tra đăng nhập ---
    if "user" not in st.session_state or not st.session_state["user"]:
        st.warning("⚠️ Bạn cần đăng nhập để truy cập tính năng này.")
        st.info("Vui lòng chuyển sang tab **Đăng nhập** để tiếp tục.")
        st.stop()

    username = st.session_state["user"]

    # --- Kết nối MongoDB ---
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=dotenv_path)
    MONGO_URI = os.getenv("MONGO_URI")
    try:
        client = MongoClient(MONGO_URI)
        db = client["mit_detection"]
        report_logs = db["report_logs"]
    except Exception as e:
        st.warning(f"⚠️ Không thể kết nối MongoDB: {e}")
        st.stop()

    # ======================= GIAO DIỆN =========================
    st.markdown("## ⚖️ So sánh mô hình YOLOv8n và YOLOv8s")
    st.caption("Đánh giá chi tiết hiệu năng mô hình nhận dạng mít – hỗ trợ chọn mô hình phù hợp cho ứng dụng.")

    # ======================= UPLOAD FILE =========================
    st.markdown("### 📂 Tải dữ liệu huấn luyện")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_n = st.file_uploader("Kết quả YOLOv8n", type=["csv"], key="n")
    with col2:
        uploaded_s = st.file_uploader("Kết quả YOLOv8s", type=["csv"], key="s")

    if uploaded_n is None or uploaded_s is None:
        st.info("⬆️ Vui lòng tải **cả hai file kết quả (.csv)** để hiển thị bảng so sánh và báo cáo.")
        st.stop()

    # ======================= ĐỌC FILE =========================
    df_n = pd.read_csv(uploaded_n)
    df_s = pd.read_csv(uploaded_s)

    # ======================= CHỈ SỐ TỔNG QUAN =========================
    st.markdown("### 📈 Tổng quan nhanh")

    summary_metrics = [
        "metrics/precision(B)", "metrics/recall(B)",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    ]

    v8n = [df_n[m].iloc[-1] if m in df_n.columns else np.nan for m in summary_metrics]
    v8s = [df_s[m].iloc[-1] if m in df_s.columns else np.nan for m in summary_metrics]

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Precision", f"{v8s[0]*100:.2f}%", delta=f"{(v8s[0]-v8n[0])*100:.2f}%")
    col2.metric("📊 Recall", f"{v8s[1]*100:.2f}%", delta=f"{(v8s[1]-v8n[1])*100:.2f}%")
    col3.metric("🔥 mAP50", f"{v8s[2]*100:.2f}%", delta=f"{(v8s[2]-v8n[2])*100:.2f}%")

    st.divider()

    # ======================= BIỂU ĐỒ DẠNG LINE =========================
    st.markdown("### 📉 Hiệu năng theo Epoch")
    chart_colors = {"n": "#A5D6A7", "s": "#2E7D32"}

    for metric in summary_metrics[:-1]:
        if metric in df_n.columns and metric in df_s.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df_n[metric], mode='lines', name="YOLOv8n",
                                     line=dict(color=chart_colors["n"], width=2)))
            fig.add_trace(go.Scatter(y=df_s[metric], mode='lines', name="YOLOv8s",
                                     line=dict(color=chart_colors["s"], width=2)))
            fig.update_layout(
                title=metric.replace("metrics/", "").replace("(B)", "").upper(),
                xaxis_title="Epoch", yaxis_title="Giá trị", template="plotly_white",
                height=320, legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                        xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ======================= BIỂU ĐỒ CỘT SO SÁNH =========================
    st.markdown("### 📊 So sánh hiệu năng trung bình")

    compare_df = pd.DataFrame({
        "Chỉ số": ["Precision", "Recall", "mAP50", "mAP50-95"],
        "YOLOv8n": v8n,
        "YOLOv8s": v8s
    })

    fig_bar = px.bar(
        compare_df.melt(id_vars="Chỉ số", var_name="Mô hình", value_name="Giá trị"),
        x="Chỉ số", y="Giá trị", color="Mô hình",
        color_discrete_sequence=["#A5D6A7", "#2E7D32"],
        barmode="group", text="Giá trị"
    )
    fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_bar.update_layout(height=350, template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ======================= NHẬN XÉT =========================
    st.markdown("### 💬 Nhận xét từ AgriVision")

    precision_diff = v8s[0] - v8n[0]
    recall_diff = v8s[1] - v8n[1]
    map_diff = v8s[2] - v8n[2]

    insights = []
    if map_diff > 0.01:
        insights.append("YOLOv8s đạt mAP50 cao hơn, phù hợp với hệ thống yêu cầu độ chính xác cao.")
    elif map_diff < -0.01:
        insights.append("YOLOv8n có mAP50 tốt hơn nhẹ, tốc độ xử lý nhanh hơn.")
    if precision_diff > 0.01:
        insights.append("YOLOv8s có Precision cao hơn, giảm nhầm lẫn trong phát hiện mít chín.")
    elif recall_diff > 0.01:
        insights.append("YOLOv8n có Recall tốt hơn, phát hiện được nhiều trái hơn.")
    insights.extend([
        "YOLOv8n huấn luyện nhanh hơn khoảng 40–60%.",
        "Với thiết bị giới hạn như Jetson hoặc Raspberry Pi, nên chọn YOLOv8n.",
        "Nếu triển khai quy mô lớn hoặc trên cloud, YOLOv8s là lựa chọn ưu tiên."
    ])

    for line in insights:
        st.markdown(f"• {line}")

    st.divider()

    # ======================= XUẤT FILE PDF =========================
    st.markdown("### 🧾 Xuất báo cáo PDF")

    def generate_pdf(username, v8n, v8s, insights):
        buffer = io.BytesIO()
        base_dir = os.path.dirname(__file__)
        font_path = os.path.join(base_dir, ".." , "fonts", "Roboto-Regular.ttf")
        bold_path = os.path.join(base_dir, ".." , "fonts", "Roboto-Bold.ttf")
        logo_path = os.path.join(base_dir, ".." , "assets", "logo.png")

        use_roboto = os.path.exists(font_path) and os.path.exists(bold_path)
        if use_roboto:
            pdfmetrics.registerFont(TTFont("Roboto", font_path))
            pdfmetrics.registerFont(TTFont("Roboto-Bold", bold_path))
            font_main, font_bold = "Roboto", "Roboto-Bold"
        else:
            pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
            font_main = font_bold = "HeiseiMin-W3"

        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        for k in ["Normal", "Title", "Heading3", "Italic"]:
            styles[k].fontName = font_main
        styles["Title"].fontName = font_bold
        styles["Title"].fontSize = 18
        styles["Heading3"].fontSize = 12
        styles["Normal"].fontSize = 11
        styles["Italic"].fontSize = 10

        story = []
        if os.path.exists(logo_path):
            story.append(Image(logo_path, width=227, height=44, hAlign="CENTER"))
            story.append(Spacer(1, 6))

        story.append(Paragraph("<b>AgriVision – So sánh hiệu suất mô hình YOLOv8</b>", styles["Title"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"Người dùng: {username}", styles["Normal"]))
        story.append(Paragraph(f"Ngày tạo báo cáo: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
        story.append(Spacer(1, 14))

        data = [["Chỉ số", "YOLOv8n", "YOLOv8s"]] + [
            [m, f"{v8n[i]:.4f}", f"{v8s[i]:.4f}"]
            for i, m in enumerate(["Precision", "Recall", "mAP50", "mAP50-95"])
        ]
        table = Table(data, hAlign="CENTER", colWidths=[100, 100, 100])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#C8E6C9")),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("FONTNAME", (0, 0), (-1, -1), font_main),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.whitesmoke, rl_colors.HexColor("#F7FBF7")]),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        story.append(Paragraph("<b>Nhận xét từ AgriVision:</b>", styles["Heading3"]))
        story.append(Spacer(1, 6))
        for line in insights:
            story.append(Paragraph(f" • {line}", styles["Normal"]))
            story.append(Spacer(1, 4))

        story.append(Spacer(1, 12))
        story.append(Paragraph("AgriVision — Đánh giá mô hình AI cho nông nghiệp hiện đại.", styles["Italic"]))
        doc.build(story)
        buffer.seek(0)
        return buffer

    if st.button("📄 Xuất báo cáo PDF"):
        pdf_buffer = generate_pdf(username, v8n, v8s, insights)
        st.session_state["pdf_buffer"] = pdf_buffer

        try:
            report_logs.insert_one({
                "timestamp": datetime.now().isoformat(),
                "username": username,
                "src_n": uploaded_n.name,
                "src_s": uploaded_s.name,
                "precision_n": float(v8n[0]), "precision_s": float(v8s[0]),
                "recall_n": float(v8n[1]), "recall_s": float(v8s[1]),
                "map50_n": float(v8n[2]), "map50_s": float(v8s[2]),
                "map50_95_n": float(v8n[3]), "map50_95_s": float(v8s[3]),
                "insights": insights
            })
            st.toast("Báo cáo đã được tạo và lưu log thành công.", icon="📄")
        except Exception as e:
            st.warning(f"⚠️ Không thể lưu log báo cáo: {e}")

    if "pdf_buffer" in st.session_state:
        st.download_button(
            label="💾 Tải xuống PDF",
            data=st.session_state["pdf_buffer"],
            file_name=f"AgriVision_YOLOv8_Comparison_{username}.pdf",
            mime="application/pdf"
        )
