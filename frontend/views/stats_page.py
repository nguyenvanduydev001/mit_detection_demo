# stats_page.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os


def show():
    # --- Tải môi trường & kiểm tra đăng nhập ---
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")

    if "user" not in st.session_state or not st.session_state["user"]:
        st.warning("⚠️ Bạn cần đăng nhập để xem thống kê vườn.")
        st.info("Vui lòng chuyển sang tab **Đăng nhập** để tiếp tục.")
        st.stop()

    username = st.session_state["user"]

    # --- Kết nối MongoDB ---
    client = MongoClient(MONGO_URI)
    db = client["mit_detection"]
    logs_col = db["analysis_logs"]   # ✅ Đọc từ collection mới

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

    latest_log = logs_col.find_one({"username": username}, sort=[("timestamp", -1)])
    counts, total = {}, 0

    if latest_log:
        counts = latest_log.get("counts", {})
        total = latest_log.get("total", sum(counts.values()))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số trái phát hiện", total)
    col2.metric("✅ Mít chín", counts.get("mit_chin", 0))
    col3.metric("🌱 Mít non", counts.get("mit_non", 0))
    col4.metric("⚠️ Mít sâu bệnh", counts.get("mit_saubenh", 0))

    if total > 0:
        st.caption(f"Cập nhật lúc {datetime.now().strftime('%H:%M – %d/%m/%Y')}")
        df_counts = pd.DataFrame(list(counts.items()), columns=["Loại", "Số lượng"])
        fig, ax = plt.subplots()
        colors = ["#7FC97F", "#FDBF6F", "#E31A1C"]
        ax.pie(df_counts["Số lượng"], labels=df_counts["Loại"],
               autopct="%1.1f%%", startangle=90, colors=colors)
        ax.set_title("Tỷ lệ các loại mít trong vườn", fontsize=12)
        fig.set_size_inches(4, 4)
        st.pyplot(fig)
    else:
        st.info("💡 Chưa có dữ liệu nhận dạng gần đây. Hãy tải video hoặc bật webcam để cập nhật vườn nhé.")

    st.divider()

    # ======================= HOẠT ĐỘNG GẦN ĐÂY =========================
    st.subheader("📅 Nhật ký hoạt động nhận dạng")

    logs = list(logs_col.find({"username": username}).sort("timestamp", -1).limit(50))
    if logs:
        df_hist = pd.DataFrame(logs)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])

        records = []
        for _, row in df_hist.iterrows():
            for k, v in row.get("counts", {}).items():
                records.append({"timestamp": row["timestamp"], "class": k, "count": v})
        df_flat = pd.DataFrame(records)

        if not df_flat.empty:
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(6, 3))

            colors = {
                "mit_chin": "#4CAF50",
                "mit_non": "#FF9800",
                "mit_saubenh": "#2196F3"
            }

            ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
            for cls_name, group in df_flat.groupby("class"):
                ax.plot(
                    group["timestamp"], group["count"],
                    marker="o", markersize=6, linewidth=2.5,
                    color=colors.get(cls_name, "#9E9E9E"),
                    label=cls_name.replace("_", " ").capitalize()
                )

            ax.legend(fontsize=9, loc="upper left", frameon=False)
            ax.set_ylabel("Số lượng phát hiện", fontsize=10)
            ax.set_title("Xu hướng nhận dạng mít theo thời gian", fontsize=12, fontweight="bold", pad=10)
            ax.tick_params(axis="x", labelrotation=20, labelsize=8)
            ax.set_xlabel("Thời gian", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.caption("📂 Chưa có lịch sử nhận dạng chi tiết.")
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
