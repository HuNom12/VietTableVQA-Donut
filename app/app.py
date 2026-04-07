import streamlit as st
import requests
from PIL import Image
import io

# Cấu hình trang
st.set_page_config(page_title="VietTableVQA - AI Dashboard", layout="centered")

st.title("🚀 Hệ thống TableVQA Tiếng Việt")
st.markdown("---")

# Sidebar thông tin
with st.sidebar:
    st.header("Thông tin dự án")
    st.info("Sử dụng mô hình Donut đã Fine-tune trên tập dữ liệu bảng biểu tiếng Việt.")
    st.write("👤 **Sinh viên:** Trần Hữu Nam")

# 1. Upload ảnh
uploaded_file = st.file_uploader("Tải lên ảnh hóa đơn hoặc bảng biểu...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

    # 2. Nhập câu hỏi
    question = st.text_input("Đặt câu hỏi cho AI (Ví dụ: Tổng tiền là bao nhiêu?):")

    if st.button("PHÂN TÍCH"):
        if question:
            with st.spinner("Donut đang 'nhai' dữ liệu..."):
                try:
                    # Gửi request sang FastAPI Backend
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    data = {"question": question}
                    
                    response = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)
                    result = response.json()

                    if result["status"] == "success":
                        st.success("✅ Đã tìm thấy kết quả!")
                        st.subheader(f"🤖 Trả lời: {result['answer']}")
                        
                        # Debug zone
                        with st.expander("Xem dữ liệu chi tiết (JSON)"):
                            st.json(result)
                    else:
                        st.error("Backend phản hồi lỗi!")
                except Exception as e:
                    st.error(f"Không thể kết nối đến Backend: {e}")
        else:
            st.warning("Vui lòng nhập câu hỏi trước khi phân tích!")

st.markdown("---")