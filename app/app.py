import streamlit as st
import requests
from PIL import Image
import io

# 1. Cấu hình trang rộng hơn để tận dụng không gian
st.set_page_config(page_title="VietTableVQA Pro", layout="wide")

# Custom CSS để giao diện nhìn "mượt" hơn
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .result-box { padding: 20px; border-radius: 10px; background-color: #ffffff; border-left: 5px solid #ff4b4b; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 VietTableVQA: Trích xuất Dữ liệu Bảng biểu")
st.markdown("Hệ thống hỗ trợ đọc hiểu hóa đơn, báo cáo tài chính và bảng biểu tiếng Việt.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình & Thông tin")
    st.info("Mô hình: **Donut-v1-bias**\n\nDataset: 400+ ảnh bảng biểu thực tế.")
    
    st.divider()

# --- MAIN CONTENT ---
# Chia làm 2 cột: Trái (Ảnh) - Phải (Câu hỏi & Kết quả)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Dữ liệu đầu vào")
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đang phân tích", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh lên để bắt đầu.")

with col2:
    st.subheader("🔍 Truy vấn dữ liệu")
    
    # Sử dụng session_state để hỗ trợ các câu hỏi gợi ý từ sidebar
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    question = st.text_input(
        "Nhập câu hỏi của bạn:", 
        value=st.session_state.current_question,
        placeholder="Ví dụ: Mã số thuế là gì?"
    )

    if st.button("PHÂN TÍCH HÌNH ẢNH"):
        if uploaded_file and question:
            with st.spinner("Đang trích xuất dữ liệu..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"question": question}
                    
                    response = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)
                    result = response.json()

                    if result.get("status") == "success":
                        result_html = f"""
                        <div class="result-box">
                        <p style='margin-bottom: 0; color: #555;'><strong>🤖 Kết quả dự đoán:</strong></p>
                        <h3 style='margin-top: 5px; color: #ff4b4b;'>{result['answer']}</h3>
                        </div>
                        """
                        st.markdown(result_html, unsafe_allow_html=True)
                        
                        # Hiển thị Score nếu Backend có trả về (để tăng độ tin cậy)
                        if "confidence" in result:
                            st.progress(result["confidence"])
                            st.caption(f"Độ tin cậy: {result['confidence']*100:.2f}%")
                            
                        with st.expander("📄 Chi tiết kỹ thuật (JSON Raw)"):
                            st.json(result)
                    else:
                        st.error("Model không tìm thấy câu trả lời phù hợp.")
                except Exception as e:
                    st.error(f"Lỗi kết nối Backend: {e}")
        elif not uploaded_file:
            st.warning("Bạn chưa tải ảnh lên kìa!")
        else:
            st.warning("Đừng quên nhập câu hỏi nhé!")

st.divider()
st.caption("Sinh viên thực hiện: Trần Hữu Nam")