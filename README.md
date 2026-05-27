# VietTableVQA-Donut: Hệ Thống Trích Xuất Dữ Liệu Bảng Biểu Tiếng Việt (Offline On-Premise)

[![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)](https://github.com/HuNom12/VietTableVQA-Donut)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-green.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](./LICENSE)

VietTableVQA-Donut là hệ thống hỗ trợ trích xuất dữ liệu và trả lời câu hỏi tự nhiên từ hình ảnh bảng biểu, báo cáo tài chính và số liệu thống kê bằng Tiếng Việt. Dự án sử dụng kiến trúc tách biệt hoàn toàn bộ khung phần mềm và trọng số mô hình (Decoupled Architecture), đóng gói đa container bằng Docker nhằm mục đích vận hành ngoại tuyến (Offline On-premise). Hệ thống phù hợp cho các bài toán xử lý dữ liệu nội bộ yêu cầu tính bảo mật thông tin cao.

---

## 2. Tính Năng Nổi Bật

- **OCR-Free Parsing:** Sử dụng kiến trúc End-to-End Transformer (Donut Model) để đọc trực tiếp cấu trúc bảng và ngữ nghĩa từ ảnh mà không cần qua các bộ OCR độc lập, giảm thiểu sai số tích lũy từ văn bản thô.
- **Xử Lý Bẫy Dữ Liệu:** Mô hình được fine-tune chuyên biệt để nhận diện và phản hồi các trường hợp khuyết thiếu thông tin trong thực tế bao gồm: dữ liệu ô trống (`None`), thực thể chưa thống kê (`N/A`), hoặc câu hỏi nằm ngoài phạm vi tài liệu (`Not found`).
- **Containerized Microservices:** Hệ thống phân rã độc lập thành các dịch vụ Frontend và Backend riêng biệt. Việc đóng gói độc lập này giúp tối ưu hóa tài nguyên phần cứng, dễ dàng mở rộng và nâng cấp độc lập cấu phần AI Engine và cho phép cập nhật trọng số mô hình mà không ảnh hưởng tới tính liên tục của giao diện người dùng.
- **Vận Hành Offline 100%:** Toàn bộ quy trình từ tiền xử lý dữ liệu cho đến chạy inference mô hình đều diễn ra cục bộ trong Docker Network nội bộ, đảm bảo tính đóng kín và an toàn thông tin tuyệt đối, không rò rỉ dữ liệu ra internet.

---

## 3. Kiến Trúc Hệ Thống / Sơ Đồ Luồng (Architecture & Workflow)

### Mô Tả Cấu Trúc
Hệ thống được chia thành hai dịch vụ độc lập, giao tiếp nội bộ thông qua mạng của Docker:
- **Frontend (Streamlit):** Giao diện Web tương tác cho người dùng cuối.
- **Backend (FastAPI):** Đảm nhiệm logic tiền xử lý ảnh, chuyển đổi tensor lên thiết bị tính toán (CPU/CUDA) và kiểm soát quá trình sinh văn bản bằng Beam Search, Repetition Penalty.

### Sơ Đồ Luồng Dữ Liệu (Workflow Diagram)
![Sơ Đồ Luồng Hoạt Động Của Hệ Thống](./image_0.png)

## 4. Hướng Dẫn Cài Đặt & Khởi Chạy (Installation & Deployment)

Dự án hỗ trợ 2 kịch bản triển khai tùy thuộc vào đối tượng sử dụng:

### 4.1. Dành cho Người Dùng Cuối
- **Yêu cầu:** Máy tính cài sẵn [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- **Khởi chạy:** 
  1. Tải file `RUN.bat` của dự án.
  2. Click đúp chuột vào `RUN.bat`.
  3. *Lưu ý:* Trong lần chạy đầu tiên, script sẽ tự động kéo toàn bộ hệ thống đã đóng gói sẵn từ Docker Hub về máy cục bộ.

### 4.2. Build Từ Mã Nguồn (Dành cho Nhà Phát Triển / AI Engineer)
Phục vụ mục đích xem mã nguồn, tinh chỉnh mô hình và đóng gói lại hệ thống.

**Bước 1: Clone mã nguồn dự án**
```bash
git clone [https://github.com/HuNom12/VietTableVQA-Donut.git](https://github.com/HuNom12/VietTableVQA-Donut.git)
cd VietTableVQA-Donut
```
**Bước 2: Nạp trọng số mô hình (Model Checkpoints)**
Tạo đúng cấu trúc thư mục local và đặt các file trọng số nặng (được chặn bởi file .gitignore) vào đúng đường dẫn sau:

```text
TABLEVQA_PROJECT/
├── data/                  
└── checkpoints/
    └── best_model_bias/   
        ├── config.json
        ├── pytorch_model.bin (hoặc model.safetensors)
        └── tokenizer.json
```
**Vui lòng liên hệ tác giả qua Email/LinkedIn để được cấp quyền truy cập kho lưu trữ mô hình và dữ liệu huấn luyện.**

**Bước 3: Build và khởi chạy môi trường Dev**

```Bash
docker-compose up --build
```

## 5. Hướng Dẫn Sử Dụng
Sau khi hệ thống được khởi chạy thành công, toàn bộ giao tiếp sẽ diễn ra trong mạng nội bộ của thiết bị.

- **Truy cập Giao Diện Web:** Mở trình duyệt web và truy cập vào địa chỉ cục bộ:

👉 http://localhost:8501

Thao tác: Kéo thả hình ảnh bảng biểu vào khung tải lên, nhập câu hỏi tự nhiên bằng Tiếng Việt và nhấn nút để nhận kết quả trích xuất.

- **Truy cập Backend API (Tùy chọn):** Dành cho hệ thống phần mềm thứ 3 muốn gọi API cục bộ:

👉 http://localhost:8000/docs

- **Tắt hệ thống:** Để dừng an toàn và giải phóng tài nguyên GPU, click đúp vào file SHUT.bat (đối với người dùng cuối) hoặc gõ docker-compose down trong Terminal.

## 6. Định Dạng Dữ Liệu & API Endpoints (Data Format & API Specifications)
Dữ liệu đầu vào phục vụ huấn luyện mô hình Donut được tổ chức theo cấu trúc chuỗi JSON nghiêm ngặt (Strict Parsing), bám sát explicit text sau:

```JSON
{
  "file_name": "images/statistics/001_statistics.png", 
  "ground_truth": "{\"gt_parse\": {\"question\": \"So với quý III/2024 thuộc Dự báo tăng trưởng GDP quý III/2025 của Trung Quốc là bao nhiêu?\", \"answer\": \"5,4\"}}"
}
```
Hệ thống cung cấp cổng API hiệu năng cao phục vụ xử lý thời gian thực:

- **Endpoint:** POST /predict

- **Content-Type:** multipart/form-data

- **Tham số đầu vào:**

    + file: File ảnh cần trích xuất dữ liệu (UploadFile)

    + question: Chuỗi văn bản câu hỏi truy vấn (Form)

Định dạng cấu trúc phản hồi mẫu (Response JSON):

```JSON
{
  "status": "success",
  "question": "Nước sản xuất của Máy phân tích huyết học 18 thông số kỹ thuật là nước nào?",
  "answer": "Mỹ",
  "debug_raw": "<s_question>Nước sản xuất của Máy phân tích huyết học 18 thông số kỹ thuật là nước nào?</s_question><s_answer>Mỹ</s_answer>"
}
```
## 7. Tác Giả & Giấy Phép (Contributors & License)
### Thông Tin Liên Hệ
- **Họ và tên:** Trần Hữu Nam

- **Học vấn:** Sinh viên Ngành Khoa học và Kỹ thuật Máy tính - Trường Đại học Bách khoa ĐHQG-HCM

- **GitHub:** HuNom12

Giấy Phép (License)
Mã nguồn của dự án này được phát hành tuân thủ theo các điều khoản của MIT License. Chi tiết vui lòng tham khảo file LICENSE đính kèm trong mã nguồn.
