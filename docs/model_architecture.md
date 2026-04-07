1. Tổng quan
File model.py chịu trách nhiệm khởi tạo và cấu hình mô hình Donut (Document Understanding Transformer). Thay vì sử dụng OCR truyền thống (quét chữ riêng, hiểu cấu trúc riêng), Donut đi theo hướng End-to-End Vision-to-Text, giúp trích xuất thông tin trực tiếp từ ảnh bảng biểu sang định dạng văn bản có cấu trúc.

Model Base: naver-clova-ix/donut-base

Kiến trúc: Transformer-based Vision Encoder-Decoder.
2. Thành phần cốt lõi
2.1. Donut processor
Đây là "bộ phận tiếp nhận", đóng hai vai trò song song:

Image Processor: Chuẩn hóa ảnh (resize, normalize) để phù hợp với Encoder.

Tokenizer: Mã hóa văn bản thành các dãy ID số để Decoder có thể xử lý. Việc sử dụng chung một Processor giúp đảm bảo sự đồng bộ tuyệt đối giữa dữ liệu hình ảnh và nhãn (labels).
2.2. Vision Encoder-Decoder
Mô hình hoạt động theo luồng:

Encoder (Swin Transformer): Nhìn vào tấm ảnh bảng biểu và trích xuất các đặc trưng hình ảnh (features) thành các vector không gian.

Decoder (BART-like): Nhận các vector từ Encoder kết hợp với các token văn bản để "dịch" tấm ảnh đó thành câu trả lời (answer) theo cấu trúc JSON.
3. Các tùy chỉnh quan trọng
Special Tokens: Thêm <s_tablevqa>, <s_answer>, v.v. để mô hình nhận diện được các "mốc" ranh giới trong câu trả lời.

Resize Token Embeddings: Cần thiết sau khi thêm Special Tokens để mở rộng ma trận trọng số của lớp Decoder, tránh lỗi lệch chỉ mục (Index Error).

Decoder Start Token: Ép mô hình luôn bắt đầu chuỗi phản hồi bằng token <s_tablevqa>, giúp kích hoạt đúng "chế độ" giải toán bảng biểu.
4. Tối ưu hóa cho phần cứng (Hardware Logic)
Device Mapping: Tự động chuyển Model lên cuda nếu có sẵn.

Memory Management: Cấu hình pad_token_id giúp tối ưu hóa bộ nhớ đệm trong quá trình sinh văn bản (Generation), giảm thiểu lãng phí VRAM khi xử lý các chuỗi ngắn.
5. Luồng hoạt động (Workflow)
Gọi get_model_and_processor().

Thêm token đặc biệt cho tiếng Việt.

Mở rộng Decoder.

Đẩy toàn bộ mô hình lên VRAM của GPU.