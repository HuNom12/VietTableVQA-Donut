1. Tổng quan
File data_loader.py đảm nhận vai trò cầu nối giữa dữ liệu thô (ảnh bảng biểu .png và nhãn .jsonl) trên ổ cứng và GPU. Nhiệm vụ chính là chuyển đổi văn bản và hình ảnh thành các Tensor (mảng số nhiều chiều) theo đúng định dạng mà mô hình Donut yêu cầu: pixel_values cho ảnh và labels cho văn bản.
2. Cơ chế cốt lõi
Trong xử lý ảnh độ phân giải cao, việc nạp toàn bộ Dataset vào RAM hệ thống thường là nguyên nhân chính gây ra hiện tượng tràn bộ nhớ (Out-of-Memory). Để giải quyết bài toán này, dự án áp dụng chiến thuật Lazy Loading (Tải lười) thông qua hàm set_transform() của thư viện Hugging Face.

Chiến thuật này hoạt động dựa trên hai nguyên tắc tối ưu sau:

2.1. Từ bỏ phương pháp tiền xử lý toàn cục (.map())
Thay vì duyệt qua toàn bộ dataset, biến đổi tất cả ảnh thành tensor và lưu thành một file cache khổng lồ trên ổ cứng, hệ thống sẽ bỏ qua hoàn toàn bước này. Việc này giúp tiết kiệm hàng chục GB không gian lưu trữ và đảm bảo RAM máy tính không bị quá tải khi kích thước bộ dữ liệu (dataset) tiếp tục mở rộng.

2.2. Xử lý "On-the-fly" để bảo vệ giới hạn VRAM
Hàm set_transform() đóng vai trò như một bộ lọc kích hoạt theo thời gian thực (On-the-fly). Nó chỉ gọi hàm biến đổi transform_fn ngay tại thời điểm mô hình yêu cầu nạp một Batch dữ liệu cụ thể để tính toán.

Nhờ cơ chế này, hệ thống chỉ tiêu tốn tài nguyên cho đúng số lượng ảnh đang nằm trong Batch hiện tại. Ngay sau khi xử lý xong và hoàn tất quá trình cập nhật trọng số, vùng nhớ VRAM đó sẽ tự động được giải phóng để đón Batch tiếp theo. Đây là yếu tố then chốt giúp quá trình huấn luyện diễn ra mượt mà và an toàn.
3. Phân tách logic hàm transform_fn
Hàm này được thiết kế để xử lý song song hai luồng dữ liệu khi một batch được gọi:

3.1. Vision Processing (Xử lý ảnh)
Quy trình: Đọc ảnh từ đường dẫn vật lý -> Chuyển sang không gian màu RGB (tránh lỗi ảnh xám/RGBA) -> Đưa qua processor.

Đầu ra: pixel_values (Tensor hình ảnh đã được Resize và Normalize chuẩn hóa).

3.2. Text Tokenization (Xử lý nhãn JSON)
Quy trình: Nhận chuỗi ground_truth (đã được format sẵn dưới dạng từ điển JSON trong file metadata.jsonl) -> Mã hóa thành ID số bằng tokenizer.

Cấu hình an toàn:

max_length=128: Cắt cụt (Truncate) hoặc đệm (Pad) các chuỗi văn bản về cùng một độ dài cố định. Giúp các Tensor có kích thước đồng nhất để xếp thành Batch.

add_special_tokens=False: Không tự động thêm các token như <s> hay </s> của bộ dịch thuần túy, vì chúng ta đã chèn <s_tablevqa> ở mức Model Architecture.

Đầu ra: labels (Dãy số đại diện cho câu trả lời).
4. Ưu điểm của kiến trúc imagefolder
Thay vì tự viết code bóc tách đường dẫn thư mục và xử lý ngoại lệ (try-except) cho file JSON, dự án tận dụng class imagefolder của Hugging Face.

Tự động quét các thư mục con (ví dụ: statistics, financial, excel).

Tự động ánh xạ file_name trong file metadata.jsonl với file ảnh thực tế trên ổ cứng.

Đảm bảo cấu trúc dữ liệu luôn "sạch" trước khi vào Pipeline.