1. Tổng quan (Overview)
File train.py là điểm hội tụ của toàn bộ hệ thống. Nó chịu trách nhiệm kết nối bộ não (model.py) với hệ tiêu hóa dữ liệu (data_loader.py), đồng thời thiết lập các chiến thuật huấn luyện (Hyperparameters) thông qua Seq2SeqTrainer.

Mục tiêu cốt lõi của cấu hình này không chỉ là giúp mô hình hội tụ (giảm Loss), mà còn là nghệ thuật "lách luật" phần cứng để huấn luyện an toàn trên không gian VRAM 6GB mà không bị lỗi Out-of-Memory (OOM).
2. Chiến lược Quản lý Bộ nhớ (VRAM Optimization)
Để vượt qua rào cản phần cứng, cấu hình Seq2SeqTrainingArguments áp dụng bộ ba kỹ thuật tối ưu hóa mức sâu:

2.1. Gradient Accumulation (Cộng dồn đạo hàm)
Thay vì đẩy một Batch Size lớn (ví dụ: 8 ảnh) vào GPU cùng lúc gây tràn bộ nhớ, hệ thống ép per_device_train_batch_size=1. Mô hình sẽ tính toán sai số cho từng ảnh một, nhưng chưa cập nhật trọng số ngay. Nó sẽ "tích lũy" (accumulate) đạo hàm của 8 bước liên tiếp (gradient_accumulation_steps=8), sau đó mới thực hiện một bước tối ưu chung. Kỹ thuật này mang lại hiệu quả học tập tương đương Batch Size 8 nhưng chỉ tiêu tốn VRAM của 1 ảnh.

2.2. Mixed Precision Training (fp16=True)
Theo mặc định, các tính toán AI sử dụng số thực dấu phẩy động 32-bit (FP32). Bằng cách kích hoạt fp16=True, mô hình chuyển sang sử dụng số thực 16-bit cho các tính toán trung gian. Việc này tận dụng tối đa sức mạnh của các nhân Tensor Core trên kiến trúc card RTX, giúp cắt giảm 50% lượng VRAM tiêu thụ và tăng tốc độ huấn luyện lên xấp xỉ 2 lần mà không làm giảm độ chính xác của mô hình.
3. Tham số Huấn luyện
3.1. Learning Rate (2e-5)Tốc độ học được thiết lập ở mức độ vừa phải ($2 \times 10^{-5}$). Với mô hình đã được Pre-train như Donut, một Learning Rate quá lớn sẽ phá vỡ các trọng số đã được học trước đó, gây ra hiện tượng "thảm họa quên" (Catastrophic Forgetting).3.2. Epochs & Loggingnum_train_epochs=10: Cho phép mô hình duyệt qua toàn bộ tập dữ liệu 10 lần. Đây là con số khởi điểm an toàn để quan sát khả năng hội tụ.save_strategy="epoch": Tạo một bản sao lưu (Checkpoint) tại ổ cứng sau mỗi vòng huấn luyện. Nếu hệ thống bị gián đoạn, quá trình train có thể được phục hồi từ Checkpoint gần nhất thay vì phải chạy lại từ đầu.
2.3. Gradient Checkpointing
Đây là kỹ thuật đánh đổi tốc độ lấy không gian lưu trữ. Trong quá trình lan truyền xuôi (Forward Pass), mô hình không lưu lại toàn bộ các biến kích hoạt (Activations) ở mọi lớp. Thay vào đó, nó chỉ lưu lại một vài "trạm kiểm soát" (Checkpoints). Khi thực hiện lan truyền ngược (Backward Pass), các biến bị thiếu sẽ được tính toán lại. Điều này làm quá trình train chậm đi khoảng 20% nhưng giải phóng một lượng VRAM khổng lồ.
