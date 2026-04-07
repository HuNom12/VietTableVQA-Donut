import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import os

# ==========================================================
# 1. HÀM HẬU KỲ "PRO MAX" (DỌN DẸP CHỮ VÀ JSON)
# ==========================================================
def clean_vietnamese_text(text):
    """Gộp các chữ cái bị tách rời nhưng giữ lại khoảng trắng giữa các từ"""
    # Xử lý: 'Thiế tbị' -> 'Thiết bị'
    # Quy tắc: Nếu có khoảng trắng giữa 1 chữ cái và 1 chữ cái viết thường -> Xóa khoảng trắng
    text = re.sub(r'(?<=[a-zA-Zà-ỹÀ-Ỹ])\s+(?=[a-zà-ỹ])', '', text)
    # Xử lý dấu tiếng Việt bị tách: 'đơ ngiá' -> 'đơn giá'
    text = re.sub(r'\s+(?=[à-ỹ])', '', text)
    return text

def post_process_output(raw_sequence, processor):
    """Bóc tách JSON chính chủ từ Donut và làm đẹp kết quả"""
    try:
        # Sử dụng hàm chính chủ của Hugging Face để parse thẻ <s_...> thành Dict
        data = processor.token2json(raw_sequence)
        
        # Nếu data là list (sớ táo quân), lấy phần tử đầu tiên
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        # Tìm câu trả lời trong dict
        answer = ""
        if isinstance(data, dict):
            # Trường hợp 1: Trả về thẳng 'answer'
            if 'answer' in data:
                answer = data['answer']
            # Trường hợp 2: Trả về một list các cặp câu hỏi-trả lời
            elif isinstance(data.get('nm'), list): # nm thường là key mặc định của Donut
                answer = data['nm'][0].get('answer', str(data['nm'][0]))
        
        # Nếu vẫn không tìm thấy, lấy phần thô sau thẻ <s_answer>
        if not answer:
            answer = raw_sequence.split("<s_answer>")[-1].split("</s_answer>")[0]

        # Cuối cùng là dọn dẹp lỗi "lắp bắp"
        final_ans = clean_vietnamese_text(str(answer))
        # Xóa các thẻ rác còn sót lại
        final_ans = re.sub(r'<.*?>', '', final_ans).replace("  ", " ").strip()
        
        return final_answer_format(final_ans)
    except:
        # Phương án dự phòng cuối cùng nếu JSON lỗi
        text = raw_sequence.split("<s_answer>")[-1].split("</s>")[0]
        text = text.replace("<pad>", "").strip()
        return clean_vietnamese_text(text)

def final_answer_format(text):
    """Chỉnh sửa các lỗi dính chữ đặc thù sau khi dùng Regex"""
    # Một số cụm từ hay bị dính do Regex trên quá mạnh, ta sửa thủ công
    corrections = {
        "cóđơn": "có đơn",
        "giálớn": "giá lớn",
        "nhất?": "nhất",
        "làbao": "là bao"
    }
    for search, replace in corrections.items():
        text = text.replace(search, replace)
    return text

# ==========================================================
# 2. CẤU HÌNH HỆ THỐNG
# ==========================================================
PARENT_DIR = "./checkpoints" 
CHECKPOINT_PATH = "./checkpoints/best_model"
IMAGE_PATH = "data/Unlabeled/internet/077_internet.png" 

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Nạp Processor và Model (Đảm bảo đường dẫn chuẩn)
    processor = DonutProcessor.from_pretrained(PARENT_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT_PATH).to(device)
    model.eval()

    # 2. Xử lý ảnh (Giữ nguyên size 960x720 như lúc train)
    image = Image.open(IMAGE_PATH).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt", size={"height": 960, "width": 720}).pixel_values.to(device)

    # 3. FIX PROMPT: Phải khớp 100% với file data_loader.py Nam đã train
    question = "Thiết bị nào trong danh sách thiết bị có đơn giá thấp nhất?" 
    # BỎ THẺ <s_tablevqa> nếu lúc train Nam không dùng nó
    prompt = f"<s_question>{question}</s_question><s_answer>"
    
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # 4. SINH KẾT QUẢ VỚI CHẾ ĐỘ "THIẾT QUÂN LUẬT"
    # Cấm nó viết dấu { hoặc " để nó không nôn ra JSON nữa
    bad_words_ids = [
        processor.tokenizer.encode("{", add_special_tokens=False),
        processor.tokenizer.encode('"', add_special_tokens=False),
        processor.tokenizer.encode("question", add_special_tokens=False),
        processor.tokenizer.encode("answer", add_special_tokens=False)
    ]

    # Sinh kết quả
    print("🧠 Donut đang 'nhai' dữ liệu...")
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=512,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            bad_words_ids=bad_words_ids, 
            num_beams=4,                 
            early_stopping=True
        )

    # Giải mã và làm đẹp
    sequence = processor.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
    clean_seq = sequence.replace(processor.tokenizer.pad_token, "").replace(processor.tokenizer.eos_token, "").strip()
    print("\n" + "="*50)
    print(f"🔍 [DEBUG] Chuỗi RAW từ Model: {clean_seq}")
    print("="*50)
    if "<s_answer>" in clean_seq:
        raw_answer = clean_seq.split("<s_answer>")[-1].split("</s_answer>")[0].strip()
    else:
        raw_answer = clean_seq

    final_answer = clean_vietnamese_text(raw_answer)
    final_answer = final_answer_format(final_answer)
    
    print("\n" + "★"*50)
    print("🎯 KẾT QUẢ TRÍ TUỆ NHÂN TẠO CHUẨN ĐOÁN")
    print("★"*50)
    print(f"❓ CÂU HỎI: {question}")
    print(f"🤖 TRẢ LỜI: {final_answer}")
    print("★"*50)

if __name__ == "__main__":
    run_inference()