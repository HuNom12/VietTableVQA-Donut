import io
import torch
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

app = FastAPI(title="VietTableVQA PRO API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả các nguồn truy cập (để test cho nhanh)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# 1. CÁC HÀM HẬU KỲ "PRO MAX" (BÊ NGUYÊN TỪ INFERENCE CỦA NAM)
# ==========================================================
def clean_vietnamese_text(text):
    text = re.sub(r'(?<=[a-zA-Zà-ỹÀ-Ỹ])\s+(?=[a-zà-ỹ])', '', text)
    text = re.sub(r'\s+(?=[à-ỹ])', '', text)
    text = re.sub(r'(?<=[\d])\s+(?=[\d])', '', text)
    return text

def final_answer_format(text):
    corrections = {
        "cóđơn": "có đơn", "giálớn": "giá lớn", 
        "nhất?": "nhất", "làbao": "là bao"
    }
    for search, replace in corrections.items():
        text = text.replace(search, replace)
    return text

# ==========================================================
# 2. KHỞI TẠO HỆ THỐNG (LOAD MODEL ĐÃ TRAIN)
# ==========================================================
MODEL_PATH = r"E:\HuNamDocument\My First Project\TableVQA_project\checkpoints\best_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Đang đưa 'Best Model' lên {device.upper()}...")
# Load processor từ thư mục gốc để lấy từ điển mới, model từ best_model
processor = DonutProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
model.eval()
print("✅ Hệ thống đã sẵn sàng chiến đấu!")

# Chuẩn bị danh sách từ cấm để tránh lặp JSON rác
bad_words_ids = [
    processor.tokenizer.encode("{", add_special_tokens=False),
    processor.tokenizer.encode('"', add_special_tokens=False),
    processor.tokenizer.encode("question", add_special_tokens=False),
    processor.tokenizer.encode("answer", add_special_tokens=False)
]

@app.post("/predict")
async def predict_invoice(
    question: str = Form(..., description="Câu hỏi truy vấn"), 
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh.")
    
    # Đọc và xử lý ảnh theo size chuẩn của Nam
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    pixel_values = processor(
        images=image, 
        return_tensors="pt", 
        size={"height": 960, "width": 720}
    ).pixel_values.to(device)

    # 3. FIX PROMPT: Khớp 100% với logic huấn luyện
    prompt = f"<s_question>{question}</s_question><s_answer>"
    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    # 4. SINH KẾT QUẢ VỚI CHẾ ĐỘ "THIẾT QUÂN LUẬT"
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=128,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=bad_words_ids, 
            num_beams=4,              # Dùng Beam Search xịn như bản inference
            repetition_penalty=1.5,   # Phạt lặp từ
            early_stopping=True,
            return_dict_in_generate=True,
        )

    # 5. GIẢI MÃ VÀ LÀM ĐẸP (CLEANING)
    sequence = processor.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
    clean_seq = sequence.replace(processor.tokenizer.pad_token, "").replace(processor.tokenizer.eos_token, "").strip()
    
    # Bóc tách phần answer
    if "<s_answer>" in clean_seq:
        raw_answer = clean_seq.split("<s_answer>")[-1].split("</s_answer>")[0].strip()
    else:
        raw_answer = clean_seq

    final_ans = clean_vietnamese_text(raw_answer)
    final_ans = final_answer_format(final_ans)
    final_ans = re.sub(r'<.*?>', '', final_ans).strip() # Xóa nốt thẻ rác nếu có

    return {
        "status": "success",
        "question": question,
        "answer": final_ans if final_ans else "Mô hình không tìm thấy câu trả lời.",
        "debug_raw": clean_seq 
    }