import io
import torch
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import sys
from pathlib import Path

app = FastAPI(title="VietTableVQA PRO API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. HẬU XỬ LÝ

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


# 2. KHỞI TẠO HỆ THỐNG

MODEL_PATH = "./checkpoints/best_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Đang nạp Best Model lên {device.upper()}...")
processor = DonutProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
model.eval()
print("Hệ thống sẵn sàng")

bad_words_ids = [
    processor.tokenizer.encode("{", add_special_tokens=False),
    processor.tokenizer.encode('"', add_special_tokens=False),
    processor.tokenizer.encode("question", add_special_tokens=False),
    processor.tokenizer.encode("answer", add_special_tokens=False)
]

root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from src.utils.image_processor import ImageFilter
img_filter = ImageFilter()

@app.post("/predict")
async def predict_invoice(
    question: str = Form(..., description="Câu hỏi truy vấn"), 
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh.")
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    image = img_filter.apply_all_filters(image)

    pixel_values = processor(
        images=image, 
        return_tensors="pt", 
        size={"height": 960, "width": 720}
    ).pixel_values.to(device)

    prompt = f"<s_question>{question}</s_question><s_answer>"
    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    # 4. SINH KẾT QUẢ 
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=128,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=bad_words_ids, 
            num_beams=4,             
            repetition_penalty=1.2,  
            early_stopping=True,
            return_dict_in_generate=True,
        )

    # 5. GIẢI MÃ 
    sequence = processor.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
    clean_seq = sequence.replace(processor.tokenizer.pad_token, "").replace(processor.tokenizer.eos_token, "").strip()
    
    # Bóc tách phần answer
    if "<s_answer>" in clean_seq:
        raw_answer = clean_seq.split("<s_answer>")[-1].split("</s_answer>")[0].strip()
    else:
        raw_answer = clean_seq

    final_ans = clean_vietnamese_text(raw_answer)
    final_ans = final_answer_format(final_ans)
    final_ans = re.sub(r'<.*?>', '', final_ans).strip() 

    return {
        "status": "success",
        "question": question,
        "answer": final_ans if final_ans else "Mô hình không tìm thấy câu trả lời.",
        "debug_raw": clean_seq 
    }