import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.components.model import get_model_and_processor
from src.components.data_loader import get_viet_table_dataset
import numpy as np

model, processor, device = get_model_and_processor()

train_dataset, eval_dataset = get_viet_table_dataset("data/VietTableVQA", processor)

training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints",          
    
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,     
    gradient_accumulation_steps=8,     
                                       
    gradient_checkpointing=True,       
    fp16=True,                         
    
    learning_rate=3e-5,                 
    num_train_epochs=10,                    
    logging_steps=10,         
    eval_strategy="epoch",          
    save_strategy="epoch",
    load_best_model_at_end=True,  
    metric_for_best_model="eval_loss",        
    greater_is_better=False,      
    save_total_limit=2,                 
    
    predict_with_generate=True,    
    generation_max_length=128,     
    report_to="none",              
    remove_unused_columns=False      
)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    # Nếu mô hình trả về tuple, lấy phần tử đầu tiên (danh sách ID dự đoán)
    if isinstance(preds, tuple):
        preds = preds[0]
        
    safe_preds = [[max(0, int(token)) for token in seq] for seq in preds]
    safe_labels = [[max(0, int(token)) for token in seq] for seq in labels]
    # Giải mã ID thành chữ, bỏ qua các token đặc biệt (<s>, </s>)
    decoded_preds = processor.batch_decode(safe_preds, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(safe_labels, skip_special_tokens=True)
    # Làm sạch khoảng trắng dư thừa ở 2 đầu chuỗi
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Đo lường Exact Match (Khớp chính xác 100%)
    exact_matches = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred == label)
    exact_match_rate = exact_matches / len(decoded_labels)
    
    print(f"\n[SAMPLE] Dự đoán: '{decoded_preds[0]}' | Thực tế: '{decoded_labels[0]}'")
    
    return {"exact_match": exact_match_rate}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    print(f"🚀 Đang bắt đầu huấn luyện trên thiết bị: {device.upper()}")
    print(f"📍 Checkpoints sẽ được lưu tại: {os.path.abspath('./checkpoints')}")
    
    import torch
    torch.cuda.empty_cache()
    
    trainer.train()
    print("✅ Đang lưu Checkpoint có Eval Loss tốt nhất...")
    trainer.save_model("./checkpoints/best_model")
    processor.save_pretrained("./checkpoints/best_model")
    print("🎉 Hoàn tất!")