import os
import json
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

def get_viet_table_dataset(data_dir, processor, max_length=512):
    # Load dataset và ép buộc không dùng cache cũ để tránh lỗi Key
    dataset = load_dataset("imagefolder", data_dir=data_dir, split="train", download_mode="force_redownload")
    
    def transform_fn(examples):
        # 1. Xử lý ảnh
        images = [img.convert("RGB") for img in examples["image"]]
        pixel_values = processor(images,
                                 return_tensors="pt",
                                 size={"height": 960, "width": 720}
                                ).pixel_values
        
        # 2. Xử lý văn bản (Linh hoạt theo cấu trúc file)
        target_sequences = []
        
        # TRƯỜNG HỢP A: File có phím ground_truth (Dạng chuỗi JSON)
        if "ground_truth" in examples:
            for gt_string in examples["ground_truth"]:
                try:
                    gt_data = json.loads(gt_string)
                    # Nếu gt_parse là dict (đã flatten)
                    if isinstance(gt_data["gt_parse"], dict):
                        q = gt_data["gt_parse"].get("question", "")
                        a = gt_data["gt_parse"].get("answer", "")
                    # Nếu gt_parse vẫn là list (chưa flatten)
                    else:
                        q = gt_data["gt_parse"][0].get("question", "")
                        a = gt_data["gt_parse"][0].get("answer", "")
                    target_sequences.append(f"<s_question>{q}</s_question><s_answer>{a}</s_answer>")
                except:
                    target_sequences.append("<s_question></s_question><s_answer></s_answer>")
        
        # TRƯỜNG HỢP B: File đã flatten và để phím question, answer trực tiếp
        elif "question" in examples and "answer" in examples:
            for q, a in zip(examples["question"], examples["answer"]):
                target_sequences.append(f"<s_question>{q}</s_question><s_answer>{a}</s_answer>")
        
        else:
            raise KeyError(f"Không tìm thấy cột dữ liệu phù hợp. Các cột hiện có: {list(examples.keys())}")

        # 3. Tokenize
        labels = processor.tokenizer(
            target_sequences,
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {"pixel_values": pixel_values, "labels": labels}

    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    df_meta = pd.read_json(metadata_path, lines=True)
    unique_images = df_meta["file_name"].unique().tolist()
    train_images, eval_images = train_test_split(unique_images, test_size=0.1, random_state=42)
    train_indices = df_meta[df_meta["file_name"].isin(train_images)].index.tolist()
    eval_indices = df_meta[df_meta["file_name"].isin(eval_images)].index.tolist()
    train_ds = dataset.select(train_indices)
    eval_ds = dataset.select(eval_indices)
    train_ds.set_transform(transform_fn)
    eval_ds.set_transform(transform_fn)
    
    print(f"📊 Tổng số file ảnh gốc: {len(unique_images)}")
    print(f"✅ Tập Train: {len(train_images)} ảnh ({len(train_ds)} câu hỏi)")
    print(f"📝 Tập Eval: {len(eval_images)} ảnh ({len(eval_ds)} câu hỏi)")
    
    return train_ds, eval_ds