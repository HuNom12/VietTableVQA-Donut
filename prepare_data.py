import json
import os
from collections import defaultdict

# Đường dẫn chuẩn theo cấu trúc của Nam
DATA_ROOT = "data/VietTableVQA"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
QA_DIR = os.path.join(DATA_ROOT, "QA_pairs")
OUTPUT_JSONL = os.path.join(DATA_ROOT, "metadata.jsonl")

def flatten_multi_domain_data():
    grouped_data = defaultdict(list)
    missing_files_count = 0
    valid_files_count = 0

    if not os.path.exists(QA_DIR):
        print(f"❌ Không tìm thấy thư mục QA_pairs tại: {QA_DIR}")
        return

    for json_file in os.listdir(QA_DIR):
        if not json_file.endswith(".json"): continue
        
        domain = json_file.replace(".json", "")
        json_path = os.path.join(QA_DIR, json_file)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for entry in data:
                    # 1. Lấy tên file nguyên bản
                    if 'file_name' in entry:
                        pure_name = entry['file_name'].split('\\')[-1].split('/')[-1]
                    elif 'images' in entry:
                        # Nếu gặp định dạng có list images, ta kiểm tra từng cái
                        for img_obj in entry['images']:
                            pure_name = img_obj['path'].split('/')[-1]
                            # Kiểm tra thực tế file có tồn tại không
                            actual_path = os.path.join(IMAGES_DIR, domain, pure_name)
                            
                            if os.path.exists(actual_path):
                                img_relative_path = f"images/{domain}/{pure_name}"
                                grouped_data[img_relative_path].extend(entry['question_answer_pairs'])
                                valid_files_count += 1
                            else:
                                missing_files_count += 1
                        continue # Bỏ qua phần xử lý chung bên dưới vì đã xử lý trong loop
                    
                    # 2. Kiểm tra thực tế cho định dạng đơn lẻ
                    actual_path = os.path.join(IMAGES_DIR, domain, pure_name)
                    if os.path.exists(actual_path):
                        img_relative_path = f"images/{domain}/{pure_name}"
                        grouped_data[img_relative_path].append({
                            "question": entry['question'],
                            "answer": entry['answer']
                        })
                        valid_files_count += 1
                    else:
                        missing_files_count += 1
                            
            except Exception as e:
                print(f"⚠️ Lỗi file {json_file}: {e}")

    # Ghi file metadata.jsonl
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for img_path, qa_pairs in grouped_data.items():
            line = {
                "file_name": img_path,
                "ground_truth": json.dumps({"gt_parse": qa_pairs}, ensure_ascii=False)
            }
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
            
    print("-" * 30)
    print(f"✅ Xử lý xong!")
    print(f"📁 Tổng ảnh hợp lệ tìm thấy: {valid_files_count}")
    print(f"🚫 Đã loại bỏ {missing_files_count} đường dẫn ảo (không có ảnh thật)")
    print(f"📍 File lưu tại: {OUTPUT_JSONL}")

if __name__ == "__main__":
    flatten_multi_domain_data()