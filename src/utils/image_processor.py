import cv2
import numpy as np
from PIL import Image

class ImageFilter:
    def __init__(self):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Giúp làm rõ chữ trong điều kiện ánh sáng không đều
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def pil_to_cv2(self, pil_image):
        """Chuyển từ PIL Image sang OpenCV format"""
        open_cv_image = np.array(pil_image)
        # RGB to BGR
        return open_cv_image[:, :, ::-1].copy()

    def cv2_to_pil(self, cv2_image):
        """Chuyển từ OpenCV sang PIL format (để Donut model nhận diện)"""
        # BGR to RGB
        img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def deskew(self, image):
        """Tự động xoay thẳng ảnh nếu bảng biểu bị nghiêng"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Xử lý góc xoay của OpenCV
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def enhance_contrast(self, image):
        """Làm rõ nét chữ và bảng biểu"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        limg = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced

    def apply_all_filters(self, pil_image):
        """Quy trình 'mông má' ảnh tổng lực"""
        # 1. Chuyển hệ
        cv2_img = self.pil_to_cv2(pil_image)
        
        # 2. Xoay thẳng (Quan trọng cho TableVQA)
        cv2_img = self.deskew(cv2_img)
        
        # 3. Tăng độ tương phản
        cv2_img = self.enhance_contrast(cv2_img)
        
        # 4. Trả về dạng PIL cho model
        return self.cv2_to_pil(cv2_img)