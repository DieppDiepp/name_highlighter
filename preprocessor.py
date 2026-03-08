import cv2

def preprocess_gray(img):
    """Tiền xử lý 1: Kênh Xám (Giỏi bắt các tên cũ)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def preprocess_green(img):
    """Tiền xử lý 2: Kênh Xanh Lá (Giỏi bắt tên dài, mờ trên nền hồng)"""
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(g)
    return cv2.merge((enhanced, enhanced, enhanced))