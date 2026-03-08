import os
import cv2
from ocr_engine import load_ocr
from preprocessor import preprocess_gray, preprocess_green
from matcher import find_best_match

IMAGE_PATH = "/content/drive/MyDrive/name_highlighter/data/image.png"
NAMES_PATH = "/content/drive/MyDrive/name_highlighter/data/names.txt"
OUTPUT_IMAGE = "/content/drive/MyDrive/name_highlighter/output/final_highlighted.png"
OUTPUT_DIR = "/content/drive/MyDrive/name_highlighter/output/final_crops"

def load_names(path):
    with open(path, "r", encoding="utf8") as f:
        return [x.strip() for x in f.readlines() if x.strip()]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    targets = load_names(NAMES_PATH)
    ocr = load_ocr()

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Không tìm thấy ảnh tại: {IMAGE_PATH}")
        return

    scale = 2.0
    ocr_pool = []

    # 1. Thu thập dữ liệu OCR (Ensemble)
    print("Đang quét AI Lần 1 (Chế độ Cân bằng xám)...")
    img_gray = cv2.resize(preprocess_gray(img), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    res_gray = ocr.ocr(img_gray, cls=False)
    if res_gray and res_gray[0]: ocr_pool.extend(res_gray[0])

    print("Đang quét AI Lần 2 (Chế độ Kênh xanh lá)...")
    img_green = cv2.resize(preprocess_green(img), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    res_green = ocr.ocr(img_green, cls=False)
    if res_green and res_green[0]: ocr_pool.extend(res_green[0])

    draw_all = img.copy()
    found_targets = set()

    # 2. Xử lý logic dò tìm và vẽ khung
    for target in targets:
        match_data = find_best_match(target, ocr_pool)
        
        if match_data:
            box_large, start_idx, end_idx, total_chars = match_data
            
            box = [[p[0] / scale, p[1] / scale] for p in box_large]
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            
            box_width = x2 - x1
            part_x1 = int(x1 + (start_idx / total_chars) * box_width)
            part_x2 = int(x1 + (end_idx / total_chars) * box_width)
            
            pad_x, pad_y = 6, 4
            px1 = max(0, part_x1 - pad_x)
            px2 = min(img.shape[1], part_x2 + pad_x)
            py1 = max(0, y1 - pad_y)
            py2 = min(img.shape[0], y2 + pad_y)
            
            found_targets.add(target)
            cv2.rectangle(draw_all, (px1, py1), (px2, py2), (0, 255, 0), 2)
            
            draw_single = img.copy()
            cv2.rectangle(draw_single, (px1, py1), (px2, py2), (0, 0, 255), 3)
            cv2.imwrite(f"{OUTPUT_DIR}/{target}.png", draw_single)

    cv2.imwrite(OUTPUT_IMAGE, draw_all)
    print(f"\nHoàn tất! Highlight thành công {len(found_targets)}/{len(targets)} tên.")
    if len(found_targets) < len(targets):
        print("Các bạn chưa bắt được:", ", ".join(set(targets) - found_targets))

if __name__ == "__main__":
    main()