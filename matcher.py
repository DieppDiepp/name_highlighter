import regex
from rapidfuzz import fuzz
from utils import normalize_text

def find_best_match(target, ocr_pool):
    """
    Thuật toán Fuzzy Regex + RapidFuzz Validation (Phiên bản bao lỗi mất dấu)
    """
    target_lower = target.lower()
    target_norm = normalize_text(target)
    
    # Dung sai động: Cho phép số lỗi bằng 1/3 chiều dài tên 
    max_err = max(3, len(target_lower) // 3)
    
    pattern = f"(?e)({target_lower}){{e<={max_err}}}"
    # Pattern dự phòng: Quét thẳng trên chuỗi không dấu nếu OCR nát quá
    pattern_norm = f"(?e)({target_norm}){{e<={max_err}}}"
    
    best_match_data = None
    best_score = 0

    for word in ocr_pool:
        box_large = word[0]
        full_text = word[1][0]
        full_text_lower = full_text.lower()
        full_text_norm = normalize_text(full_text)
        
        # Thử tìm trên chuỗi có dấu gốc
        match = regex.search(pattern, full_text_lower, regex.BESTMATCH)
        
        # Nếu không thấy, thử tìm trên chuỗi không dấu
        if not match:
            match = regex.search(pattern_norm, full_text_norm, regex.BESTMATCH)
        
        if match:
            start_idx = match.start()
            end_idx = match.end()
            
            # Kiểm tra chéo bằng Rapidfuzz: So sánh 2 chuỗi KHÔNG DẤU
            # OCR có làm rơi rụng dấu thì điểm vẫn cao chót vót
            substring_norm = full_text_norm[start_idx : end_idx]
            score = fuzz.ratio(target_norm, substring_norm)
            
            # Điểm > 65% là duyệt
            if score > best_score and score >= 65:
                best_score = score
                best_match_data = (box_large, start_idx, end_idx, max(len(full_text), 1))
                
    return best_match_data