import regex
from rapidfuzz import fuzz

def find_best_match(target, ocr_pool):
    """
    Thuật toán Fuzzy Regex + RapidFuzz Validation.
    Trả về: (box_large, start_idx, end_idx, total_chars) nếu tìm thấy, ngược lại trả về None
    """
    target_lower = target.lower()
    
    # Dung sai động: Tên ngắn cho sai 3, Tên siêu dài cho sai 4
    max_err = 3 if len(target_lower) <= 15 else 4
    pattern = f"(?e)({target_lower}){{e<={max_err}}}"
    
    best_match_data = None
    best_score = 0

    for word in ocr_pool:
        box_large = word[0]
        full_text_lower = word[1][0].lower()
        
        match = regex.search(pattern, full_text_lower, regex.BESTMATCH)
        
        if match:
            substring = full_text_lower[match.start() : match.end()]
            score = fuzz.ratio(target_lower, substring)
            
            # Phải giống ít nhất 65% mới qua vòng xác duyệt
            if score > best_score and score >= 65:
                best_score = score
                best_match_data = (box_large, match.start(), match.end(), max(len(full_text_lower), 1))
                
    return best_match_data