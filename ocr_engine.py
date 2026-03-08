from paddleocr import PaddleOCR

def load_ocr():
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="vi",
        use_gpu=True
    )
    return ocr


def run_ocr(ocr, image_path):
    result = ocr.ocr(image_path)
    return result