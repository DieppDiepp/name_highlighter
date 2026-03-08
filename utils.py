import unidecode

def normalize_text(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    return text.strip()