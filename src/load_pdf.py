from pathlib import Path
from pypdf import PdfReader
import json

PDF_PATH = Path("data/documents/PZ49X-42D12-PL.pdf")
OUTPUT_PATH = Path("data/processed/rav4_manual.json")

reader = PdfReader(PDF_PATH)

unique_chars = set()
char_map = {
    'à': 'ą',
    'Ñ': 'Ą',
    'ç': 'ć',
    'å': 'Ć',
    '´': 'ę',
    '¢': 'Ę',
    '∏': 'ł',
    '¸': 'Ł',
    'ƒ': 'ń',
    'Ê': 'ś',
    'Â': 'Ś',
    '˝': 'ż',
    '˚': 'Ż',
    'ê': 'ź',
    'è': 'Ź',
    '¡': '→',
}

pages_data = []

def normalize_text(text):
    for wrong_char, correct_char in char_map.items():
        text = text.replace(wrong_char, correct_char)
    return text

print(f"Liczba stron: {len(reader.pages)}")

for page_num, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    text = normalize_text(text)

    page_data = {
        "page": page_num,
        "source": PDF_PATH.name,
        "text": text,
    }

    pages_data.append(page_data)

with open(OUTPUT_PATH, "w", encoding="utf-8")as file:
    json.dump(pages_data, file, ensure_ascii=False, indent=2)

print(f"Liczba zapisanych stron: {len(pages_data)}")

# page_num = reader.pages[99]
# text = page_num.extract_text()
# print(text)

# wrong_char = 'µ'

# for page in reader.pages:
#     text = page.extract_text()
#     if wrong_char in text:
#         lines = text.split('\n')
#         for line in lines:
#             if wrong_char in line:
#                 print(line)
#     for char in text:
#         unique_chars.add(char)

# print(unique_chars)

# text_test = normalize_text(text)
# print(text_test)

# doc = pymupdf.open(PDF_PATH)
# page = doc[100]
# print(page.get_text())