from pathlib import Path
import json
import re
from transformers import AutoTokenizer

INPUT_PATH = Path("data/processed/rav4_manual.json")

FOOTER_PATTERN = r"19 RAV4 HV OM42D12E\s+\d+/\d+/\d+\s+\d+:\d+\s+Page\s+\d+"

HYPHENATION_PATTERN = r"-\n(?=\w)"

SECTION_PATTERN = r"^\d+-\d+\..*"

CHUNK_SIZE = 500

OVERLAP_SIZE = 75

STEP_SIZE = CHUNK_SIZE - OVERLAP_SIZE

TOKENIZER = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")

def split_into_chunks(text):
    tokens = TOKENIZER.encode(text, add_special_tokens=False)
    local_chunks = []
    for start in range(0, len(tokens), STEP_SIZE):
        stop = start + CHUNK_SIZE
        chunk_tokens = tokens[start : stop]
        chunk_text = TOKENIZER.decode(chunk_tokens)
        local_chunks.append(chunk_text)
    return local_chunks
                       


with open(INPUT_PATH, "r", encoding="utf-8")as file:
    pages_data = json.load(file)

    current_section = None
    section_text = ""
    section_chunks = []
    chunks = []

    for page in pages_data:

        text = page["text"]

        text = re.sub(FOOTER_PATTERN, "", text)
        text = re.sub(HYPHENATION_PATTERN, "", text)

        sections = re.findall(SECTION_PATTERN, text, flags=re.MULTILINE)

        for section in sections:
            if section.split()[-1].isdigit():
                page_num = int(section.split()[-1])
                if page_num == page["page"]:
                    clean_header = " ".join(section.split()[:-1])

                    if section_text:
                        section_chunks = split_into_chunks(section_text)
                        print(section_chunks)
                    current_section = clean_header
                    text = text.replace(section, "")

        section_text += " " + text

        # for page in pages_data:

        #     text = page["text"]

        #     text = re.sub(FOOTER_PATTERN, "", text)

        #     if page["page"] == 313:
        #         print(text)
        #         break

        # for section in sections:
        #     if not section.split()[-1].isdigit():
        #         if page["page"] > 4:
        #             print(page["page"], section)
