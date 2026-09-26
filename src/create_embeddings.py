from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import torch

# print(torch.cuda.is_available())

INPUT_PATH = Path("data/processed/rav4_chunks.json")

MODEL_NAME = "intfloat/multilingual-e5-base"

with open(INPUT_PATH, "r", encoding="utf-8")as file:
    chunks = json.load(file)

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer(MODEL_NAME, device=device)

# print(MODEL.device)

texts = []

for chunk in chunks:
    texts.append("passage: " + chunk["text"])

embeddings = MODEL.encode(
    texts, 
    normalize_embeddings=True, 
    show_progress_bar=True
    )

print(embeddings.shape)