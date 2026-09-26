from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss

INPUT_PATH = Path("data/processed/rav4_chunks.json")

INDEX_PATH = Path("data/processed/rav4_faiss.index")

MODEL_NAME = "intfloat/multilingual-e5-base"

with open(INPUT_PATH, "r", encoding="utf-8")as file:
    chunks = json.load(file)

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer(MODEL_NAME, device=device)

texts = []

for chunk in chunks:
    texts.append("passage: " + chunk["text"])

embeddings = MODEL.encode(
    texts, 
    normalize_embeddings=True, 
    show_progress_bar=True
    )

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(index, str(INDEX_PATH))


print(chunks[0])

# similarity = np.dot(embeddings[0], embeddings[1])
# print(similarity)