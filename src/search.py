from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import ollama

CHUNKS_PATH = Path("data/processed/rav4_chunks.json")
INDEX_PATH = Path("data/processed/rav4_faiss.index")

with open(CHUNKS_PATH, "r", encoding="utf-8")as file:
    chunks = json.load(file)

index = faiss.read_index(str(INDEX_PATH))

MODEL_NAME = "intfloat/multilingual-e5-base"

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer(MODEL_NAME, device=device)

# QUERY = "Jak wyłączyć system PCS?"
# QUERY = "Jak odblokować lewarek zmiany biegów w razie awarii?"
# QUERY = "Jak oblokować ręcznie klapę bagażnika w przypadku rozładowania akumulatora?" 
# QUERY = "Jak wyregulować fotel kierowcy?"
# QUERY = "Jak wymienić żarówkę kierunkowskazu prawy przód?"
# QUERY = "Jaka żarówka do kierunkowskazu przód?"
QUERY = "Jak ugotować bigos?"

query = "query: " + QUERY

query_embedding = MODEL.encode(
    [query], 
    normalize_embeddings=True, 
    show_progress_bar=True
    )

k = 20

scores, indices = index.search(query_embedding, k)

pairs = []

# print(scores)
# print(indices)

# for indice, score in zip(indices[0], scores[0]):
#     print(indice, score)

for indice in indices[0]:
    text = chunks[indice]["text"]

    pairs.append([QUERY, text])

RERANKER = CrossEncoder(RERANKER_NAME, device=device)
reranker_scores = RERANKER.predict(pairs)

reranked_results = zip(indices[0], reranker_scores)
sorted_reranked_results = sorted(
    reranked_results, 
    key= lambda pair: pair[1],
    reverse=True
    )

top_results = sorted_reranked_results[:5]

context_parts = []

for indice, reranker_score in top_results:
    context_parts.append(chunks[indice]["text"])

context = "\n\n".join(context_parts)

LLM_MODEL = "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0"

SYSTEM_PROMPT = """Jesteś asystentem technicznym.
Odpowiadaj na pytanie wyłącznie na podstawie dostarczonego kontekstu.
Jeżeli w kontekście nie ma odpowiedzi, powiedz, że nie znalazłeś
tej informacji w dostarczonej dokumentacji."""

USER_PROMPT = f"""Kontekst:
{context}

Pytanie: 
{QUERY}"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT}
] 

response = ollama.chat(
    model=LLM_MODEL,
    messages=messages
)

print(response["message"]["content"])

# for chunk in chunks:
#     if "PCS" in chunk["text"] and "wyłącz" in chunk["text"]:
#         print(f'{chunk["chunk_id"]}, {chunk["text"]}')
