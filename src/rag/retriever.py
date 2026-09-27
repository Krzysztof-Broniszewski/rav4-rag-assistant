from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer
import torch

class Retriever:
    def __init__(self):
        self.chunks_path = Path("data/processed/rav4_chunks.json")
        self.index_path = Path("data/processed/rav4_faiss.index")
        with open(self.chunks_path, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)
        self.index = faiss.read_index(str(self.index_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "intfloat/multilingual-e5-base"
        self.model = SentenceTransformer(
            self.model_name, 
            device=self.device
        )

    def search(self, question, k=20):
        query = "query: " + question
        query_embedding = self.model.encode(
            [query], 
            normalize_embeddings=True, 
            show_progress_bar=False
            )
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for indice, score in zip(indices[0], scores[0]):
            chunk = self.chunks[indice]
            result = {
                "chunk_id": int(indice),
                "section": chunk["section"],
                "text": chunk["text"],
                "retrieval_score": float(score),
            }
            results.append(result)
        return results

if __name__ == "__main__":
    retriever = Retriever()

# results = retriever.search("Jak wyłączyć system PCS?")
# print(results)