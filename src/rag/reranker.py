from sentence_transformers import CrossEncoder
import torch

class Reranker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "BAAI/bge-reranker-v2-m3"
        self.model = CrossEncoder(
            self.model_name, 
            device=self.device
        )

    def rerank(self, question, candidates, top_k=5):
        pairs = []

        for candidate in candidates:
            pairs.append([question, candidate["text"]])

        reranker_scores = self.model.predict(pairs)
        reranked_results = []
        
        for candidate, reranker_score in zip(candidates, reranker_scores):
            result = candidate.copy()
            result["reranker_score"] = float(reranker_score)
            reranked_results.append(result)
        sorted_results = sorted(
            reranked_results,
            key=lambda result: result["reranker_score"],
            reverse=True
        )
        return sorted_results[:top_k]