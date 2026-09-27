from .retriever import Retriever
from .reranker import Reranker

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()

    def retrieve(self, question):
        candidates = self.retriever.search(question)
        reranked_results = self.reranker.rerank(question, candidates)
        return reranked_results

if __name__ == "__main__":
    pipeline = RAGPipeline()

result = pipeline.retrieve("Jak wyłączyć system PCS?")
print(result)