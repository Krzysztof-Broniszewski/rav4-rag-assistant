from .retriever import Retriever
from .reranker import Reranker
from .generator import Generator

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.generator = Generator()

    def retrieve(self, question):
        candidates = self.retriever.search(question)
        reranked_results = self.reranker.rerank(question, candidates)
        return reranked_results

    def answer(self, question):
        results = self.retrieve(question)
        return self.generator.generate(question, results)

if __name__ == "__main__":
    pipeline = RAGPipeline()
    answer = pipeline.answer("Jakie ciśnienie powinno być w oponach?")
    print(answer)