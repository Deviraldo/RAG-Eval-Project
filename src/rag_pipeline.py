from src.reranker import RerankingRetriever
from src.generator import generate


class RagPipeline:
    def __init__(self, fetch_k=100, top_k=5):
        # one retriever instance — loads the store + reranker model once
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)
    
    def invoke(self, query: str) -> dict:
        # 1. RETRIEVE: shortlist with BM25, then rerank the best candidates
        docs = self.retriever.invoke(query)

        # 2. UNPACK: generator wants list[str], the triad wants the same strings
        context = [doc.page_content for doc in docs]

        # 3. GENERATE: grounded answer from the retrieved context
        answer = generate(query, context)

        # return all three legs of the triad so the eval harness can score them
        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


# Interactive terminal use: python -m src.rag_pipeline
if __name__ == "__main__":
    rag = RagPipeline()
    print("Ask questions about the course transcripts. Type 'quit' to exit.")

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if not query:
            continue

        result = rag.invoke(query)
        print(f"\nAssistant: {result['answer']}")
