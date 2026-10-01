import math
import re
from collections import Counter

from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.retriever import load_transcripts

# Local query-passage relevance model; no embedding API is needed.
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "what", "when", "where", "which", "who", "why", "with",
}


def _tokens(text):
    return [word for word in TOKEN_RE.findall(text.lower()) if word not in STOP_WORDS]


class RerankingRetriever:
    def __init__(self, fetch_k=100, top_k=5):
        # BM25 first finds likely passages cheaply; the cross-encoder then ranks
        # those passages using query and passage together.
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.documents = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ).split_documents(load_transcripts())
        if not self.documents:
            raise RuntimeError("No transcript chunks are available for retrieval")

        self.doc_tokens = [_tokens(doc.page_content) for doc in self.documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) or 1
        self.doc_freq = Counter(
            term for tokens in self.doc_tokens for term in set(tokens)
        )
        self.reranker = CrossEncoder(CROSS_ENCODER)

    def _bm25_candidates(self, query):
        query_terms = set(_tokens(query))
        if not query_terms:
            return self.documents[:self.fetch_k]

        total_docs = len(self.documents)
        k1, b = 1.5, 0.75
        scored = []
        for index, tokens in enumerate(self.doc_tokens):
            frequencies = Counter(tokens)
            length = self.doc_lengths[index]
            score = 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = self.doc_freq[term]
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                score += idf * tf * (k1 + 1) / (
                    tf + k1 * (1 - b + b * length / self.avg_doc_length)
                )
            if score > 0:
                scored.append((score, index))

        scored.sort(reverse=True)
        return [self.documents[index] for _, index in scored[:self.fetch_k]]

    def invoke(self, query):
        if not query or not query.strip():
            return []

        candidates = self._bm25_candidates(query)
        if not candidates:
            return []
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_k]]
