import os
import re
import glob

from langchain_core.documents import Document

DATA_DIR = "data"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():
    """Load VTT transcripts as documents with session metadata."""
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.vtt")):
        lines = []
        with open(path, encoding="utf-8") as transcript:
            for line in transcript:
                line = line.strip()
                if not line or line == "WEBVTT" or "-->" in line:
                    continue
                lines.append(line)

        text = " ".join(lines)
        session_match = re.search(r"Session[ _]*(\d+)", os.path.basename(path), re.IGNORECASE)
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"session": session_match.group(1) if session_match else os.path.basename(path)},
            ))

    if not docs:
        raise RuntimeError(
            f"No readable .vtt transcripts found in {os.path.abspath(DATA_DIR)}"
        )
    return docs


# 2. BUILD ---- use the local BM25 + cross-encoder retriever.
def build_retriever():
    # Import lazily to avoid a module cycle: reranker uses load_transcripts above.
    from src.reranker import RerankingRetriever
    return RerankingRetriever()


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")
    
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")