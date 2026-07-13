from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.rag.vector_store import CHROMA_DB_DIR
from typing import List, Dict, Any


class NewsRetriever:
    def __init__(self, persist_dir=None):
        self.persist_dir = persist_dir or str(CHROMA_DB_DIR)
        print("Loading embedding model in retriever...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load persisted Chroma store
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def retrieve_evidence(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top K articles with metadata for a given query.
        """
        results = self.db.similarity_search_with_score(query, k=top_k)

        evidence = []
        for doc, score in results:
            evidence.append(
                {
                    "content": doc.page_content,
                    "title": doc.metadata.get("title", "Unknown Article"),
                    "label": doc.metadata.get("label", "Unknown"),
                    "subject": doc.metadata.get("subject", "general"),
                    "score": float(score),  # Distance score (lower distance = higher similarity)
                }
            )
        return evidence


if __name__ == "__main__":
    import sys

    # Quick dry-run check
    try:
        retriever = NewsRetriever()
        query = "NASA discoveries aliens"
        print(f"Retrieving evidence for: '{query}'")
        evidence = retriever.retrieve_evidence(query, top_k=2)
        for idx, ev in enumerate(evidence):
            print(f"\nEvidence {idx+1}:")
            print(f"Title: {ev['title']} ({ev['label']})")
            print(f"Snippet: {ev['content'][:200]}...")
    except Exception as e:
        print("Retriever test failed (perhaps vector store is not built yet):", str(e))
