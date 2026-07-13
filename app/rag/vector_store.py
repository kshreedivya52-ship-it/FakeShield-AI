import argparse
import pandas as pd
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config.settings import OUTPUT_DIR
from app.utils.data_loader import DataLoader as ProjectDataLoader

CHROMA_DB_DIR = OUTPUT_DIR / "chroma_db"


class VectorStoreBuilder:
    def __init__(self, persist_dir=None):
        self.persist_dir = persist_dir or str(CHROMA_DB_DIR)
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        # Load local lightweight sentence-transformers embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def build_store(self, limit_data=500):
        print("Loading dataset...")
        loader = ProjectDataLoader()
        df = loader.load_dataset()

        df["text"] = df["text"].fillna("")
        df["title"] = df["title"].fillna("")
        df = df[df["text"].str.strip() != ""]

        if limit_data and limit_data > 0:
            df = df.sample(n=min(len(df), limit_data), random_state=42).reset_index(drop=True)
            print(f"Subsampled to {len(df)} articles for vector store indexing speed.")

        print("Chunking documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        documents = []
        metadatas = []

        for idx, row in df.iterrows():
            text = row["text"]
            title = row["title"]
            label = int(row["label"])
            subject = row.get("subject", "general")

            chunks = text_splitter.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append(
                    {
                        "title": title,
                        "label": "REAL" if label == 1 else "FAKE",
                        "subject": subject,
                        "chunk_id": chunk_idx,
                        "article_id": idx,
                    }
                )

        print(f"Total chunks created: {len(documents)}")
        print(f"Indexing in ChromaDB at {self.persist_dir}...")

        # Build and persist Chroma Vector DB
        vectordb = Chroma.from_texts(
            texts=documents,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.persist_dir,
        )
        print("Vector store successfully built.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Chroma vector store")
    parser.add_argument("--limit-data", type=int, default=500, help="Number of articles to index")
    args = parser.parse_args()

    builder = VectorStoreBuilder()
    builder.build_store(limit_data=args.limit_data)
