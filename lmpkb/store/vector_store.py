import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
from PIL.Image import Image

from lmpkb.ingest.pdf import load_pdf


@dataclass
class RetrievedPage:
    image: Image
    source: str
    page_number: int


class VectorStore:

    def __init__(self, path: str, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.directory =  Path(os.environ.get("LMPKB_CONFIG")).parent 

    def index(self, embedding: list[float], source: str, page_number: int) -> None:
        self.collection.upsert(
            ids=[f"{source}::{page_number}"],
            embeddings=[embedding],
            metadatas=[{"source": source, "page_number": page_number}],
        )

    def retrieve(self, query_embedding: list[float], top_k: int) -> list[RetrievedPage]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas"],
        )
        pages = []
        for meta in results["metadatas"][0]:
            source = self.directory / meta["source"]
            page_number = meta["page_number"]
            image = load_pdf(source)[page_number][1]
            pages.append(RetrievedPage(image=image, source=str(source), page_number=page_number))
        return pages
