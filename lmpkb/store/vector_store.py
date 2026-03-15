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

        config_path = os.environ.get("LMPKB_CONFIG")
        self.directory = Path(config_path).resolve().parent if config_path else Path.cwd()

    def index(self, embedding: list[float], source: str, page_number: int) -> None:
        source_path = str(Path(source).resolve())
        self.collection.upsert(
            ids=[f"{source_path}::{page_number}"],
            embeddings=[embedding],
            metadatas=[{"source": source_path, "page_number": page_number}],
        )

    def retrieve(self, query_embedding: list[float], top_k: int) -> list[RetrievedPage]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas"],
        )
        pages = []
        for meta in results["metadatas"][0]:
            stored_source = Path(meta["source"])
            source = stored_source if stored_source.is_absolute() else self.directory / stored_source
            page_number = meta["page_number"]
            image = load_pdf(source)[page_number][1]
            pages.append(RetrievedPage(image=image, source=str(source), page_number=page_number))
        return pages
