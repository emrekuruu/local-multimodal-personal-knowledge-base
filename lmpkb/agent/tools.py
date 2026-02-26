from langchain_core.tools import Tool

from lmpkb.store.vector_store import VectorStore


def build_retrieve_tool(store: VectorStore) -> Tool:
    ...
