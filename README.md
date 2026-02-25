# Personal Knowledge Base

A local, multimodal RAG system for querying a personal document collection.

PDFs are embedded as page images using [ColPali](https://github.com/illuin-tech/colpali) and stored in a vector database. A CLI agent retrieves relevant pages and generates answers via a local LLM through [Ollama](https://ollama.com).

## Installation

```bash
poetry install
```

## Usage

**Embed a folder of PDFs:**
```bash
pkb embed <folder>
```

**Query your knowledge base:**
```bash
pkb query "<question>"
```

## Stack

| Component | Library |
|-----------|---------|
| Multimodal embeddings | `colpali-engine` (BiQwen2.5) |
| Vector store | — |
| LLM | Ollama |
| CLI | Typer |
