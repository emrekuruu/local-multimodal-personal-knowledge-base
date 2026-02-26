# Local Multimodal Personal Knowledge Base

Local multimodal RAG for PDFs. Pages are embedded as images via VoyageAI, stored in Chroma, and queried through a local vision LLM via Ollama. Fully CLI-driven.

## Stack

Defaults — all swappable via CLI flags or env vars.

| Component | Default |
|---|---|
| Embeddings | VoyageAI (`voyage-multimodal-3`) |
| Vector store | Chroma |
| LLM | Ollama (`qwen3-vl:4b`) |
| CLI | Typer |

## Setup

```bash
poetry install
```

Add to `.env`:
```
VOYAGE_API_KEY=your_key_here
```

Pull the model:
```bash
ollama pull qwen3-vl:4b
```

## Repo Structure

```
lmpkb/
├── cli.py                # entry point — embed / retrieve / query commands
├── ingest/
│   └── pdf.py            # PDF → page images (PyMuPDF)
├── embed/
│   └── embedder.py       # VoyageAI image + query embeddings
├── store/
│   └── vector_store.py   # Chroma persistence + retrieval
└── agent/
    └── agent.py          # ChatOllama — question + images → answer
```

## Usage

### `embed` — index a folder of PDFs

```bash
lmpkb embed ./papers/
```
```
Found 3 PDF(s)
Embedding attention-is-all-you-need.pdf...
Embedding rag-survey.pdf...
Embedding colpali.pdf...
Done.
```

<video src="assets/embed.mov" controls></video>

### `retrieve` — inspect what pages would be retrieved

Useful for debugging retrieval quality without calling the LLM.

```bash
lmpkb retrieve "What datasets were used for evaluation?" --top-k 3
```
```
Retrieved pages:
  1. rag-survey.pdf  —  page 4
  2. rag-survey.pdf  —  page 11
  3. colpali.pdf  —  page 2
```

<video src="assets/retrieve.mp4" controls></video>

### `query` — retrieve + generate an answer

```bash
lmpkb query "What datasets were used for evaluation?"
```
```
Retrieved pages:
  1. rag-survey.pdf  —  page 4

Answer:
The paper evaluates on Natural Questions, TriviaQA, and WebQuestions ...
```

<table>
<tr>
<td><video src="assets/query.mp4" controls></video></td>
<td><img src="assets/correct_page.png"></td>
</tr>
</table>

Options:

| Flag | Default | Description |
|---|---|---|
| `--top-k` | `3` (retrieve) / `1` (query) | Number of pages to retrieve |
| `--model` | `qwen3-vl:4b` | Ollama model (overrides `OLLAMA_MODEL` env var) |
