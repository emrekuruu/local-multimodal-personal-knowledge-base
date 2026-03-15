# Local Multimodal Personal Knowledge Base

Local multimodal RAG for personal documents.

A multi-hop multimodal RAG system to chat with your PDFs locally, using iterative retrieval and grounded answers from page-level evidence.

## Overview

This project turns your PDF collection into a searchable multimodal knowledge base.
It indexes each page as an image and exposes it to an autonomous reasoning agent that decides its own plan for answering based on available tools.
The agent is goalless by design — it reads tool descriptions to decide what to do, rather than following a hardcoded retrieval-first policy.

The agent has four tools:
- **`retrieve`** — semantic search over your local document index, returns page images
- **`web_search`** — live web search via Tavily for information not in your documents
- **`code_exec`** — executes Python for calculations, algorithms, and data processing
- **`final_answer`** — signals completion and delivers the answer

The agent can call multiple tools in a single step (parallel tool calls), so it can fan out several searches simultaneously and synthesize results in the next turn.

The pipeline will:
1. Convert PDF pages to images
2. Embed pages through a multimodal embedder
3. Store vectors in a vector store
4. Run an autonomous tool-calling loop until `final_answer` is called

## Features

- Parallel tool calls 
- PDF page-level ingestion and indexing
- Image-first retrieval over document pages
- Live web search via Tavily 
- Python code execution for calculations and data processing 

## Requirements

- Python `>=3.11,<3.14`
- Poetry
- Provider credentials:
  - `VOYAGE_API_KEY` — VoyageAI key for multimodal embeddings
  - `OPENAI_API_KEY` — if using `openai` models
  - `TAVILY_API_KEY` — for the web search tool
  - Ollama running locally if using `ollama` models

## Installation

```bash
poetry install
```

If you use Ollama, pull your model first:

```bash
ollama pull qwen3-vl:4b
```

## Configuration

Runtime config is resolved from:
1. CLI flags
2. YAML file (`--config`, or auto-detected `lmpkb.yaml` )

There are no fallback defaults for agent runtime settings. Missing required fields raise an error.

### `lmpkb.yaml` example

OpenAI format:

```yaml
model:
  type: openai        
  name: gpt-5.2
  reasoning:
    effort: medium    # none|minimal|low|medium|high|xhigh
    summary: detailed # auto|concise|detailed (optional)

retrieve:
  top_k: 3
```

Ollama format:

```yaml
model:
  type: ollama
  name: qwen3-vl:4b
  reasoning: false    # true|false

retrieve:
  top_k: 3
```

## CLI

### `embed`

Indexes all PDFs under a folder.

![embed demo](assets/embed.gif)

### `retrieve`

Retrieves top matching pages for a question, without generation.

![retrieve demo](assets/retrieve.gif)

### Chat Mode (`lmpkb`)

Running the package name directly starts persistent chat mode:

```bash
lmpkb
```

Chat mode reuses the saved session history between turns, so you can converse continuously.
Exit with `/bye` or `Ctrl+C`.
Use `/help` to list commands.
Use `/clear` to reset memory and clear the visible chat.
Use `/verbose` to toggle step/reasoning traces.

You can override YAML settings when starting chat:

```bash
lmpkb --model-type openai --model gpt-5.2 --reasoning medium --top-k 3
```

#### Example Usage

**Needed Context 1 (Page 14)**

<p align="center">
  <img src="assets/needed_context_1.png" alt="multi hop 1" width="700" />
</p>


**Needed Context 2 (Page 7)**

<p align="center">
  <img src="assets/needed_context_2.png" alt="multi hop 2" width="500" />
</p>

**Multi Hop Retrieval Loop (with Subqueries and Code Execuation)**

![Multi Hop Video](assets/multi-hop.gif)

## Key Chat Flags

- `--config`, `-c`: Path to YAML config
- `--model-type`: `ollama` or `openai`
- `--model`: model name
- `--reasoning`:
  - `ollama`: boolean (`true`/`false`)
  - `openai`: effort (`none|minimal|low|medium|high|xhigh`)
- `--top-k`: retrieval count for query

## Project Structure

```text
lmpkb/
├── cli.py                    # Typer commands: chat mode (root) / embed / retrieve
├── ingest/
│   └── pdf.py                # PDF -> page images
├── embed/
│   └── embedder.py           # VoyageAI multimodal embeddings
├── store/
│   └── vector_store.py       # Chroma persistence + retrieval
└── agent/
    ├── agent.py              # Autonomous tool-calling loop
    ├── config.py             # YAML + CLI config resolution/validation
    ├── llm_factory.py        # Provider-specific LLM creation
    ├── tools.py              # retrieve/web_search/code_exec/final_answer tool definitions
    └── ui.py                 # terminal rendering + reasoning parsing
```
