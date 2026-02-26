from pathlib import Path

import typer

from lmpkb.agent.agent import answer
from lmpkb.embed.embedder import ImageEmbedder
from lmpkb.ingest.pdf import count_pages, load_pdf
from lmpkb.store.vector_store import RetrievedPage, VectorStore

app = typer.Typer()

CHROMA_PATH = ".chroma"
COLLECTION_NAME = "documents"


def _print_retrieved_pages(pages: list[RetrievedPage]) -> None:
    typer.echo(typer.style("Retrieved pages:", bold=True))
    for i, page in enumerate(pages, 1):
        typer.echo(
            f"  {typer.style(str(i), bold=True)}. "
            f"{typer.style(Path(page.source).name, fg=typer.colors.CYAN)}"
            f"  —  page {page.page_number + 1}"
        )
    typer.echo("")


PREVIEW_LIMIT = 5


@app.command()
def embed(folder: Path) -> None:
    pdf_files = list(folder.glob("**/*.pdf"))

    if not pdf_files:
        typer.echo(typer.style("No PDFs found.", fg=typer.colors.RED, bold=True))
        raise typer.Exit()

    page_counts = {p: count_pages(p) for p in pdf_files}
    total_pages = sum(page_counts.values())

    typer.echo("")
    typer.echo(typer.style("── Embed Preview ──────────────────────────", bold=True))
    typer.echo(
        f"  {typer.style(str(len(pdf_files)), fg=typer.colors.CYAN, bold=True)} PDF(s)   "
        f"{typer.style(str(total_pages), fg=typer.colors.CYAN, bold=True)} pages total"
    )
    typer.echo("")

    for pdf_path in pdf_files[:PREVIEW_LIMIT]:
        typer.echo(
            f"  {typer.style(pdf_path.name, fg=typer.colors.CYAN)}"
            f"  {typer.style(str(page_counts[pdf_path]) + ' pages', fg=typer.colors.WHITE)}"
        )

    if len(pdf_files) > PREVIEW_LIMIT:
        typer.echo(typer.style(f"  ... and {len(pdf_files) - PREVIEW_LIMIT} more", fg=typer.colors.BRIGHT_BLACK))

    typer.echo(typer.style("───────────────────────────────────────────", bold=True))
    typer.echo("")

    typer.confirm("Proceed with embedding?", abort=True)
    typer.echo("")

    embedder = ImageEmbedder()
    store = VectorStore(path=CHROMA_PATH, collection_name=COLLECTION_NAME)

    for pdf_path in pdf_files:
        typer.echo(f"Embedding {typer.style(pdf_path.name, fg=typer.colors.CYAN)}...")
        for page_num, image in load_pdf(pdf_path):
            embedding = embedder.embed_document(image)
            store.index(embedding, str(pdf_path), page_num)

    typer.echo("")
    typer.echo(typer.style("Done.", fg=typer.colors.GREEN, bold=True))


@app.command()
def retrieve(
    question: str,
    top_k: int = typer.Option(3, help="Number of pages to retrieve"),
) -> None:
    embedder = ImageEmbedder()
    store = VectorStore(path=CHROMA_PATH, collection_name=COLLECTION_NAME)

    query_embedding = embedder.embed_query(question)
    pages = store.retrieve(query_embedding, top_k=top_k)

    _print_retrieved_pages(pages)


@app.command()
def query(
    question: str,
    top_k: int = typer.Option(1, help="Number of pages to retrieve"),
    model: str = typer.Option("qwen3-vl:4b", envvar="OLLAMA_MODEL", help="Ollama model to use"),
) -> None:
    embedder = ImageEmbedder()
    store = VectorStore(path=CHROMA_PATH, collection_name=COLLECTION_NAME)

    query_embedding = embedder.embed_query(question)
    pages = store.retrieve(query_embedding, top_k=top_k)

    _print_retrieved_pages(pages)
    typer.echo(typer.style("Answer:", bold=True))
    answer(question, [p.image for p in pages], model=model)
