from pathlib import Path
import sys

import typer

from lmpkb.agent import agent, ui
from lmpkb.agent.config import AgentConfig, resolve_agent_config, resolve_store_path
from lmpkb.agent.llm_factory import build_llm
from lmpkb.embed.embedder import ImageEmbedder
from lmpkb.ingest.pdf import count_pages, load_pdf
from lmpkb.session import clear_session, load_session, save_session
from lmpkb.store.vector_store import RetrievedPage, VectorStore

app = typer.Typer()

COLLECTION_NAME = "documents"
EXIT_COMMANDS = {"/bye"}
CLEAR_COMMANDS = {"/clear"}
HELP_COMMANDS = {"/help"}
VERBOSE_COMMANDS = {"/verbose"}


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


def _supports_ansi() -> bool:
    return sys.stdout.isatty()


def _erase_last_input_line() -> None:
    if not _supports_ansi():
        return
    # After input(), cursor is on the next line. Move up and clear the prompt line.
    sys.stdout.write("\033[1A\r\033[2K")
    sys.stdout.flush()


def _clear_terminal() -> None:
    if _supports_ansi():
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        return
    typer.echo("\n" * 40)


def _print_chat_banner() -> None:
    typer.echo(typer.style("lmpkb chat", bold=True))
    typer.echo(
        f"{typer.style('/help', fg=typer.colors.CYAN, bold=True)} for commands, "
        f"{typer.style('/bye', fg=typer.colors.CYAN, bold=True)} to exit."
    )
    typer.echo("")


def _run_chat_loop(agent_config: AgentConfig) -> None:
    embedder = ImageEmbedder()
    store = VectorStore(path=agent_config.store_path, collection_name=COLLECTION_NAME)
    llm = build_llm(agent_config)

    def retrieve_fn(q: str, k: int) -> list[RetrievedPage]:
        return store.retrieve(embedder.embed_query(q), top_k=k)

    history = load_session()
    verbose = False

    typer.echo("")
    _print_chat_banner()

    while True:
        try:
            question = input("You > ")
        except (EOFError, KeyboardInterrupt):
            typer.echo("")
            break

        question = question.strip()
        if not question:
            continue

        command = question.lower()
        if command in EXIT_COMMANDS:
            _erase_last_input_line()
            break
        if command in CLEAR_COMMANDS:
            _erase_last_input_line()
            clear_session()
            history = []
            _clear_terminal()
            _print_chat_banner()
            continue
        if command in HELP_COMMANDS:
            _erase_last_input_line()
            typer.echo("Commands:")
            typer.echo("  /help     show commands")
            typer.echo("  /clear    clear chat history")
            typer.echo("  /verbose  toggle reasoning/step traces")
            typer.echo("  /bye      exit chat")
            continue
        if command in VERBOSE_COMMANDS:
            _erase_last_input_line()
            verbose = not verbose
            state = "on" if verbose else "off"
            typer.echo(typer.style(f"Verbose {state}.", fg=typer.colors.GREEN))
            continue

        ui.print_agent_turn_header()

        try:
            messages = agent.run(
                question,
                retrieve_fn,
                top_k=agent_config.top_k,
                llm=llm,
                history=history,
                show_steps=verbose,
                show_reasoning=verbose,
            )
        except Exception as exc:
            typer.echo(typer.style(f"Error: {exc}", fg=typer.colors.RED, bold=True))
            typer.echo("")
            ui.print_agent_turn_footer()
            continue

        save_session(messages)
        history = load_session()
        ui.print_agent_turn_footer()

    typer.echo(typer.style("Bye.", fg=typer.colors.GREEN))


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    top_k: int | None = typer.Option(None, help="Number of pages to retrieve"),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file (defaults to lmpkb.yaml / lmpkb.yml if present)",
    ),
    model_type: str | None = typer.Option(
        None,
        help="Model provider: ollama or openai",
    ),
    model: str | None = typer.Option(
        None,
        help="Model name (defaults to provider-specific env var)",
    ),
    reasoning: str | None = typer.Option(
        None,
        help="Reasoning setting (ollama: true/false, openai: none|minimal|low|medium|high|xhigh)",
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    try:
        agent_config = resolve_agent_config(
            config_path=str(config_file) if config_file is not None else None,
            model_type=model_type,
            model=model,
            reasoning=reasoning,
            top_k=top_k,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _run_chat_loop(agent_config)


@app.command()
def embed(
    folder: Path,
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file (defaults to lmpkb.yaml / lmpkb.yml if present)",
    ),
) -> None:
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

    try:
        store_path = resolve_store_path(str(config_file) if config_file is not None else None)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    embedder = ImageEmbedder()
    store = VectorStore(path=store_path, collection_name=COLLECTION_NAME)

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
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file (defaults to lmpkb.yaml / lmpkb.yml if present)",
    ),
) -> None:
    try:
        store_path = resolve_store_path(str(config_file) if config_file is not None else None)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    embedder = ImageEmbedder()
    store = VectorStore(path=store_path, collection_name=COLLECTION_NAME)

    query_embedding = embedder.embed_query(question)
    pages = store.retrieve(query_embedding, top_k=top_k)

    _print_retrieved_pages(pages)
