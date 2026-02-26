from pathlib import Path
from typing import Any

from lmpkb.store.vector_store import RetrievedPage

_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREY = "\033[90m"
_RESET = "\033[0m"
_SEP = "──────────────────────────────────────────"


def print_step(step: int) -> None:
    print(f"\n{_BOLD}{_SEP}{_RESET}")
    print(f"{_BOLD}Step {step}{_RESET}")
    print(f"{_BOLD}{_SEP}{_RESET}\n")


def print_thinking_header() -> None:
    print(f"{_GREY}Thinking:{_RESET}")


def print_thinking_chunk(text: str) -> None:
    print(f"{_GREY}{text}{_RESET}", end="", flush=True)


def parse_reasoning(chunk: Any) -> str | None:
    additional_kwargs = getattr(chunk, "additional_kwargs", None) or {}
    reasoning_content = additional_kwargs.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content

    content = getattr(chunk, "content", None)
    if not isinstance(content, list):
        return None

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "reasoning":
            summary = block.get("summary")
            if isinstance(summary, list):
                for item in summary:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str) and text:
                            parts.append(text)
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
            continue

        if block_type in {"reasoning_content", "thinking"}:
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)

    if not parts:
        return None
    return "".join(parts)


def print_action_retrieve(query: str) -> None:
    print(f"{_BOLD}Action:{_RESET} retrieve  →  {_CYAN}{query}{_RESET}\n")


def print_action_answer() -> None:
    print(f"{_BOLD}Action:{_RESET} answer\n")


def print_retrieved_pages(pages: list[RetrievedPage]) -> None:
    print(f"{_BOLD}Retrieved pages:{_RESET}")
    for i, page in enumerate(pages, 1):
        print(
            f"  {_BOLD}{i}{_RESET}. "
            f"{_CYAN}{Path(page.source).name}{_RESET}"
            f"  —  page {page.page_number + 1}"
        )
    print()


def print_answer_header() -> None:
    print(f"{_BOLD}Answer:{_RESET}")
