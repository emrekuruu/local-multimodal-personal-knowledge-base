import re
from pathlib import Path
from typing import Any

from lmpkb.store.vector_store import RetrievedPage

_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREY = "\033[90m"
_ORANGE = "\033[38;5;208m"
_RESET = "\033[0m"


def print_agent_turn_header() -> None:
    print(f"\n{_BOLD}{_CYAN}Assistant{_RESET}")


def print_agent_turn_footer() -> None:
    print()


def print_step(step: int) -> None:
    print(f"{_GREY}  step {step}{_RESET}")


def print_thinking_header() -> None:
    print(f"{_GREY}  thinking: {_RESET}", end="", flush=True)


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
    print(f"{_GREY}  • retrieve: {query}{_RESET}")


def print_action_web_search(query: str) -> None:
    print(f"{_GREY}  • web_search: {query}{_RESET}")


def print_action_code_exec(code: str) -> None:
    print(f"{_GREY}  • code_exec{_RESET}")
    for line in code.splitlines():
        print(f"{_GREY}    {line}{_RESET}")
    print()


def print_action_answer() -> None:
    return


def print_retrieved_pages(pages: list[RetrievedPage]) -> None:
    print(f"{_GREY}    sources:{_RESET}")
    for i, page in enumerate(pages, 1):
        print(
            f"{_GREY}      {i}. {Path(page.source).name} (page {page.page_number + 1}){_RESET}"
        )
    print()


def print_answer_header() -> None:
    return


def _render_markdown_bold(text: str) -> str:
    # Robust inline bold parsing without relying on complex regex edge cases.
    parts = text.split("**")
    if len(parts) < 3:
        return text

    rendered: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1 and part:
            rendered.append(f"{_ORANGE}{_BOLD}{part}{_RESET}")
        else:
            rendered.append(part)
    return "".join(rendered)


def print_assistant_text(text: str) -> None:
    print(_render_markdown_bold(text))
