import base64
import io
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool

from lmpkb.store.vector_store import RetrievedPage


@dataclass
class RetrieveToolBundle:
    tool: BaseTool
    execute: Callable[[str], tuple[list[dict[str, Any]], dict[str, Any], list[RetrievedPage]]]


def _page_to_image_url_block(page: RetrievedPage) -> dict[str, Any]:
    buf = io.BytesIO()
    page.image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def make_retrieve_tool(
    retrieve_fn: Callable[[str, int], list[RetrievedPage]],
    top_k: int,
) -> RetrieveToolBundle:
    def _run_retrieve(query: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[RetrievedPage]]:
        pages = retrieve_fn(query, top_k)
        image_blocks = [_page_to_image_url_block(page) for page in pages]
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"Retrieved {len(pages)} page(s):"},
            *image_blocks,
        ]
        artifact = {
            "query": query,
            "pages": [
                {
                    "source": page.source,
                    "page_number": page.page_number,
                }
                for page in pages
            ],
        }
        return content, artifact, pages

    @tool("retrieve", response_format="content_and_artifact")
    def retrieve(query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search the personal document knowledge base for one atomic information need.

        Use this tool whenever any required fact is missing, ambiguous, or ungrounded.
        For multi-hop questions, call in series with one query per hop.
        """
        content, artifact, _ = _run_retrieve(query)
        return content, artifact

    return RetrieveToolBundle(
        tool=retrieve,
        execute=_run_retrieve,
    )


def make_final_answer_tool() -> BaseTool:
    @tool("final_answer")
    def final_answer(answer: str) -> str:
        """Submit the final user-facing answer when all required evidence is gathered."""
        return answer

    return final_answer
