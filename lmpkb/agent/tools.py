import base64
import io
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool
from tavily import TavilyClient

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
        """Search the knowledge base by embedding a natural-language query and returning the most relevant pages as images.

        When to use:
        - Any time you need factual information before forming a conclusion
        - For multi-part questions, decompose into separate calls — one query per atomic information need
        - When uncertain about something, retrieve rather than speculate
        - If a retrieval result is weak or off-target, reformulate and retrieve again

        Query rules:
        - Each query must target exactly ONE specific information need
        - Be precise and concrete — avoid vague or compound queries
        - Ground queries in known entities from prior retrieval results
        - If ambiguous, run a disambiguation query first

        Call this tool as many times as needed to collect all required evidence.
        """
        content, artifact, _ = _run_retrieve(query)
        return content, artifact

    return RetrieveToolBundle(
        tool=retrieve,
        execute=_run_retrieve,
    )


def make_web_search_tool() -> BaseTool:
    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the web for current information not available in the knowledge base.

        When to use:
        - When the task requires up-to-date or real-world information
        - When the knowledge base returns weak or irrelevant results for a query
        - For facts, news, definitions, or general knowledge not in local documents

        Query rules:
        - Write a concise, specific query as you would type into a search engine
        - One query per call — decompose multi-part searches into separate calls
        - Prefer specific queries over broad ones for better results

        Returns a ranked list of web results with titles, URLs, and content snippets.
        """
        client = TavilyClient()
        response = client.search(query, max_results=5, search_depth="basic")
        results: list[dict[str, Any]] = response["results"]
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['url']}")
            lines.append(f"    {r['content']}")
        return "\n".join(lines)

    return web_search


def make_final_answer_tool() -> BaseTool:
    @tool("final_answer")
    def final_answer(answer: str) -> str:
        """Deliver the final answer to the user's request.

        Call this exactly once when you have collected sufficient evidence to fully address the request.

        When to use:
        - Only after all required facts are grounded in retrieved evidence
        - Never call this before retrieving relevant information
        - If any part of the answer would require guessing, call retrieve again first
        - Only when you can answer directly without uncertainty caveats like "unclear", "not found", or "insufficient"
        """
        return answer

    return final_answer
