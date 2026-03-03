from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from lmpkb.agent import ui
from lmpkb.agent.tools import make_final_answer_tool, make_retrieve_tool, make_web_search_tool
from lmpkb.store.vector_store import RetrievedPage

MAX_STEPS = 8

_SYSTEM = """\
You are an autonomous reasoning agent. You have access to a set of tools — \
read their descriptions carefully to understand what they do and when to use them.

Your job: understand the user's request, plan an approach using your available tools, \
and complete the task fully. Do not guess or fabricate — use your tools to gather \
what you need.\
"""


def _extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def run(
    task: str,
    retrieve_fn: Callable[[str, int], list[RetrievedPage]],
    top_k: int,
    llm: Any,
) -> None:
    retrieve_tool = make_retrieve_tool(retrieve_fn, top_k=top_k)
    web_search_tool = make_web_search_tool()
    final_answer_tool = make_final_answer_tool()

    llm_with_tools = llm.bind_tools(
        [retrieve_tool.tool, web_search_tool, final_answer_tool],
        tool_choice="any",
    )
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=task)]

    for step in range(1, MAX_STEPS + 1):
        ui.print_step(step)

        showed_thinking = False
        response = None
        for chunk in llm_with_tools.stream(messages):
            reasoning_text = ui.parse_reasoning(chunk)
            if reasoning_text:
                if not showed_thinking:
                    ui.print_thinking_header()
                    showed_thinking = True
                ui.print_thinking_chunk(reasoning_text)
            response = chunk if response is None else response + chunk
        if showed_thinking:
            print("\n")

        messages.append(response)

        if not response.tool_calls:
            answer_text = _extract_text_content(response.content)
            if not answer_text:
                raise RuntimeError("Model returned plain text with no tool call and empty content.")
            ui.print_action_answer()
            ui.print_answer_header()
            print(answer_text)
            print()
            break

        answered = False
        tool_messages: list[ToolMessage] = []

        for tc in response.tool_calls:
            tc_name = tc["name"]

            if tc_name == "retrieve":
                query = tc["args"]["query"]
                ui.print_action_retrieve(query)

                content, artifact, pages = retrieve_tool.execute(query)
                ui.print_retrieved_pages(pages)

                tool_messages.append(ToolMessage(
                    content=content,
                    artifact=artifact,
                    name="retrieve",
                    tool_call_id=tc["id"],
                ))

            elif tc_name == "web_search":
                query = tc["args"]["query"]
                ui.print_action_web_search(query)

                result = web_search_tool.invoke(tc["args"])
                tool_messages.append(ToolMessage(
                    content=result,
                    name="web_search",
                    tool_call_id=tc["id"],
                ))

            elif tc_name == "final_answer":
                answer_text = tc["args"]["answer"].strip()
                ui.print_action_answer()
                ui.print_answer_header()
                if answer_text:
                    print(answer_text)
                print()
                tool_messages.append(ToolMessage(
                    content=f"Final answer delivered: {answer_text}",
                    artifact={"answer": answer_text},
                    name="final_answer",
                    tool_call_id=tc["id"],
                ))
                answered = True

            else:
                raise ValueError(f"Unknown tool called: {tc_name!r}")

        messages.extend(tool_messages)

        if answered:
            break
    else:
        raise RuntimeError(f"Agent reached max steps ({MAX_STEPS}) without answering")
