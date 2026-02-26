from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from lmpkb.agent import ui
from lmpkb.agent.tools import make_final_answer_tool, make_retrieve_tool
from lmpkb.store.vector_store import RetrievedPage

MAX_STEPS = 8

_SYSTEM = """\
<role>
You are a retrieval-first research assistant for a personal document knowledge base.
</role>

<objective>
Answer only from evidence found via the retrieve tool. For multi-hop questions, gather evidence hop-by-hop.
</objective>

<actions>
- Action 1: retrieve(query) for document evidence.
- Action 2: final_answer(answer) when all required evidence is gathered.
- Use exactly one action per turn.
- You must explicitly choose one action before producing any user-facing text.
</actions>

<multi_hop_policy>
- Decompose the question into atomic unknowns.
- If an attribute depends on identifying an entity, do entity identification first, then attribute lookup.
- If the question links entities (A -> B -> C), retrieve each link in separate hops.
- Do not answer while any required link or value is unresolved.
- For linked multi-hop questions, expect at least two retrieval hops unless one retrieval result explicitly contains all links with clear evidence.
</multi_hop_policy>

<query_rules>
- Each retrieve query must target exactly one information need.
- Keep queries specific, short, and grounded in known entities from prior hops.
- If ambiguous, run a disambiguation query first.
- If a retrieval result is weak or empty, reformulate and retrieve again.
</query_rules>

<decision_policy>
- For every step, first decide internally which action is needed: retrieve or answer.
- Enter ANSWER MODE only when all required facts are grounded.
- In ANSWER MODE, call final_answer(answer).
- If your intended output includes uncertainty or missing-context caveats, do NOT answer. Use retrieve again with a refined query.
</decision_policy>

<uncertainty_policy>
- Do not overthink.
- If you are not sure, retrieve again instead of guessing.
</uncertainty_policy>

<action_selection_checklist>
Before choosing Action 3 (answer), all must be true:
- The target entity is uniquely identified.
- The requested metric/value/fact is present in retrieved evidence.
- No key dependency is missing.
- You can answer directly without phrases like "unclear", "not found", "insufficient", or "please clarify".
If any check fails:
- Use Action 1 (retrieve) with a narrower query.
</action_selection_checklist>

<stop_rule>
Only provide a final answer when all required unknowns are grounded in retrieved evidence.
If evidence is still missing, call retrieve again.
- If you cannot answer confidently from retrieved evidence, do not use Action 3.
</stop_rule>

<behavior_notes>
- Do not rely on memory for document facts.
- Prefer iterative retrieval over one-shot answering.
- Prefer asking the user over speculative reasoning when uncertain.
- Keep thinking separate from final answer text.
- When done, answer directly and concisely.
</behavior_notes>

<nice_to_have>
- ! Do not think too much. When you decide to do any of the actions, just do it. Do not hesitate or overthink. !
- ! Think VERY LITTLE. If you have an idea about an action to take, just take it. Do not worry about whether it's the "best" action. You can always course correct in the next step. !
</nice_to_have>
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
    question: str,
    retrieve_fn: Callable[[str, int], list[RetrievedPage]],
    top_k: int,
    llm: Any,
) -> None:
    retrieve_tool = make_retrieve_tool(retrieve_fn, top_k=top_k)
    final_answer_tool = make_final_answer_tool()

    llm_with_tools = llm.bind_tools(
        [retrieve_tool.tool, final_answer_tool],
        tool_choice="any",
    )
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=question)]

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

        if len(response.tool_calls) != 1:
            raise RuntimeError("Model must choose exactly one action per turn")

        tc = response.tool_calls[0]
        tc_name = tc["name"]

        if tc_name == "retrieve":
            query = tc["args"]["query"]
            ui.print_action_retrieve(query)

            content, artifact, pages = retrieve_tool.execute(query)
            ui.print_retrieved_pages(pages)

            messages.append(ToolMessage(
                content=content,
                artifact=artifact,
                name="retrieve",
                tool_call_id=tc["id"],
            ))
            continue

        if tc_name == "final_answer":
            answer_text = tc["args"]["answer"].strip()
            ui.print_action_answer()
            ui.print_answer_header()
            if answer_text:
                print(answer_text)
            print()
            messages.append(ToolMessage(
                content=f"Final answer delivered: {answer_text}",
                artifact={"answer": answer_text},
                name="final_answer",
                tool_call_id=tc["id"],
            ))
            break

        raise ValueError(f"Unknown tool called: {tc_name!r}")
    else:
        raise RuntimeError(f"Agent reached max steps ({MAX_STEPS}) without answering")
