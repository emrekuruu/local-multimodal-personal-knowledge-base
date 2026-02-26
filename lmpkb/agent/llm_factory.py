from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from lmpkb.agent.config import AgentConfig


def build_llm(config: AgentConfig) -> Any:
    if config.model_type == "ollama":
        return ChatOllama(
            model=config.model,
            reasoning=config.reasoning,
        )

    kwargs: dict[str, Any] = {
        "model": config.model,
        "output_version": "responses/v1",
    }
    if config.reasoning is not None:
        kwargs["reasoning"] = config.reasoning

    return ChatOpenAI(**kwargs)
