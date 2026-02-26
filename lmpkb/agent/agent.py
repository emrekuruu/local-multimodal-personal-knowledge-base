import base64
import io

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from PIL.Image import Image


def answer(question: str, images: list[Image], model: str) -> str:
    llm = ChatOllama(model=model, think=False)

    image_parts = []
    for img in images:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    message = HumanMessage(content=[{"type": "text", "text": question}, *image_parts])
    return llm.invoke([message]).content
