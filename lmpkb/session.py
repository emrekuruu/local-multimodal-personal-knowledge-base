import json
import os
from pathlib import Path

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict

def _session_file() -> Path:
    config = os.environ.get("LMPKB_CONFIG")
    if config:
        return Path(config).parent / "session.json"
    return Path.cwd() / "session.json"


def _strip_artifacts(messages: list[BaseMessage]) -> list[BaseMessage]:
    result = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.artifact is not None:
            result.append(ToolMessage(
                content=msg.content,
                name=msg.name,
                tool_call_id=msg.tool_call_id,
            ))
        else:
            result.append(msg)
    return result


def load_session() -> list[BaseMessage]:
    path = _session_file()
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return messages_from_dict(data)


def save_session(messages: list[BaseMessage]) -> None:
    path = _session_file()
    saveable = [m for m in messages if not isinstance(m, SystemMessage)]
    saveable = _strip_artifacts(saveable)
    path.write_text(json.dumps(messages_to_dict(saveable), indent=2))


def clear_session() -> None:
    path = _session_file()
    if path.exists():
        path.unlink()
