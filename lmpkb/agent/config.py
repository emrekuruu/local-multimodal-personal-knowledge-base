import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

ModelType = Literal["ollama", "openai"]

_OPENAI_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_OPENAI_REASONING_SUMMARIES = {"auto", "concise", "detailed"}
_DEFAULT_CONFIG_FILES = ("lmpkb.yaml", "lmpkb.yml", ".lmpkb.yaml", ".lmpkb.yml")
_ENV_CONFIG_VAR = "LMPKB_CONFIG"


@dataclass(frozen=True)
class AgentConfig:
    model_type: ModelType
    model: str
    reasoning: bool | dict[str, str] | None
    top_k: int
    store_path: str


def _parse_bool(raw: str, *, field_name: str) -> bool:
    value = raw.strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{field_name} must be a boolean (true/false)")


def _load_yaml_config(config_path: str | None) -> tuple[Path | None, dict]:
    resolved_path: Path | None = None
    if config_path:
        resolved_path = Path(config_path).resolve()
        if not resolved_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
    else:
        env_path = os.environ.get(_ENV_CONFIG_VAR)
        if env_path:
            resolved_path = Path(env_path).resolve()
            if not resolved_path.exists():
                raise ValueError(f"Config file from {_ENV_CONFIG_VAR} not found: {env_path}")
        else:
            for candidate in _DEFAULT_CONFIG_FILES:
                candidate_path = Path(candidate)
                if candidate_path.exists():
                    resolved_path = candidate_path.resolve()
                    break

    if resolved_path is None:
        return None, {}

    data = yaml.safe_load(resolved_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping: {resolved_path}")
    return resolved_path, data


def _resolve_path_from_yaml(raw: object, yaml_path: Path | None) -> str:
    p = Path(str(raw))
    if not p.is_absolute() and yaml_path is not None:
        p = yaml_path.parent / p
    return str(p.resolve())


def resolve_store_path(config_path: str | None) -> str:
    yaml_path, data = _load_yaml_config(config_path)
    raw = _yaml_get(data, "store", "path")
    if raw is None:
        raise ValueError("store.path is required (set in YAML: store.path)")
    return _resolve_path_from_yaml(raw, yaml_path)


def _yaml_get(data: dict, *keys: str):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _normalize_model_type(raw: object) -> str:
    return str(raw).strip().lower()


def resolve_agent_config(
    *,
    config_path: str | None,
    model_type: str | None,
    model: str | None,
    reasoning: str | None,
    top_k: int | None,
) -> AgentConfig:
    yaml_path, yaml_cfg = _load_yaml_config(config_path)

    yaml_model_type = _yaml_get(yaml_cfg, "model", "type")
    if yaml_model_type is None:
        yaml_model_type = yaml_cfg.get("model_type")
    raw_model_type = model_type or yaml_model_type
    if raw_model_type is None:
        raise ValueError("model_type is required (set via CLI or YAML: model.type)")
    normalized_model_type = _normalize_model_type(raw_model_type)
    if normalized_model_type not in {"ollama", "openai"}:
        raise ValueError("MODEL_TYPE must be one of: ollama, openai")

    yaml_top_k = _yaml_get(yaml_cfg, "retrieve", "top_k")
    if yaml_top_k is None:
        yaml_top_k = yaml_cfg.get("top_k")
    raw_top_k = top_k if top_k is not None else yaml_top_k
    if raw_top_k is None:
        raise ValueError("top_k is required (set via CLI or YAML: retrieve.top_k)")
    try:
        resolved_top_k = int(raw_top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("retrieve amount (top_k) must be an integer") from exc
    if resolved_top_k < 1:
        raise ValueError("retrieve amount (top_k) must be >= 1")

    raw_store_path = _yaml_get(yaml_cfg, "store", "path")
    if raw_store_path is None:
        raise ValueError("store.path is required (set in YAML: store.path)")
    resolved_store_path = _resolve_path_from_yaml(raw_store_path, yaml_path)

    yaml_model = _yaml_get(yaml_cfg, "model", "name")
    if yaml_model is None:
        yaml_model = yaml_cfg.get("model")
    yaml_reasoning = _yaml_get(yaml_cfg, "model", "reasoning")
    if yaml_reasoning is None:
        yaml_reasoning = yaml_cfg.get("reasoning")

    if normalized_model_type == "ollama":
        resolved_model = model or yaml_model
        if resolved_model is None:
            raise ValueError("model name is required for ollama (set via CLI --model or YAML: model.name)")
        raw_reasoning: object = reasoning if reasoning is not None else yaml_reasoning
        if raw_reasoning is None:
            raise ValueError(
                "ollama reasoning is required (set via CLI --reasoning true/false or YAML: model.reasoning)"
            )
        if isinstance(raw_reasoning, bool):
            resolved_reasoning = raw_reasoning
        else:
            resolved_reasoning = _parse_bool(str(raw_reasoning), field_name="OLLAMA_REASONING")
        return AgentConfig(
            model_type="ollama",
            model=resolved_model,
            reasoning=resolved_reasoning,
            top_k=resolved_top_k,
            store_path=resolved_store_path,
        )

    resolved_model = model or yaml_model
    if resolved_model is None:
        raise ValueError("model name is required for openai (set via CLI --model or YAML: model.name)")

    yaml_effort: object | None = None
    yaml_summary: object | None = None
    if isinstance(yaml_reasoning, dict):
        yaml_effort = yaml_reasoning.get("effort")
        yaml_summary = yaml_reasoning.get("summary")
    elif yaml_reasoning is not None:
        yaml_effort = yaml_reasoning

    effort_raw = reasoning if reasoning is not None else yaml_effort
    if effort_raw is None:
        raise ValueError(
            "openai reasoning effort is required "
            "(set via CLI --reasoning or YAML: model.reasoning.effort)"
        )
    effort = str(effort_raw).strip().lower()
    if effort not in _OPENAI_REASONING_EFFORTS:
        raise ValueError(
            "OPENAI_REASONING_EFFORT must be one of: none, minimal, low, medium, high, xhigh"
        )
    if effort == "none":
        resolved_reasoning = None
    else:
        resolved_reasoning = {"effort": effort}
        if yaml_summary is not None:
            summary = str(yaml_summary).strip().lower()
            if summary not in _OPENAI_REASONING_SUMMARIES:
                raise ValueError("OPENAI_REASONING_SUMMARY must be one of: auto, concise, detailed")
            resolved_reasoning["summary"] = summary

    return AgentConfig(
        model_type="openai",
        model=resolved_model,
        reasoning=resolved_reasoning,
        top_k=resolved_top_k,
        store_path=resolved_store_path,
    )
