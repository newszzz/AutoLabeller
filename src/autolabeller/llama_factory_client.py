from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import LlamaFactoryConfig
from .utils import extract_json_object


class LlamaFactoryChatClient:
    def __init__(self, config: LlamaFactoryConfig):
        self.config = config

    def invoke_structured(
        self,
        model_name: str,
        messages: list[BaseMessage],
        response_model: type[BaseModel],
    ) -> BaseModel:
        model = ChatOpenAI(
            model=model_name,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            timeout=self.config.request_timeout,
        )
        response = model.invoke(messages)
        content = _coerce_text_content(response)
        payload = extract_json_object(content)
        return response_model.model_validate(payload)


def _coerce_text_content(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        text_chunks: list[str] = []
        for item in message.content:
            if isinstance(item, str):
                text_chunks.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_chunks.append(text)
        return "\n".join(text_chunks).strip()
    raise ValueError("Model response did not contain text content.")
