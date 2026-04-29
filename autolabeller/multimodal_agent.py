from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import ModelApiConfig, ObjectClassConfig
from .dataset import validate_llm_annotation_result
from .prompts import (
    build_annotation_system_prompt,
    build_annotation_user_content,
    build_review_system_prompt,
    build_review_user_content,
)
from .schemas import AnnotationResult, ImageRecord, LlmAnnotationResult, ReviewResult


class MultimodalAgent:
    def __init__(
        self,
        config: ModelApiConfig,
        classes: list[ObjectClassConfig],
        few_shot_visual_dir: Path,
    ):
        self.config = config
        self.classes = classes
        self.class_names = [item.name for item in classes]
        self.few_shot_visual_dir = few_shot_visual_dir
        self.client = self._chat()

    def annotate(
        self,
        record: ImageRecord,
        yolo_result: AnnotationResult,
        yolo_image_path: Path,
    ) -> LlmAnnotationResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_annotation_system_prompt()},
            {
                "role": "user",
                "content": build_annotation_user_content(
                    classes=self.classes,
                    target_image_path=record.image_path,
                    yolo_result=yolo_result,
                    yolo_image_path=yolo_image_path,
                    few_shots=self.config.annotator_few_shots,
                    few_shot_visual_dir=self.few_shot_visual_dir,
                ),
            },
        ]
        result = self._invoke_structured(messages, LlmAnnotationResult)
        return validate_llm_annotation_result(
            result,
            self.class_names,
            record.width,
            record.height,
        )

    def review(
        self,
        record: ImageRecord,
        yolo_result: AnnotationResult,
        yolo_image_path: Path,
        llm_result: LlmAnnotationResult,
        llm_image_path: Path,
    ) -> ReviewResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_review_system_prompt()},
            {
                "role": "user",
                "content": build_review_user_content(
                    classes=self.classes,
                    target_image_path=record.image_path,
                    yolo_result=yolo_result,
                    yolo_image_path=yolo_image_path,
                    llm_result=llm_result,
                    llm_image_path=llm_image_path,
                ),
            },
        ]
        return self._invoke_structured(messages, ReviewResult)

    def _invoke_structured(
        self,
        messages: list[dict[str, Any]],
        schema_model: type[BaseModel],
    ):
        try:
            response = self.client.bind(response_format=_json_schema_format(schema_model)).invoke(messages)
        except Exception as exc:
            if not _is_response_format_error(exc):
                raise
            response = self.client.bind(response_format={"type": "json_object"}).invoke(messages)
        return _parse_response(response.content, schema_model)

    def _chat(self) -> ChatOpenAI:
        api_key = self.config.api_key or ("ollama" if self.config.backend == "ollama" else "0")
        if self.config.base_url is None or self.config.model is None:
            raise ValueError("llm_api.base_url and llm_api.model are required.")
        return ChatOpenAI(
            model=self.config.model,
            api_key=api_key,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            timeout=self.config.request_timeout,
            max_retries=0,
        )


def _json_schema_format(schema_model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_model.__name__,
            "schema": schema_model.model_json_schema(),
        },
    }


def _parse_response(content: Any, schema_model: type[BaseModel]):
    text = _extract_text(content)
    try:
        return schema_model.model_validate_json(text)
    except Exception:
        return schema_model.model_validate(_extract_json_object(text))


def _is_response_format_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "json_schema" in message


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        parts = candidate.split("```")
        if len(parts) >= 3:
            candidate = parts[1].strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Model response did not contain a valid JSON object: {text}")

