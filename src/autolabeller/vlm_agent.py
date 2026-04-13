from __future__ import annotations

import json

from langchain_ollama import ChatOllama

from .config import ObjectClassConfig, OllamaConfig
from .dataset import annotation_result_to_llm_payload, load_yolo_annotation
from .prompts import (
    build_annotation_example_user_prompt,
    build_annotation_system_prompt,
    build_annotation_user_prompt,
)
from .schemas import AnnotationResult, BoundingBox, ImageRecord, LlmAnnotationPayload
from .utils import image_to_data_url


class MultimodalAnnotatorAgent:
    def __init__(self, config: OllamaConfig, classes: list[ObjectClassConfig]):
        self.config = config
        self.classes = classes
        self.class_names = [item.name for item in classes]
        self.model = ChatOllama(
            model=config.annotator_model,
            base_url=config.base_url,
            temperature=config.temperature,
        ).with_structured_output(LlmAnnotationPayload)

    def annotate(self, record: ImageRecord) -> AnnotationResult:
        messages: list[tuple[str, str | list[dict[str, str]]]] = [("system", build_annotation_system_prompt())]
        messages.extend(self._build_few_shot_messages())
        messages.append(
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": build_annotation_user_prompt(
                            self.classes,
                            few_shot_count=len(self.config.annotator_few_shots),
                        ),
                    },
                    {"type": "image_url", "image_url": image_to_data_url(record.image_path)},
                ],
            )
        )
        payload = self.model.invoke(messages)
        return AnnotationResult(
            image_path=record.image_path,
            boxes=[
                BoundingBox(
                    label=item.label,
                    x_center=item.bbox[0],
                    y_center=item.bbox[1],
                    width=item.bbox[2],
                    height=item.bbox[3],
                    confidence=item.confidence,
                    rationale=item.rationale,
                    source="vlm",
                )
                for item in payload.objects
            ],
            source="vlm",
            summary=payload.summary,
            issues=payload.issues,
        )

    def _build_few_shot_messages(self) -> list[tuple[str, str | list[dict[str, str]]]]:
        messages: list[tuple[str, str | list[dict[str, str]]]] = []
        for idx, example in enumerate(self.config.annotator_few_shots, start=1):
            if not example.image_path.exists():
                raise FileNotFoundError(f"Few-shot image not found: {example.image_path}")
            if not example.label_path.exists():
                raise FileNotFoundError(f"Few-shot label not found: {example.label_path}")
            annotation = load_yolo_annotation(
                image_path=example.image_path,
                label_path=example.label_path,
                class_names=self.class_names,
                source="ground_truth",
            )
            payload = annotation_result_to_llm_payload(annotation)
            messages.append(
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": build_annotation_example_user_prompt(self.classes, example_index=idx),
                        },
                        {"type": "image_url", "image_url": image_to_data_url(example.image_path)},
                    ],
                )
            )
            messages.append(("assistant", json.dumps(payload.model_dump(), ensure_ascii=False)))
        return messages
