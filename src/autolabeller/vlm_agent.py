from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import LlamaFactoryConfig, ObjectClassConfig
from .dataset import annotation_result_to_llm_result, load_yolo_annotation, validate_llm_annotation_result
from .llama_factory_client import LlamaFactoryChatClient
from .prompts import (
    build_annotation_example_user_prompt,
    build_annotation_system_prompt,
    build_annotation_user_prompt,
)
from .schemas import ImageRecord, LlmAnnotationResult
from .utils import image_to_data_url


class MultimodalAnnotatorAgent:
    def __init__(self, config: LlamaFactoryConfig, classes: list[ObjectClassConfig]):
        self.config = config
        self.classes = classes
        self.class_names = [item.name for item in classes]
        self.client = LlamaFactoryChatClient(config)

    def annotate(self, record: ImageRecord) -> LlmAnnotationResult:
        messages = [SystemMessage(content=build_annotation_system_prompt())]
        messages.extend(self._build_few_shot_messages())
        messages.append(
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": build_annotation_user_prompt(
                            self.classes,
                            few_shot_count=len(self.config.annotator_few_shots),
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(record.image_path)},
                    },
                ]
            )
        )
        result = self.client.invoke_structured(
            model_name=self.config.annotator_model,
            messages=messages,
            response_model=LlmAnnotationResult,
        )
        return validate_llm_annotation_result(result, self.class_names)

    def _build_few_shot_messages(self) -> list[HumanMessage | AIMessage]:
        messages: list[HumanMessage | AIMessage] = []
        for idx, example in enumerate(self.config.annotator_few_shots, start=1):
            if not example.image_path.exists():
                raise FileNotFoundError(f"Few-shot image not found: {example.image_path}")
            if not example.label_path.exists():
                raise FileNotFoundError(f"Few-shot label not found: {example.label_path}")
            annotation = load_yolo_annotation(
                label_path=example.label_path,
                class_names=self.class_names,
            )
            payload = annotation_result_to_llm_result(annotation)
            messages.append(
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": build_annotation_example_user_prompt(self.classes, example_index=idx),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(example.image_path)},
                        },
                    ]
                )
            )
            messages.append(AIMessage(content=json.dumps(payload.model_dump(), ensure_ascii=False)))
        return messages
