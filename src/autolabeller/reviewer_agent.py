from __future__ import annotations

from langchain_ollama import ChatOllama

from .config import ObjectClassConfig, OllamaConfig
from .dataset import llm_boxes_to_bounding_boxes
from .prompts import build_review_system_prompt, build_review_user_prompt
from .schemas import AnnotationResult, ImageRecord, LlmAnnotationResult, ReviewPayload, ReviewResult
from .utils import image_to_data_url


class ReviewAgent:
    def __init__(self, config: OllamaConfig, classes: list[ObjectClassConfig]):
        self.classes = classes
        self.class_names = [item.name for item in classes]
        self.model = ChatOllama(
            model=config.reviewer_model,
            base_url=config.base_url,
            temperature=config.temperature,
        ).with_structured_output(ReviewPayload)

    def review(
        self,
        record: ImageRecord,
        yolo_result: AnnotationResult,
        vlm_result: LlmAnnotationResult,
    ) -> ReviewResult:
        messages = [
            ("system", build_review_system_prompt()),
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": build_review_user_prompt(self.classes, yolo_result, vlm_result),
                    },
                    {"type": "image_url", "image_url": image_to_data_url(record.image_path)},
                ],
            ),
        ]
        payload = self.model.invoke(messages)
        return ReviewResult(
            final_boxes=llm_boxes_to_bounding_boxes(payload.final_objects, self.class_names),
            summary=payload.summary,
            missing_from_yolo=payload.missing_from_yolo,
            missing_from_vlm=payload.missing_from_vlm,
            suspicious_labels=payload.suspicious_labels,
        )
