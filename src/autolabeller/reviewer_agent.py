from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .config import LlamaFactoryConfig, ObjectClassConfig
from .dataset import llm_boxes_to_bounding_boxes
from .llama_factory_client import LlamaFactoryChatClient
from .prompts import build_review_system_prompt, build_review_user_prompt
from .schemas import AnnotationResult, ImageRecord, LlmAnnotationResult, ReviewPayload, ReviewResult
from .utils import image_to_data_url


class ReviewerAgent:
    def __init__(self, config: LlamaFactoryConfig, classes: list[ObjectClassConfig]):
        self.classes = classes
        self.class_names = [item.name for item in classes]
        self.config = config
        self.client = LlamaFactoryChatClient(config)

    def review(
        self,
        record: ImageRecord,
        yolo_result: AnnotationResult,
        vlm_result: LlmAnnotationResult,
    ) -> ReviewResult:
        messages = [
            SystemMessage(content=build_review_system_prompt()),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": build_review_user_prompt(self.classes, yolo_result, vlm_result),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(record.image_path)},
                    },
                ]
            ),
        ]
        payload = self.client.invoke_structured(
            model_name=self.config.reviewer_model,
            messages=messages,
            response_model=ReviewPayload,
        )
        return ReviewResult(
            final_boxes=llm_boxes_to_bounding_boxes(payload.final_objects, self.class_names),
            has_issues=payload.has_issues,
            issue_summary=payload.issue_summary,
        )
