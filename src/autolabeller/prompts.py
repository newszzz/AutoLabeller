from __future__ import annotations

import json

from .config import ObjectClassConfig
from .dataset import build_class_catalog_text
from .schemas import AnnotationResult, LlmAnnotationResult


def build_annotation_system_prompt() -> str:
    return (
        "You are an image annotation assistant. Understand the image and produce structured"
        " object annotations that follow the required schema. Return exactly one JSON object."
        " Do not output markdown or chain-of-thought."
    )


def build_annotation_example_user_prompt(classes: list[ObjectClassConfig], example_index: int) -> str:
    class_catalog = build_class_catalog_text(classes)
    return f"""
This is few-shot example #{example_index}.
Study the image and the corresponding gold annotation format.

Class definitions:
{class_catalog}

Requirements:
1. Labels must exactly match the configured class names.
2. Bounding boxes use normalized YOLO coordinates with fields x_center, y_center, width, height.
3. Output contains all valid objects and no duplicates.
4. The assistant response for this example is the correct gold annotation.
""".strip()


def build_annotation_user_prompt(classes: list[ObjectClassConfig], few_shot_count: int) -> str:
    class_catalog = build_class_catalog_text(classes)
    few_shot_hint = (
        f"Use the {few_shot_count} few-shot examples above as reference for label semantics and annotation style."
        if few_shot_count
        else ""
    )
    return f"""
Annotation task:
Identify every object in the image that belongs to the configured classes and return normalized bounding boxes.

Class definitions:
{class_catalog}

{few_shot_hint}

Output requirements:
1. Return exactly one JSON object that fits the required structured schema.
2. All coordinates must be normalized to the range [0, 1].
3. Use fields x_center, y_center, width, height for each object.
4. label must exactly match one configured class name.
5. Do not annotate the same object multiple times.
6. If an object is ambiguous, occluded, or very small, mention that in issues.
7. If no configured objects are present, return an empty objects list and explain that in summary.
""".strip()


def build_review_system_prompt() -> str:
    return (
        "You are an image annotation reviewer. Compare the image itself, the YOLO proposal,"
        " and the multimodal annotation result, then produce the final structured annotation"
        " together with a compact issue diagnosis."
        " Return exactly one JSON object and do not output markdown or chain-of-thought."
    )


def build_review_user_prompt(
    classes: list[ObjectClassConfig],
    yolo_result: AnnotationResult,
    vlm_result: LlmAnnotationResult,
) -> str:
    class_catalog = build_class_catalog_text(classes)
    yolo_json = json.dumps(_annotation_to_prompt_payload(yolo_result), ensure_ascii=False, indent=2)
    vlm_json = json.dumps(vlm_result.model_dump(), ensure_ascii=False, indent=2)
    return f"""
Review task:
Use the image, the YOLO proposal, and the multimodal result to produce the final annotation.

Class definitions:
{class_catalog}

YOLO result:
{yolo_json}

Multimodal result:
{vlm_json}

Output requirements:
1. Validate the image itself instead of blindly trusting either source.
2. Remove duplicates, obvious mistakes, and undefined labels.
3. Keep only the final correct annotations in final_objects.
4. Set has_issues to true if either candidate input contains missing boxes, extra boxes, wrong labels, or other notable problems.
5. Fill issue_summary with one short explanation of the main problem, or say that no obvious issues were found.
6. Use fields x_center, y_center, width, height, all values in [0, 1].
7. label must exactly match one configured class name.
""".strip()


def _annotation_to_prompt_payload(result: AnnotationResult) -> dict:
    return {
        "summary": result.summary,
        "issues": result.issues,
        "objects": [
            {
                "label": box.label,
                "x_center": box.x_center,
                "y_center": box.y_center,
                "width": box.width,
                "height": box.height,
                "confidence": box.confidence,
            }
            for box in result.boxes
        ],
    }
